import sys

sys.path.append("/root/autodl-tmp/ClearSight/LLaVA")
import llava
import argparse
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

import os
import os.path as osp
import json
from tqdm import tqdm
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.model.builder import load_pretrained_model

from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from PIL import Image
import math
import yaml
from types import SimpleNamespace
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def compute_candidate_saliency(
    model, 
    seq: torch.Tensor, 
    candidate_id: torch.Tensor, 
    images: Optional[torch.Tensor],
    sys_token_len: int,
    img_token_len: int,
    target_layers: list
) -> float:
    """
    计算单个候选 token 的显著性 S(c)（对齐论文 Eq.(6)）：
      - 用完整序列（含候选）前向，拿 attention（真实展开后的 T×T）
      - 以候选位置的 query 行，取其对历史输出 J 的显著性均值，跨目标层再平均
      - 显著性矩阵 = |Attn ⊙ Grad(Attn)|，再按头平均 + L2 归一化（论文 Eq.(4)(5)）
    """
    device = seq.device
    #将候选拼到末尾（P 位置）
    cand = candidate_id.view(1, 1).to(device)
    seq_with_cand = torch.cat([seq, cand], dim=-1)  #(1, P+1)

    #用完整序列做一次前向，取 attentions（注意：这里的 T 是真实展开后长度）
    #关键：use_cache=False，保证返回完整 T×T 注意力矩阵
    torch.set_grad_enabled(True)
    outputs = model(
        input_ids=seq_with_cand,
        images=images,
        use_cache=False,
        output_attentions=True,
        return_dict=True
    )

    #用候选位置（P）的 logits 做损失，得到对 Attn 的梯度
    #logits 的第 -2 位对应位置 P 的预测（LM 预测下一个 token）
    logits_P = outputs.logits[:, -2, :]        #(1, V)
    logp_P = F.log_softmax(logits_P, dim=-1)
    loss = -logp_P[0, candidate_id.item()]     #标量

    total_sal = 0.0
    L = len(target_layers)

    #真实注意力长度（已包含图像展开）
    #通常是 (1, H, T, T)
    #用它来构造掩码和索引，不能再用 seq_with_cand.shape[-1]
    for l in target_layers:
        attn = outputs.attentions[l]                #(1, H, T, T)
        #让autograd知道我们要对这个张量求梯度
        attn = attn.requires_grad_(True)
        attn.retain_grad()

        #dL/dAttn
        grad_attn = torch.autograd.grad(loss, attn, retain_graph=True, allow_unused=False)[0]  #(1, H, T, T)

        #显著性矩阵：
        S = (attn * grad_attn).abs()                #(1, H, T, T)

        #下三角因果掩码（用注意力的真实 T 来造）
        T = attn.shape[-1]
        tril_mask = torch.tril(torch.ones((T, T), device=device, dtype=S.dtype)).view(1,1,T,T)
        S = S * tril_mask                           #(1, H, T, T)

        #跨头平均 + L2 归一化（对最后一维做范数）
        S_head_avg = S.mean(dim=1)                  #(1, T, T)
        #避免除零
        denom = torch.clamp(torch.norm(S_head_avg, p=2, dim=-1, keepdim=True), min=1e-9)
        S_bar = S_head_avg / denom                  #(1, T, T)

        #索引：候选位置的 query 行在最后一行（真实 T 的最后一位）
        q_idx = T - 1

        #历史输出 J：从系统+图像之后到候选之前（不含候选自身）
        #注意：这里用真实 T（含展开图像），因此 J_start = sys + img
        J_start = sys_token_len + img_token_len
        J_end_excl = T - 1                          #不包含候选自身
        if J_start >= J_end_excl:
            #没有历史输出（通常只有在第一个输出 token 时出现）
            layer_sal = 1.0
        else:
            J_idx = torch.arange(J_start, J_end_excl, device=device)
            #候选行对历史列的显著性均值
            layer_sal = S_bar[0, q_idx, J_idx].mean().item()

        total_sal += layer_sal

    #跨层平均（论文 Eq.(6) 外层平均）
    cand_sal = total_sal / max(L, 1)

    #清理梯度
    model.zero_grad(set_to_none=True)
    torch.set_grad_enabled(False)

    return float(cand_sal)

@torch.no_grad()
def sgrs_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    images: Optional[torch.Tensor],
    args,
    stop_ids: Optional[set] = None
) -> torch.Tensor:
    """
    SGRS 生成逻辑（Algorithm 1）：
      Top-K → 显著性过滤（R 轮重采样）→ 全部被拒 Fallback 取最大显著性
    修复点：
      - 先用完整序列预热一次 forward，建立 past_key_values
      - 之后增量前向显式传入 attention_mask，避免 LLaVA 内部对 None 做 shape 访问
    """
    device = input_ids.device
    model.eval()
    seq = input_ids

    #-------- 预热前向：建立 cache，并拿到首个 logits --------
    warmout = model(
        input_ids=seq,
        images=images,
        use_cache=True,
        output_attentions=False,
        return_dict=True
    )
    past_key_values = warmout.past_key_values
    logits = warmout.logits  #(1, T, V)

    accepted_sals: list[float] = []

    def _get_past_len(pkv):
        #对于 llama 系列：pkv[layer][0] 是 key: (B, n_head, past_len, head_dim)
        return pkv[0][0].shape[-2]

    for _ in range(args.max_new_tokens):
        #-------- 基于当前 logits 取下一步分布 --------
        step_logits = logits[:, -1, :]  #(1, V)
        probs = F.softmax(step_logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, k=args.sgrs_top_k, dim=-1)  #(1, K)
        topk_probs = topk_probs.squeeze(0)   #(K,)
        topk_idx = topk_idx.squeeze(0)       #(K,)
        candidates = topk_idx.tolist()

        #τ = α * 最近 W 个已接受 token 的显著性均值；首步无历史则 τ=0
        if len(accepted_sals) == 0:
            tau = 0.0
        else:
            window = accepted_sals[-args.sgrs_history_window:] if args.sgrs_history_window > 0 else accepted_sals
            tau = args.sgrs_alpha * (sum(window) / len(window))

        accepted = False
        chosen_id = None
        best_sal = -1e9

        #Top-K重采样
        remaining = candidates.copy()

        def _probs_for_remaining(rem):
            idxs = [candidates.index(x) for x in rem]
            p = topk_probs[idxs]
            p = p / p.sum()
            return p

        for _ in range(args.sgrs_max_resample):
            if not remaining:
                break
            p = _probs_for_remaining(remaining)
            pick = torch.multinomial(p, num_samples=1).item()
            cand_id = remaining[pick]

            sal = compute_candidate_saliency(
                model=model,
                seq=seq,
                candidate_id=torch.tensor(cand_id, device=device),
                images=images,
                sys_token_len=args.sys_token_len,
                img_token_len=args.img_token_len,
                target_layers=args.sgrs_target_layers
            )

            if sal >= tau:
                accepted = True
                chosen_id = cand_id
                accepted_sals.append(float(sal))
                break
            else:
                remaining.pop(pick)
                if sal > best_sal:
                    best_sal = float(sal)
                    chosen_id = cand_id  #记录当前见过的最大显著性（用于 fallback）

        #全部被拒时，从原始Top-K里选显著性最高
        if not accepted:
            sal_map = {}
            #已记过best_sal的候选
            if chosen_id is not None:
                sal_map[chosen_id] = best_sal
            #对其余候选补算显著性
            for cid in candidates:
                if cid in sal_map:
                    continue
                s_val = compute_candidate_saliency(
                    model=model,
                    seq=seq,
                    candidate_id=torch.tensor(cid, device=device),
                    images=images,
                    sys_token_len=args.sys_token_len,
                    img_token_len=args.img_token_len,
                    target_layers=args.sgrs_target_layers
                )
                sal_map[cid] = float(s_val)
            #选择显著性最大的候选
            chosen_id = max(sal_map.keys(), key=lambda k: sal_map[k])
            accepted_sals.append(sal_map[chosen_id])

        #追加所选 token 并做一次增量前向（显式 attention_mask）
        chosen_id_t = torch.tensor([[chosen_id]], device=device, dtype=torch.long)
        seq = torch.cat([seq, chosen_id_t], dim=-1)

        #显式attention_mask，长度=past_len + 1
        past_len = _get_past_len(past_key_values)  #之前上下文长度
        attn_mask = torch.ones((seq.shape[0], past_len + 1), dtype=torch.long, device=device)

        inc_out = model(
            input_ids=chosen_id_t,
            images=images,
            attention_mask=attn_mask,       #★ 关键：显式传入
            use_cache=True,
            past_key_values=past_key_values,
            output_attentions=False,
            return_dict=True
        )
        logits = inc_out.logits
        past_key_values = inc_out.past_key_values

        #可选：遇到停止符提前结束
        if stop_ids and int(chosen_id) in stop_ids:
            break

    return seq

def eval_model(args):
    #加载实验配置
    #设备初始化
    #device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cpu")
    #print(f"Using device: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'}")
    #Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    #reset attention modules in model 
    if args.use_visaug == True:
        for i, layer in enumerate(model.model.layers):
            if i > 8 and i < 15:
                attn_adap = AttnAdapter(layer.self_attn.config, args.enh_para, args.sup_para)
                attn_adap.load_state_dict(layer.self_attn.state_dict())
                attn_adap = attn_adap.half().cuda()
                layer.self_attn = attn_adap

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    #停止符工具函数（根据conv模板获取）
    def get_stop_ids(tokenizer, conv):
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stop_ids = tokenizer.encode(stop_str, add_special_tokens=False)
        return set(stop_ids)

    #逐样本生成
    for i, line in enumerate(tqdm(questions)):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        qs = qs + " Please just answer yes or no."
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        #构造对话模板
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        #input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        #获取停止符ID
        stop_ids = get_stop_ids(tokenizer, conv)

        #调用SGRS生成
        out_ids = sgrs_generate(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            args=args,
            stop_ids=stop_ids
        )
        #只解码生成
        prompt_len = input_ids.shape[-1]  
        gen_ids = out_ids[0, prompt_len:].detach().cpu()  
        generated_text = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True).strip()

        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,       
            "text": generated_text,
            "model_id": model_name
        }) + "\n")
        ans_file.flush()

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #基础模型参数
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None, help="基础语言模型路径（如Llama-2-7B）")
    parser.add_argument("--image-folder", type=str, default="", help="图像文件夹路径")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl", help="问题文件路径")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl", help="答案文件路径（废弃，用answer-file-ver）")
    parser.add_argument("--answer-file-ver", type=str, default="0", help="当前版本答案文件路径")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="对话模板（如llava_v1、qwen_vl）")
    parser.add_argument("--num-chunks", type=int, default=1, help="数据分块数（多卡并行用）")
    parser.add_argument("--chunk-idx", type=int, default=0, help="当前分块索引（多卡并行用）")

    parser.add_argument("--gpu", type=int, default=None, help="GPU卡号（如0、1）")
    parser.add_argument("--category", type=str, default="NONE", help="数据集类别筛选（避免未定义错误）")

    #消融实验参数
    parser.add_argument("--abl-sink", action="store_true", help="消融注意力sink机制")
    parser.add_argument("--abl-head", action="store_true", help="消融注意力头干预")

    #注意力替换参数（visaug）
    parser.add_argument("--use-visaug", action='store_true', default=False, help="启用注意力适配层替换")
    parser.add_argument("--target_layer", type=int, default=7, help="替换注意力层的索引")
    parser.add_argument("--single-pred-prompt", action="store_true", help="启用单选项提示（仅输出选项字母）")
    parser.add_argument("--temperature", type=float, default=0.2, help="采样温度（SGRS中暂未用，保留兼容）")

    #实验配置文件
    #parser.add_argument("--exp-config", type=str, default=None, required=True, help="实验配置yaml文件路径")

    #SGRS核心超参数（对齐论文，在parse_args前定义）
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="最大生成token数")
    parser.add_argument("--sgrs_top_k", type=int, default=5, help="SGRS Top-K候选数量（论文K）")
    parser.add_argument("--sgrs_max_resample", type=int, default=3, help="SGRS最大重采样轮次（论文R）")
    parser.add_argument("--sgrs_alpha", type=float, default=0.6, help="SGRS阈值灵敏度（论文α，推荐0.6）")
    parser.add_argument("--sgrs_history_window", type=int, default=5, help="SGRS历史窗口大小（论文W）")
    parser.add_argument("--sgrs_target_layers", type=list, default=[6,7,8], help="SGRS目标层（中深层，如6-8）")
    parser.add_argument("--sys_token_len", type=int, default=35, help="系统token长度（论文Sys_L，LLaVA-1.5默认35）")
    parser.add_argument("--img_token_len", type=int, default=576, help="图像token长度（论文Img_L，LLaVA-1.5默认576）")

    #解析参数
    args = parser.parse_args()

    #修正类别筛选（避免未定义错误）
    if args.category.lower() == "none":
        args.category = ""

    #启动评估
    eval_model(args)