import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['GRADIO_TEMP_DIR'] = 'gradio_temp'
import gradio as gr
import torch
import argparse

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import os
import time  
from datasets import load_from_disk,load_dataset
import torch
import json
from tqdm import tqdm
import re	
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as Colormap
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F



import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import torch
import seaborn as sns
from matplotlib.colors import LogNorm
from io import BytesIO
from PIL import Image


import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import torch
import seaborn as sns
from matplotlib.colors import LogNorm
from io import BytesIO
from PIL import Image

from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor,AutoTokenizer
from transformers import Qwen2VLForConditionalGeneration
from hallu_index import find_sentence_index , compute_hallu_output

from enum import auto, Enum



def count_tokens_around_image(token_ids, image_token):
    """
    计算图像 token 前和图像 token 后的 token 数

    参数：
      token_ids: 一维 token 序列，可以是 list 或 torch.Tensor（可能为嵌套列表）
      image_token: 图像 token 的标识，可以是单个标识或多个标识的集合
      
    返回：
      一个二元组 (before_count, after_count)
    """
    # 如果是 torch.Tensor，则转换为 list
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    
    # 如果 token_ids 是嵌套列表（例如 [[...]]），则取第一个元素
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    
    # 如果 image_token 不是可迭代对象（比如单个整数），转换为集合
    if not isinstance(image_token, (list, tuple, set)):
        image_tokens = {image_token}
    else:
        image_tokens = set(image_token)
    
    # 从前往后统计直到遇到第一个图像 token
    before_count = 0
    for token in token_ids:
        if token in image_tokens:
            break
        before_count += 1

    # 从后往前统计直到遇到第一个图像 token
    after_count = 0
    for token in reversed(token_ids):
        if token in image_tokens:
            break
        after_count += 1

    return before_count, after_count


def disable_torch_init():

    # 禁用 Linear 层的默认初始化
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    # 禁用 LayerNorm 层的默认初始化
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def log_normalize(array):
    array = np.log1p(array)  # log(1 + x)，避免 log(0) 错误
    return min_max_normalize(array)  # 再次进行 Min-Max 归一化

def min_max_normalize(array):
    """
    对输入的 numpy 数组进行 Min-Max 归一化，缩放到 [0, 1] 范围。
    """
    array_min = np.min(array)
    array_max = np.max(array)
    if array_max == array_min:  # 避免除以零
        return np.zeros_like(array)
    return (array - array_min) / (array_max - array_min)

def fore_after_img_token(input_tokens,img_index):
    
    if img_index in input_tokens:
        first_img_index = input_tokens.index(img_index)
        num_tokens_before = first_img_index  # 第一个 151655 前的所有 token个数
    else:
        num_tokens_before = len(input_tokens)  # 如果没有 151655，则认为全部是文本

    # 2. 统计最后一个 151655 后的 token 数量（输出 token 数量）
    num_tokens_after = 0
    for token in reversed(input_tokens):
        if token == 151655:
            break
        num_tokens_after += 1
    return num_tokens_before,num_tokens_after

def visualize_attention(multihead_attention,title="Layer 5",sample_style="All layers"):
    averaged_attention = torch.mean(multihead_attention, axis=1)[0].float()
    averaged_attention = torch.nn.functional.avg_pool2d(
        averaged_attention.unsqueeze(0).unsqueeze(0), 20, stride=20).squeeze(0).squeeze(0)

    cmap = plt.cm.get_cmap("viridis")

    plt.figure(figsize=(5, 5),dpi=400)
    log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention.max())

    ax = sns.heatmap(averaged_attention,
                     cmap=cmap,
                     norm=log_norm)

    x_ticks = [str(i*20) for i in range(0,averaged_attention.shape[0])]
    y_ticks = [str(i*20) for i in range(0,averaged_attention.shape[0])]
    ax.set_xticks([i for i in range(0,averaged_attention.shape[0])])
    ax.set_yticks([i for i in range(0,averaged_attention.shape[0])])
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)

    # label ticks
    for label in ax.get_xticklabels():
        tick_location = int(label.get_text())
        if 0 <= tick_location <= 40:
            # set the color of the tick labels
            label.set_color('blue')
            label.set_fontweight('bold')
        elif 40 < tick_location <= 600:
            label.set_color('red')

    for label in ax.get_yticklabels():
        tick_location = int(label.get_text())
        if 0 <= tick_location <= 40:
            # set the color of the tick labels
            label.set_color('blue')
            label.set_fontweight('bold')
        elif 40 < tick_location <= 600:
            label.set_color('red')


    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)

    plt.title(title, fontsize=20)

    buf = BytesIO()
    plt.savefig(buf,format='png', bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf).copy()
    if sample_style == "All layers":
        image = image.resize((768, 768))
    else:
        image = image.resize((1024, 1024))
    buf.close()
    plt.close()

    return image



def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def concatenate_images(images_list, number_rows=5, number_cols=7):
    assert len(images_list) == number_rows * number_cols

    # Assuming all images are the same size
    img_width, img_height = images_list[0].size

    # Creating a blank canvas for the final image
    final_img = Image.new('RGB', (img_width * number_cols, img_height * number_rows))

    # Loop over the images and paste them onto the canvas
    for idx, img in enumerate(images_list):
        row = idx // number_cols  # row index
        col = idx % number_cols  # column index

        # paste the image at the correct position on the canvas
        final_img.paste(img, (img_width * col, img_height * row))

    return final_img



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument('--model-path', type=str, required=False, default="/root/autodl-tmp/llava-v1.5-7b")
    parser.add_argument('--prompt', type=str, default="Describe the image in detail.", help="Text prompt for the model")
    parser.add_argument('--layers', type=str, default='All layers', help='Specify number of layers to show')
    pargs = parser.parse_args()

    examples = [
        ["./figs/COCO_val2014_000000499775.jpg","Describe the image in detail."],
        
    ]

    #examples = [
    #["path_to_example_image1.jpg", "Describe the first image in detail."],
    #["path_to_example_image2.jpg", "Describe the second image in detail."]
    #]
    class InferenceArgs:
        model_path = pargs.model_path
        model_base = None
        image_file = None
        device = "cuda"
        conv_mode = None
        temperature = 0.2
        max_new_tokens = 512
        load_8bit = False
        load_4bit = False
        debug = False
        image_aspect_ratio = 'pad'
        layers = pargs.layers
        prompt = pargs.prompt
    
    args = InferenceArgs()
    disable_torch_init()
    print('Loading model...')

    
    model_path = "/root/autodl-tmp/Qwen2-VL-7B-Instruct"
    model_name = get_model_name_from_path(model_path)
    messages = [
    [{"role": "user", "content": [{"type": "image", "image": "/root/autodl-tmp/FastV/figs/COCO_val2014_000000406451.jpg"}, {"type": "text", "text": "Please describe this image in detail."}]}]
    ]


    tokenizer = AutoTokenizer.from_pretrained(model_path)
    #model = Qwen2VLForConditionalGeneration.from_pretrained(model_path,attn_implementation="eager")
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path,_attn_implementation = "eager")
    model.to('cuda')
    #image_processor = AutoProcessor.from_pretrained(model_path,use_fast=True)
    processor = AutoProcessor.from_pretrained(model_path,use_fast=True)

    # 提取 text
    prompt_textbox = messages[0][0]["content"][1]["text"]
    
    # 提取 image 路径
    image_input = messages[0][0]["content"][0]["image"]

    #model.config.use_fast_v = False
    #model.model.reset_fastv()

    total_layers = args.layers
    

    attention_Layer = args.layers

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    def temp_inference(prompts,images,append_output=None):
        outputs = []#用来保存最终生成的文本列表
        outputs_attention = []#用来保存模型生成过程中的注意力列表
        if append_output is None:
            append_output_str=""
        else:
            append_output_str=append_output

        for prompt,image in tqdm(zip(prompts,images),total=len(prompts)):
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            #input_ids = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt").to("cuda") 
            processed_inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt").to("cuda") 
            
            #input_ids = processed_inputs["input_ids"]  # 提取文本 token IDs
            #pixel_values = processed_inputs.get("pixel_values", None)  # 提取图像特征
            input_ids = processed_inputs["input_ids"]
            input_seq_len = input_ids.shape[1]  # 输入 Token 长度
            print("input_seq_len",input_seq_len)
            print("input_ids",input_ids)
            #exit()

            
            name = '499775'
            loss_model = True
            hmode = 'hall'
            #word = '\u2581pillows'  # \u2581
            #word = ' pillows'
            word = " black"
            
            only_text = True
            if loss_model:
                model.train()
                model.requires_grad_(True)
                max_new_tokens = 1
            else:
                max_new_tokens = 255
            # loss_model
            with torch.enable_grad():
            #else
            # with torch.inference_mode():
                start = time.time()
                if loss_model:
                    finla_input_input_ids = torch.load(f'Qwen.pt').cuda()
                    attention_mask = torch.ones_like(finla_input_input_ids)
                    #print(f"finla_input_input_ids token: {type(finla_input_input_ids[0])}")
                    #token_to_remove = 151645
                    #
                    #finla_input_input_ids = finla_input_input_ids[finla_input_input_ids != token_to_remove]
                    #print(f"len token: {len(finla_input_input_ids[0])}")
                    print(f"finla_input_input_ids token: {type(finla_input_input_ids[0])}")
                    print(f"finla_input_input_ids token: {finla_input_input_ids[0]}")
                    shou_inputs = tokenizer.decode(finla_input_input_ids[0],skip_spectial_tokens=True)
                    print(shou_inputs)
                    #exit()
                    

                    
                else:
                    finla_input_input_ids = input_ids 
                print(model.generation_config)
                #exit()
                output_ids = model.generate(
                    finla_input_input_ids,
                    max_new_tokens=max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                    output_attentions=True,
                    top_k=0,
                    top_p=1.0,
                    no_repeat_ngram_size=None,  # 禁用n-gram重复屏蔽
                    temperature=1.0,    # 调整温度
                    num_beams = 2,
                    #stopping_criteria=[stopping_criteria],
                    #attention_mask = attention_mask,
                    )
                print("output_ids.scores[0][0]",output_ids.scores)
                print(output_ids["sequences"][0])
                #exit()
                #output_ids.attentions
                print("output_ids的类型是",type(output_ids))
                print(output_ids.keys())  # 检查有哪些字段
                #print("output_ids.attentions",output_ids.attentions)
                #exit()
                reponse = tokenizer.decode(output_ids["sequences"][0],skip_spectial_tokens=True)
                print('output_ids["sequences"][0]',output_ids["sequences"][0])
                print(reponse)
                #exit()

                print("output_ids的类型是",type(output_ids))
                print(output_ids.keys())  # 检查有哪些字段

                time_cost = time.time() - start
                #new_tokens = output_ids[0, input_ids.shape[1]:]  # 张量切片
                new_tokens = output_ids["sequences"][0, input_ids.shape[1]:]
                sentence_index,hallu_logits_index,out_sub_text = compute_hallu_output(new_tokens, word, tokenizer)
                #out_sub_text = out_sub_text.split()
                #out_sub_token_ids = new_tokens[:sentence_index].tolist()
                #原来的处理：
                out_sub_text = tokenizer.tokenize(out_sub_text) 
                #print("out_sub_text",out_sub_text)
                
                #exit()
    
                #exit()            
                if loss_model:
                    #image_token_ids = [ 151652,151653,151655,151644,151645]
                    image_token_ids = [151655]
                    #image_token_ids = [151655,151656]
                    filtered_ids_list = []
                    for seq in input_ids:
                        filtered = [tid for tid in seq.tolist() if tid not in image_token_ids]
                        filtered_ids_list.append(filtered)
                    #print(filtered_ids_list)   
                    #out_sub_text = tokenizer.tokenize(out_sub_text) 

                    #all_token_ids = filtered_ids_list[0] + out_sub_token_ids
                    #all_text_token_str_clr = tokenizer.decode(all_token_ids, skip_special_tokens=False)
                    #all_text_token_str_clr = tokenizer.tokenize(all_text_token_str_clr) 
                    #原来的处理：
                    show_token_input = tokenizer.decode(filtered_ids_list[0],skip_spectial_tokens=False)
                    show_token_input = tokenizer.tokenize(show_token_input) 
                    all_text_token_str = show_token_input + out_sub_text
                    all_text_token_str_clr = [token.lstrip('Ġ') for token in all_text_token_str] 
                    all_text_token_str_clr = [token.lstrip('Ċ') for token in all_text_token_str_clr] 
                    all_text_token_str_clr = [token.lstrip('.ĊĊ') for token in all_text_token_str_clr] 

                    print("all_text_token_str_clr",all_text_token_str_clr)
                    

                    #exit()
                image_index = None
                if only_text:
                    fore_text_len,after_text_len = count_tokens_around_image(finla_input_input_ids,151655)
                    print(f"fore_text_len:{fore_text_len} after_text_len {after_text_len}")
                    #exit()
                    
                if loss_model:
                    hallucination_score = output_ids.scores[0][0]
                    loss = F.cross_entropy(hallucination_score, torch.tensor(hallu_logits_index).cuda()).requires_grad_(True)
                    print("hallucination_score",hallucination_score)
                    print("loss",loss)
                    print("torch.tensor(hallu_logits_index)",torch.tensor(hallu_logits_index))
                    # 假设 vocab_size=50000
                    vocab_size = model.config.vocab_size
                    assert hallu_logits_index < vocab_size, f"标签索引 {hallu_logits_index} 超出词表大小 {vocab_size}"
                    #exit()
                    loss.backward()
                    # 确保梯度未被错误截断或归一化
                    print("梯度范围:", model.model.layers[0].self_attn.attention_dapter.params.grad.min(), model.model.layers[0].self_attn.attention_dapter.params.grad.max())
                    #exit()
                    
                    # get attn_saliency
                    layer_attn_saliency_list = []
                    layer_attn_weight_list = []
                    for l in range(28):
                        head_attn_saliency_list = []
                        head_attn_weight_list = []
                        for h in range(28):
                            if only_text:
                                torch.set_grad_enabled(True)
                                #print("现在查看output_ids的的注意力的结构")
                                attentions = output_ids.attentions

                                attn_weight_sm_raw, attn_weight_ma_raw = output_ids.attentions[0][l][0][0][h], output_ids.attentions[0][l][1][0][h]
                                #attn_weight_sm_raw, attn_weight_ma_raw = output_ids.attentions[l][0][0][h], output_ids.attentions[l][1][0][h]
                                #print(attn_weight_ma_raw)
                                #print(attn_weight_ma_raw)
                                
                                
                                attn_saliency_raw = torch.abs(attn_weight_ma_raw * model.model.layers[0].self_attn.attention_dapter.params.grad[0,h,::])
                                print("length attn_saliency_raw",{len(attn_saliency_raw)})
                                #exit()
                                print("梯度是：",model.model.layers[0].self_attn.attention_dapter.params.grad[0,h,::])
                                #exit()
                                #exit()
                                # 提取下三角矩阵
                                attn_weight_sm_raw = torch.tril(attn_weight_sm_raw)
                                attn_saliency_raw = torch.tril(attn_saliency_raw)
                                
                                seq_len = attn_saliency_raw.size(0)  # 获取序列长度
                                print("attn_saliency_raw.size(0)  # 获取序列长度",seq_len)
                                #exit()
                                keep_indices = torch.cat([
                                    torch.arange(0, fore_text_len),          # 保留 0-x1
                                    torch.arange(seq_len-after_text_len, seq_len)     # 保留 x2 到末尾
                                ])
                                
                                print(f"keep_indices,{keep_indices}")
                                print("keep_indices 的长度：",len(keep_indices))
                                #exit()
                                print(attn_saliency_raw)

                                #attn_saliency = attn_saliency_raw[keep_indices][171]
                                attn_saliency = attn_saliency_raw[keep_indices][:, keep_indices]
                                attn_weight = attn_weight_sm_raw[keep_indices][:, keep_indices]
                            else:
                                print('not yet')
                            
                            head_attn_saliency_list.append(attn_saliency)
                            head_attn_weight_list.append(attn_weight)
                            
                        layer_attn_saliency_list.append(head_attn_saliency_list)
                        layer_attn_weight_list.append(head_attn_weight_list)
                    # 画图
                    # 遍历每个 layer
                    # 查看tensor值： torch.unique(torch.stack([torch.stack(inner, dim=0) for inner in layer_attn_saliency_list],dim=0))
                    
                    global_sum_attention_saliency_map = None
                    global_sum_attention_weight_map = None
                    for layer_idx in range(28):
                        # 获取当前 layer 的所有 head 的 attention map
                        head_attention_saliency_maps = layer_attn_saliency_list[layer_idx]
                        head_attention_weight_maps = layer_attn_weight_list[layer_idx]
                        # 提取每个 head 的下三角矩阵（包括对角线）
                        np_saliency_maps = []
                        np_weight_maps = []
                        for head_idx in range(28):
                            attn_attn_saliency = head_attention_saliency_maps[head_idx] 
                            attn_attn_weight = head_attention_weight_maps[head_idx] 
                            # 将 tensor 转换为 numpy 数组
                            attn_attn_saliency_np = attn_attn_saliency.cpu().detach().numpy()
                            attn_attn_weight_np = attn_attn_weight.cpu().detach().numpy()
                            
                            #放入列表
                            np_saliency_maps.append(attn_attn_saliency_np)
                            np_weight_maps.append(attn_attn_weight_np)
                            
                        # 先对 head 取平均
                        avg_lower_saliency_map = np.mean(np_saliency_maps, axis=0)  
                        avg_avg_lower_saliency_map = min_max_normalize(avg_lower_saliency_map)
                        
                        avg_lower_weight_map = np.mean(np_weight_maps, axis=0)  
                        avg_avg_lower_weight_map = min_max_normalize(avg_lower_weight_map)
                        
                        # 创建全局求和
                        if global_sum_attention_saliency_map is None:
                            global_sum_attention_saliency_map = np.zeros_like(avg_avg_lower_saliency_map)
                        global_sum_attention_saliency_map += avg_avg_lower_saliency_map
                        
                        if global_sum_attention_weight_map is None:
                            global_sum_attention_weight_map = np.zeros_like(avg_avg_lower_weight_map)
                        global_sum_attention_weight_map += avg_avg_lower_weight_map
                    
                    global_sum_attention_saliency_map_normalized = log_normalize(global_sum_attention_saliency_map)
                    print("global_sum_attention_saliency_map_normalized",global_sum_attention_saliency_map_normalized)
                    print("len(global_sum_attention_saliency_map_normalized)",len(global_sum_attention_saliency_map_normalized))
                    print("len(all_text_token_str_clr)",len(all_text_token_str_clr))
                    print("keep_indices:", keep_indices)
                    print("keep_indices.shape:", keep_indices.shape)
                    #exit()
                    global_sum_attention_weight_map_normalized = log_normalize(global_sum_attention_weight_map)
                    # 绘制 saliency map
                    plt.figure(figsize=(20, 20))
                    im = plt.imshow(global_sum_attention_saliency_map_normalized, cmap='viridis', interpolation='nearest')
                    plt.xticks(np.arange(len(all_text_token_str_clr)), all_text_token_str_clr, rotation=90, fontsize=8)   # 横坐标
                    plt.yticks(np.arange(len(all_text_token_str_clr)), all_text_token_str_clr, fontsize=8)  # 纵坐标
                    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)  # fraction 控制颜色条的长度，pad 控制颜色条与图像的间距
                    cbar.ax.tick_params(labelsize=8)  # 设置 colorbar 刻度字体大小
                    plt.title(f"All_Layer - Saliency Map")
                    plt.xlabel("Key")
                    plt.ylabel("Query")
                    plt.show()
                    
                    if only_text:
                        plt.savefig(f'onlytext_{hmode}_{name}_ALL_layer_avg_lower_saliency_map_normalized.png')
                    else:
                        plt.savefig(f'{hmode}_{name}_ALL_layer_avg_lower_tri_map_normalized.png')
                    plt.close()
                    #------------------------------------------#
                    # 绘制 weight map
                    plt.figure(figsize=(20, 20))
                    # 绘制当前 layer 的平均下三角 attention map 热力图
                    im = plt.imshow(global_sum_attention_weight_map_normalized, cmap='viridis', interpolation='nearest')
                    plt.xticks(np.arange(len(all_text_token_str_clr)), all_text_token_str_clr, rotation=90, fontsize=8)   # 横坐标
                    plt.yticks(np.arange(len(all_text_token_str_clr)), all_text_token_str_clr, fontsize=8)  # 纵坐标
                    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)  # fraction 控制颜色条的长度，pad 控制颜色条与图像的间距
                    cbar.ax.tick_params(labelsize=8)  # 设置 colorbar 刻度字体大小
                    plt.title(f"All_Layer - Weight Map (Attention Map B)")
                    plt.xlabel("Key")
                    plt.ylabel("Query")
                    plt.show()
                    if only_text:
                        plt.savefig(f'onlytext_{hmode}_{name}_ALL_layer_avg_lower_weight_map_normalized.png')
                    else:
                        plt.savefig(f'{hmode}_{name}_ALL_layer_avg_lower_tri_map_normalized.png')
                    plt.close()
                    
                    print('1')
                else:
                    only_output_ids = output_ids.sequences[0,len(finla_input_input_ids[0]):]
                    only_outputs_ids_before_hallu = only_output_ids[:sentence_index]
                    new_input_ids = torch.cat([input_ids,only_outputs_ids_before_hallu.unsqueeze(0)],1)
                    torch.save(new_input_ids.cpu(), f'{name}_{hmode}.pt')
                
                exit()
                output = output.strip().replace("</s>","")
                outputs.append(output)

                outputs_attention.append(output_ids['attentions'])
                if len(outputs) > 1:
                    print(output)
                if append_output is None:
                    return outputs,outputs_attention,time_cost
        return outputs,outputs_attention

    def select_numbers(n, x):
        return [(i*(n-1))//(x-1) for i in range(x)]
    def inference(image_input, prompt, num_of_layers="All layers"):
        prompts = [prompt]
        images = [image_input]

        model_output_ori,outputs_attention,time_cost = temp_inference(prompts,images)
        # time cost in seconds

        model_output,outputs_attention = temp_inference(prompts,images,append_output=model_output_ori[0])
        print(model_output_ori)
        images_list = []
        for i in outputs_attention:
            if num_of_layers == "All layers":
                show_layers = list(range(0,total_layers))
            elif num_of_layers == "Sample 3 layers":
                show_layers = select_numbers(total_layers,3)
            elif num_of_layers == "Sample 5 layers":
                show_layers = select_numbers(total_layers,5)
            elif num_of_layers == "Sample 10 layers":
                show_layers = select_numbers(total_layers,10)
            else:
                show_layers = list(range(0,total_layers))
            for j in show_layers:
                images_list.append(visualize_attention(i[0][j].cpu(),title="Layer "+str(j+1), sample_style=num_of_layers))
        # final_images = concatenate_images(images_list, number_rows=5, number_cols=7)
        # return final_images,images_list
        output = model_output_ori if isinstance(model_output_ori, str) else model_output_ori[0]
        total_time_cost = "Total Time Cost:{:.2f}s".format(time_cost)
        return images_list,output,total_time_cost


    
    import base64
    from io import BytesIO
    def pil_to_base64(pil_image):
        pil_image = Image.open(pil_image)
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG") 
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    fastv_tradeoff = pil_to_base64('./figs/fastv_tradeoff.png')
    attn_map = pil_to_base64('./figs/attn_map.png')

    description= f'''# FastV Demo
Welcome to the demonstration for [FastV](https://arxiv.org/abs/2403.06764), an innovative plug-and-play inference accelerator specifically designed for Large Vision-Language Models. 

FastV enables a **45% reduction in theoretical FLOPs** without compromising on performance by pruning redundant visual tokens in deep layers.

<center><img src="data:image/png;base64,{fastv_tradeoff}" alt="FastV tradeoff image" style="width: 40%;"/></center>

This demo unveils the attention maps of the llava-1.5-7B model to illustrate the inefficient attention phenomena prevalent in Large Vision-Language Models (LVLMs).

The **System Prompt tokens are highlighted in blue**, followed by **Image tokens marked in red**. The remaining tokens are the text tokens marked in black.

## Guidelines:

1. **Upload an image**, **enter a prompt** and **select the number of layers** to visualize the attention maps.

2. The model output with FastV, time cost, and the attention maps for the sampled layers will be displayed.

* **Note**: Due to the Network constraints, the attention map generation may take up to 30 seconds.*

# Dive in, explore and enjoy the capabilities of FastV! 

For more details, visit the [FastV GitHub page](https://github.com/pkunlp-icler/FastV).

'''
# <img src="data:image/png;base64,{attn_map}" alt="Attention Map image" style="width: 33%;"/>
     



    #demo = gr.Interface(
    #                    fn=inference,
    #                    inputs=[image_input,prompt_textbox,attention_Layer],
    #                    description=description,
    #                    outputs=[gallery,"text","text"],
    #                    examples=examples,
    #                    allow_flagging="never",
    #                )
        
    #demo.launch(share=True,server_name="127.0.0.1", server_port=6006, inbrowser=True)

    def display_images(images_list):
        for img in images_list:
            plt.imshow(img)
            plt.axis('off')  # 去掉坐标轴
            plt.show()

    #example_image = Image.open(exmaples[0][0])
    #example_prompt = examples[0][1]
    #images_list,output,total_time_cost = inference(example_image, example_prompt, num_of_layers="All layers")

    images_list,output,total_time_cost = inference(image_input, prompt_textbox, num_of_layers="All layers")
    display_images(images_list)
    print("Inference Output:", output)  # 打印推理文本输出
    print(total_time_cost)  # 打印推理时间


