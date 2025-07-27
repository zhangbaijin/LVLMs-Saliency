import math
import os
import json
import random
import torch
import torch.nn.functional as F
from transformers.models.llama.configuration_llama import LlamaConfig
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, IMAGE_TOKEN_INDEX
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from AttnAdapter import AttnAdapter
from PIL import Image
import argparse


def compute_attention_entropy(attn_weights: torch.Tensor) -> float:
    """
    Compute average attention entropy over all layers, heads, and tokens.
    attn_weights shape: (num_layers, batch, num_heads, seq_len, seq_len)
    """
    A = attn_weights.clamp(min=1e-9)
    ent = - (A * A.log()).sum(dim=-1)  # shape (num_layers, batch, num_heads, seq_len)
    return ent.mean().item()


def estimate_sara_params(model, tokenizer, image_processor, sample_prompts, sample_images, gamma: float = 0.2):
    """
    Estimate SARA parameters automatically using attention entropy and model depth.
    """
    model.eval()
    entropies = []
    device = next(model.parameters()).device

    for text, img in zip(sample_prompts, sample_images):
        # Build multimodal prompt
        conv = conv_templates['vicuna_v1'].copy()
        if model.config.mm_use_im_start_end:
            prefix = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n'
        else:
            prefix = DEFAULT_IMAGE_TOKEN + '\n'
        prompt_text = prefix + text
        prompt_text = prefix + text
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize multimodal input
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).to(device)
        img_t = image_processor.preprocess(img, return_tensors='pt')['pixel_values']
        img_t = img_t.unsqueeze(0).to(device).half()

        # Forward pass with attentions
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                images=img_t,
                output_attentions=True,
                return_dict=True
            )
        attn = torch.stack(outputs.attentions)
        H = compute_attention_entropy(attn)
        entropies.append(H)

    if not entropies:
        raise RuntimeError("No attention entropies computed: check sample_prompts and sample_images.")
    avg_H = sum(entropies) / len(entropies)
    # Determine sequence length from attention tensor
    L = attn.shape[-1] if hasattr(attn, 'shape') else None
    # Avoid division by zero
    if L is None or L <= 1:
        norm_H = 0.0
    else:
        denom = math.log(L)
        if denom == 0:
            norm_H = 0.0
        else:
            norm_H = avg_H / denom
    if math.isnan(norm_H):
        print(f"Warning: normalized entropy is NaN (avg_H={avg_H}, L={L}), defaulting to 0")
        norm_H = 0.0

    output_para = gamma * (1 - norm_H)
    # Clamp to valid range
    if math.isnan(output_para) or output_para < 0:
        print(f"Warning: computed output_para={output_para}, defaulting to gamma={gamma}")
        output_para = gamma

    # Empirical enh_para and sup_para
    enh_para = 1 + 0.15  # for depth ~32 models
    sup_para = 1.0

    print(f"Estimated SARA params: enh_para={enh_para:.3f}, sup_para={sup_para:.3f}, output_para={output_para:.3f}")
    return enh_para, sup_para, output_para

def main():
    parser = argparse.ArgumentParser(description="SARA auto-parameterized inference for llava-v1.5-7B")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None,
                        help="Optional base model path or namespace for tokenizer loading")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="Folder containing all images for both sampling and inference")
    parser.add_argument("--question-file", type=str, required=True,
                        help="JSONL file with list of questions for inference (and sampling)")
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    args = parser.parse_args()

    disable_torch_init()
    #tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path)
    model_name = "llava-v1.5-7B"

    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load all questions
   # Load all questions
    with open(args.question_file, "r") as f:
        all_qs = [json.loads(l) for l in f]
    # Sample up to 50 for entropy estimation
    sample_qs = random.sample(all_qs, min(50, len(all_qs)))
    sample_prompts = [q["text"] for q in sample_qs]
    sample_images = [Image.open(os.path.join(args.image_folder, q["image"])) for q in sample_qs]

    # Estimate SARA parameters on sample
    enh_para, sup_para, output_para = estimate_sara_params(
        model, tokenizer, image_processor, sample_prompts, sample_images)

    # Replace attention modules
    for i, layer in enumerate(model.model.layers):
        if 8 < i < 15:
            adapter = AttnAdapter(layer.self_attn.config, enh_para, sup_para, output_para)
            adapter.load_state_dict(layer.self_attn.state_dict())
            layer.self_attn = adapter.half().to(device)

    # Run inference on all questions
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    with open(args.answers_file, "w") as ans_f:
        for q in all_qs:
            idx = q["question_id"]
            image_file = q["image"]
            text = q["text"] + " Please just answer yes or no."
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + text
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
            img = Image.open(os.path.join(args.image_folder, image_file))
            img_t = image_processor.preprocess(img, return_tensors='pt')['pixel_values'].unsqueeze(0).to(device).half()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=img_t,
                    max_new_tokens=1024,
                    use_cache=True)
            new_tokens = output_ids[0, input_ids.shape[1]:]
            text_out = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            ans_f.write(json.dumps({"question_id": idx, "text": text_out}) + "\n")

if __name__ == "__main__":
    main()
