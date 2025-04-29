import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['GRADIO_TEMP_DIR'] = 'gradio_temp'
import gradio as gr
import torch
import argparse
#from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
#from llava.conversation import conv_templates, SeparatorStyle
#from llava.model.builder import load_pretrained_model
#from llava.utils import disable_torch_init
#from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
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
    parser.add_argument('--model-path', type=str, required=False, default="/root/autodl-tmp/Qwen2-VL-7B-Instruct")
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
    # 打印 Tokenizer 的对话模板
    print("Chat Template:\n", tokenizer.chat_template)
    #exit()
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path,attn_implementation="eager" )
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
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            #input_ids = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt").to("cuda") 
            processed_inputs = processor(text=text, images=images, videos=videos, padding=True, add_special_tokens=False,return_tensors="pt").to("cuda") 
            input_ids = processed_inputs["input_ids"]
            image_token_ids = [151652, 151653, 151654, 151655, 151656]
            filtered_ids_list = []
            for seq in input_ids:
                filtered = [tid for tid in seq.tolist() if tid not in image_token_ids]
                filtered_ids_list.append(filtered)
            
            # 如果还需要维持成 tensor，可能需要补齐或者根据需求进行后续处理
            # 这里只是演示，把它变成不定长的列表

            
            #print("input_ids",input_ids)
            print(filtered_ids_list)
            input_seq_len = input_ids.shape[1]  # 输入 Token 长度
            print(input_seq_len)
            # 获取图像占位符的 Token ID（假设 Qwen-VL 使用 "<img>"）
            #image_token_id = tokenizer.convert_tokens_to_ids("<img>")  # 例如 151857
            #print(f"img token为：{image_token_id}")
            


            loss_model = False
            hmode = 'hall'
            word = " black"
            word_ = "bus"
            tokens = tokenizer.tokenize(word)
            tokens = tokenizer.tokenize(word_)
            print(f"'{word}' 的分词结果：{tokens}")
            print(f"'{word_}' 的分词结果：{tokens}")
            print(f'bus:{tokenizer.encode("bus", add_special_tokens=False) }')
            print(f'  bus:{tokenizer.encode(" bus", add_special_tokens=False) }')
            #exit()
            
            
            #piloow = tokenizer.encode("pillows", add_special_tokens=False)
            #piloow_ = tokenizer.encode(" pillows", add_special_tokens=False)
            #print(piloow)
            #print("上面是pillows的encoder")
            #print("下面是 pillows的encoder")
            #print(piloow_)
            only_text = True
            max_new_tokens = 255
            #print("2222222222222222222222222222222")
            with torch.inference_mode():
                start = time.time()
                finla_input_input_ids = input_ids
                output_ids = model.generate(
                    **processed_inputs,
                    max_new_tokens=max_new_tokens,
                    output_scores=True,
                    output_attentions=True,
                    top_k=0,
                    top_p=1.0,
                    no_repeat_ngram_size=None,  # 禁用n-gram重复屏蔽
                    temperature=0.3,    # 调整温度
                    num_beams = 2,
                    return_dict_in_generate=True
                    )
                #exit()
                #response = tokenizer.decode(output_ids,skip_spectial_tokens=True)
                print(output_ids['sequences'])
                #exit()
                output_sequences = output_ids['sequences']
                response = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
                print("下面是模型最初始的回答")
                print(response)
                #exit()

                time_cost = time.time() - start

                new_tokens = output_ids["sequences"][0, input_ids.shape[1]:]
                sentence_index,hallu_logits_index,out_sub_text = compute_hallu_output(new_tokens, word, tokenizer)
                print("下面是out_sub_text的文本：")
                print(f"out_sub_text:{out_sub_text}")
                #exit()
                #print(f"索引位置:{hallu_logits_index}")
                #print(f"索引文本:{out_sub_text}")
                #print("new_tokens:",type(new_tokens))


                only_output_ids = output_ids["sequences"][:, input_ids.shape[1]:]

                #only_output_ids = output_ids.sequences[0,len(finla_input_input_ids[0]):]
                
                only_outputs_ids_before_hallu = only_output_ids[:, :sentence_index]
                #only_outputs_ids_before_hallu = only_output_ids[:sentence_index]
                
                new_input_ids = torch.cat([input_ids, only_outputs_ids_before_hallu], dim=1)


                #new_input_ids = torch.cat([input_ids,only_outputs_ids_before_hallu.unsqueeze(0)],1)
                torch.save(new_input_ids.cpu(), f'Qwen.pt')
                #print(new_input_ids)
                response_save = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
                print("下面是reponse_save的生成：")
                print("response_save",response_save)
                    #保存token后退出
                
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



