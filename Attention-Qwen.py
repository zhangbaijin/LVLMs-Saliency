import torch
from transformers import AutoTokenizer, AutoConfig, Qwen2VLForConditionalGeneration

# 1. 加载模型及配置（确保 output_attentions=True, use_flash_attn=False）
model_path = "/root/autodl-tmp/Qwen2-VL-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
config.output_attentions = True
config.use_flash_attn = False
model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, attn_implementation="eager")
model.train()  # 因为需要梯度分析，必须处于训练模式
model.requires_grad_(True)

for idx, layer in enumerate(model.model.layers):
    print(f"Layer {idx} submodules: {list(layer._modules.keys())}")


# 2. 定义 AttentionAdapter 类（此处为简化版本）
class AttentionAdapter:
    def __init__(self):
        self.params = None

    def _forward(self, attn_weights):
        if self.params is None:
            self.params = torch.ones_like(attn_weights, requires_grad=True)
        else:
            # 每次 forward 时重置（如果希望参数持续学习，可注释此行）
            self.params.data = torch.ones_like(attn_weights)
        return attn_weights * self.params

    @property
    def grad(self):
        return self.params.grad

    def zero_grad(self, set_to_none=True):
        if set_to_none:
            self.params = None

# 实例化全局 adapter
adapter_instance = AttentionAdapter()

# 3. 定义前向钩子函数



def adapter_forward_hook(module, input, output):
    attn_weights = output[1]
    if attn_weights is None:
        return output
    new_attn_weights = adapter_instance._forward(attn_weights)
    return (output[0], new_attn_weights, output[2])


# 4. 注册前向钩子到所有 transformer 层中的注意力模块
# 根据你打印的模型属性，transformer 层存放在 model.model.layers


for layer in model.model.layers:
    layer.self_attn.register_forward_hook(adapter_forward_hook)

# 5. 构造输入并测试前向与反向传播
text = "你好，世界！"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"], output_attentions=True)
loss = outputs.loss
loss.backward()

if adapter_instance.params is not None and adapter_instance.params.grad is not None:
    print("Adapter 参数梯度形状：", adapter_instance.params.grad.shape)
else:
    print("未获得 adapter 参数梯度，请检查钩子是否生效。")

