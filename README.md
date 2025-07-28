## Saliency of LLaVA and Qwen-VL for exploring hallunction 

Re-cpip install the current environment of the transformer; 
Then directly operate demo_step1.py and demo_step2.py, mainly divided into two steps: 

# For LLaVA1.5 
```
python demo_step1.py
```
the file appears: then get the pre-input token successfully, 

```
python demo_step2.py
```
calculate the gradient, get the significance map and the attention weight.


# For Qwen2-VL
```
python Qwen_step1.py
```
the file appears: then get the pre-input token successfully, 

```
python Qwen_step2.py
```
calculate the gradient, get the significance map and the attention weight.

# Different fusion strage for saliency-map
you can see that these methods are unable to distinguish the patterns of hallucinations and correct tokens. In contrast, attention*grad fusion is clearer in distinguishing important and unimportant tokens.
**Attention*Grad**:
![image](https://github.com/zhangbaijin/LVLMs-Saliency/blob/master/Nips-attention*grad.jpg)

**Attention+rad**:
![image](https://github.com/zhangbaijin/LVLMs-Saliency/blob/master/Nips-Attention%20%2B%20Gradient.jpg)

**Attention-substracted**:
![image](https://github.com/zhangbaijin/LVLMs-Saliency/blob/master/Nips-attention-subtracted.jpg)

**Concat-mlp**:
![image](https://github.com/zhangbaijin/LVLMs-Saliency/blob/master/Nips-concat%2BMLP.jpg)

**max-value**:
![image](https://github.com/zhangbaijin/LVLMs-Saliency/blob/master/Nips-max-value.jpg)


