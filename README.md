## Saliency of LLaVA and Qwen-VL for exploring hallunction 

Then directly operate demo_step1.py and demo_step2.py, mainly divided into two steps: 

# For LLaVA1.5 
```
python demo_step1.py
```
```
python demo_step2.py
```

# For Qwen2-VL
```
python Qwen_step1.py
```
```
python Qwen_step2.py
```
# Different fusion strage for saliency-map
you can see that these methods are unable to distinguish the patterns of hallucinations and correct tokens. In contrast, attention*grad fusion is clearer in distinguishing important and unimportant tokens.
**Attention*Grad**:
![image](https://github.com/zhangbaijin/LVLMs-Saliency/blob/master/Nips-rebutal/nips-attention-grad.png)

**Attention+Gad**:
![image](https://github.com/zhangbaijin/LVLMs-Saliency/blob/master/Nips-rebutal/nips-attention-add-grad.png)

**Attention-substracted**:
![image](https://github.com/zhangbaijin/LVLMs-Saliency/blob/master/Nips-rebutal/nips-attention-substracted-grad.png)

**Concat-mlp**:
![image](https://github.com/zhangbaijin/LVLMs-Saliency/blob/master/Nips-rebutal/nips-concat-mlp.png)

**max-value**:
![image](https://github.com/zhangbaijin/LVLMs-Saliency/blob/master/Nips-rebutal/nips-max-value.png)


