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
you can see ```Nips-Attention+Gradient.jpg```,```Nips-attention-substracted.jpg```,```Nips-attention*grad.jpg```,```Nips-concat+mlp.jpg```,```Nips-max-value.jpg```.
These methods are unable to distinguish the patterns of hallucinations and correct tokens. In contrast, multiplicative fusion is clearer in distinguishing important and unimportant tokens.
