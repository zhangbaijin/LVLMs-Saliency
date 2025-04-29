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
