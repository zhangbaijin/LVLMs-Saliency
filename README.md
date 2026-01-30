# [ICLR 2026] Hallucination Begins Where Saliency Drops
## Saliency of LLaVA and Qwen-VL for exploring hallunction 
### The motivation of this paper is
![image](https://github.com/zhangbaijin/LVLMs-Saliency/paper/motivation.png)

### The patterns of incorrect and correct tokens are as follows:
![image](https://github.com/zhangbaijin/LVLMs-Saliency/paper/pattern.png)

### After intervention, incorrect tokens became correct tokens, and the saliency score increased significantly.


### The patterns of incorrect and correct tokens are as follows:
![image](https://github.com/zhangbaijin/LVLMs-Saliency/paper/difference.png)


# For LLaVA1.5 
```
python demo_step1.py,python demo_step2.py
```


# For Qwen2-VL
```
python Qwen_step1.py,python Qwen_step2.py
```

# Different fusion strage for saliency-map
you can see that these methods are unable to distinguish the patterns of hallucinations and correct tokens. In contrast, attention*grad fusion is clearer in distinguishing important and unimportant tokens.

**Attention*Grad**:
![image](https://github.com/zhangbaijin/LVLMs-Saliency/blob/master/Nips-rebutal/nips-attention-grad.png)

'''
@inproceedings{
anonymous2026hallucination,
title={Hallucination Begins Where Saliency Drops},
author={Xiaofeng Zhang, Yuanchao Zhu, Chaochen Gu, Xiaosong Yuan, Qiyan Zhao, Jiawei Cao, Feilong Tang, Sinan Fan, Yaomin Shen, Chen Shen, Hao Tang },
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=sjnErRHXf3}
}
'''
