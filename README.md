# [ICLR 2026] Hallucination Begins Where Saliency Drops

[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/zhangbaijin/LVLMs-Saliency?style=social)](zhangbaijin/LVLMs-Saliency)


### The motivation of this paper is
![image](https://github.com/zhangbaijin/LVLMs-Saliency/blob/master/paper/motivation.png)

### The patterns of incorrect and correct tokens are as follows:
![image](https://github.com/zhangbaijin/LVLMs-Saliency/blob/master/paper/pattern.png)

### After intervention, incorrect tokens became correct tokens, and the saliency score increased significantly.


### The patterns of incorrect and correct tokens are as follows:
![image](https://github.com/zhangbaijin/LVLMs-Saliency/blob/master/paper/difference.png)


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

## Citation
```bibtex
@inproceedings{
anonymous2026hallucination,
title={Hallucination Begins Where Saliency Drops},
author={Xiaofeng Zhang, Yuanchao Zhu, Chaochen Gu, Xiaosong Yuan, Qiyan Zhao, Jiawei Cao, Feilong Tang, Sinan Fan, Yaomin Shen, Chen Shen, Hao Tang },
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=sjnErRHXf3}
}
'''

## Acknowledgement

This repo is built on [LLaVA](https://github.com/haotian-liu/LLaVA) (models), [OPERA](https://github.com/shikiw/OPERA) (CHAIR evaluation) and [FastV](https://github.com/pkunlp-icler/FastV) (Image Token Truncation). Many thanks for their efforts. The use of our code should also follow the original licenses.


