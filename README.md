# 🔥🔥🔥[ICLR 2026 Oral] Hallucination Begins Where Saliency Drops

[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/zhangbaijin/LVLMs-Saliency?style=social)](zhangbaijin/LVLMs-Saliency)


### The motivation of this paper is
![image](https://github.com/zhangbaijin/LVLMs-Saliency/blob/master/paper/motivation.png)

### The patterns of incorrect and correct tokens are as follows:
![image](https://github.com/zhangbaijin/LVLMs-Saliency/blob/master/paper/pattern.png)

### After intervention, incorrect tokens became correct tokens, and the saliency score increased significantly.
![image](https://github.com/zhangbaijin/LVLMs-Saliency/blob/master/paper/difference.png)


# For LLaVA1.5 
```
python demo_step1.py,python demo_step2.py
```


# For Qwen2-VL
```
python Qwen_step1.py,python Qwen_step2.py
```


## Citation
```bibtex
@inproceedings{
zhang-saliency,
title={Hallucination Begins Where Saliency Drops},
author={Xiaofeng Zhang, Yuanchao Zhu, Chaochen Gu, Xiaosong Yuan, Qiyan Zhao, Jiawei Cao, Feilong Tang, Sinan Fan, Yaomin Shen, Chen Shen, Hao Tang },
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=sjnErRHXf3}
}
```

## Acknowledgement

This repo is built on [LLaVA](https://github.com/haotian-liu/LLaVA) (models), [Qwen2.5-VL]([https://github.com/shikiw/OPER](https://github.com/QwenLM/Qwen3-VL)) (CHAIR evaluation) and [Label words]([https://github.com/pkunlp-icler/FastV](https://github.com/lancopku/label-words-are-anchors)). Many thanks for their efforts. The use of our code should also follow the original licenses.

