## Saliency of LLaVA and Qwen-VL for exploring hallunction 
重新pip install 一下当前环境的transformer;
然后直接操作demo_step1.py和demo_step2.py,主要分为两步：
a)运行demo_step1.py:文件出现:则获取预输入token成功，

b)进行step2，计算梯度。

c)demo_step2：获取梯度,获取saliency：

d)得到显著性map和attention weight
