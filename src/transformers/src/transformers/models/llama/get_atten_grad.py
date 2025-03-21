import torch
import torch.nn.functional as F

batch_size = 2
seq_len = 3
d_k = 4
Q = torch.randn(batch_size, seq_len, d_k, requires_grad=True)
K = torch.randn(batch_size, seq_len, d_k, requires_grad=True)

scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

A = F.softmax(scores, dim=-1)  # A 是 attention weight

loss = A.sum()

loss.backward()

dA = A.grad  # loss 对 A 的梯度

print("Attention Weight A:", A)
print("Gradient of loss w.r.t. A:", dA)