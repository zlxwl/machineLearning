import  torch
# a = torch.rand(4, 4)
# b = torch.ones_like(a, dtype=torch.double)
# print(a)
# print(b)
# print(b.view(2, -1))
# print(b.view(1, -1))

# tensor 转 numpy
# c = torch.rand(4, 4)
# d = c.numpy()
# print(c)
# print(d)
# c[1, 2] = 3
# print(c)
# print(d)

# numpy 转 tensor
# import numpy as np
# a = np.ones(4)
# b = torch.from_numpy(a)
# print(a)
# print(b)
# a = a+1
# print(a)
# print(b)

# numpy 实现神经网络.
# import numpy as np
# N, D_in, H, D_out = 64, 1000, 100, 10
# x = np.random.randn(N, D_in)
# y = np.random.randn(N, D_out)
#
# w1 = np.random.randn(D_in, H)
# w2 = np.random.randn(H, D_out)
#
# learning_rate = 1e-6
# for i in range(500):
#      h = x.dot(w1)
#      h_relu = np.maximum(h, 0) # N * H
#      y_pred = h_relu.dot(w2) # N * D_out
#      # loss
#      loss = np.square(y_pred - y).sum()
#      print(i, loss)
#
#      # backward loss
#      # compute the gradient
#      grad_y_pred = 2 *(y_pred - y)
#      grad_w2 = h_relu.T.dot(grad_y_pred)
#
#      grad_h_relu = grad_y_pred.dot(w2.T)
#      grad_h = grad_h_relu.copy()
#      grad_h[h<0] = 0
#      grad_w1 = x.T.dot(grad_h)
#
#      w1 -= learning_rate * grad_w1
#      w2 -= learning_rate * grad_w2
#
# h = x.dot(w1)
# a = np.maximum(0, h)
# y_pred = a.dot(w2)
#
# print((y_pred - y).sum())

# tensor版本的神经网络。
# import torch
# N, D_in, H, D_out = 64, 1000, 100, 10
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)
#
# w1 = torch.randn(D_in, H)
# w2 = torch.randn(H, D_out)
#
# learning_rate = 1e-6
# for i in range(500):
#      h = x.mm(w1)
#      h_relu = h.clamp(min=0) # N * H
#      y_pred = h_relu.mm(w2) # N * D_out
#      # loss
#      loss = (y_pred - y).pow(2).sum()
#      print(i, loss.item())
#
#      # backward loss
#      # compute the gradient
#      grad_y_pred = 2 *(y_pred - y)
#      grad_w2 = h_relu.t().mm(grad_y_pred)
#
#      grad_h_relu = grad_y_pred.mm(w2.t())
#      grad_h = grad_h_relu.to()
#      grad_h[h<0] = 0
#      grad_w1 = x.t().mm(grad_h)
#
#      w1 -= learning_rate * grad_w1
#      w2 -= learning_rate * grad_w2
#
# h = x.mm(w1)
# a = h.clamp(min=0)
# y_pred = a.mm(w2)
#
# print((y_pred - y).sum().item())

# tensor的实际应用版本。只需要定义前向网络，反向传播的梯度计算自动进行不需要自己定义和计算。
# import torch
# N, D_in, H, D_out = 64, 1000, 100, 10
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)
#
# w1 = torch.randn(D_in, H, requires_grad=True)
# w2 = torch.randn(H, D_out, requires_grad=True)
#
# learning_rate = 1e-6
# for i in range(500):
#      h = x.mm(w1)
#      h_relu = h.clamp(min=0) # N * H
#      y_pred = h_relu.mm(w2) # N * D_out
#      # loss
#      loss = (y_pred - y).pow(2).sum()
#      print(i, loss.item())
#
#      # backward loss
#      # compute the gradient
#      loss.backward()
#      # with torch.no_grad:
#      # grad_y_pred = 2 *(y_pred - y)
#      # grad_w2 = h_relu.t().mm(grad_y_pred)
#      # grad_h_relu = grad_y_pred.mm(w2.t())
#      # grad_h = grad_h_relu.to()
#      # grad_h[h<0] = 0
#      # grad_w1 = x.t().mm(grad_h)
#      # 每次更新梯度时需要将梯度清零。
#      with torch.no_grad():
#         w1 -= learning_rate * w1.grad
#         w2 -= learning_rate * w2.grad
#         w1.grad.zero_()
#         w2.grad.zero_()
#
# h = x.mm(w1)
# a = h.clamp(min=0)
# y_pred = a.mm(w2)
#
# print((y_pred - y).sum().item())

# ## torch.nn来做。
# import torch
# # import torch.nn as nn
# N, D_in, H, D_out = 64, 1000, 100, 10
#
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)
#
# w1 = torch.randn(D_in, H, requires_grad=True)
# w2 = torch.randn(H, D_out, requires_grad=True)
#
# learning_rate = 1e-6
# model = torch.nn.Sequential(
#     torch.nn.Linear(D_in, H, bias=False),
#     torch.nn.ReLU(),
#     torch.nn.Linear(H, D_out, bias=False),
# )
# # torch.nn.init.normal_(model[0].weight)
# # torch.nn.init.normal_(model[2].weight)
#
# loss_fn = torch.nn.MSELoss(reduction='sum')
# learning_rate = 1e-5
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# for i in range(500):
#      # h = x.mm(w1)
#      # h_relu = h.clamp(min=0) # N * H
#      # y_pred = h_relu.mm(w2) # N * D_out
#      y_pred = model(x)
#      # loss
#      # loss = (y_pred - y).pow(2).sum()
#      loss = loss_fn(y_pred, y)
#      print(i, loss.item())
#      # backward loss
#      # compute the gradient
#      model.zero_grad()
#      loss.backward()
#      # with torch.no_grad:
#      # grad_y_pred = 2 *(y_pred - y)
#      # grad_w2 = h_relu.t().mm(grad_y_pred)
#      # grad_h_relu = grad_y_pred.mm(w2.t())
#      # grad_h = grad_h_relu.to()
#      # grad_h[h<0] = 0
#      # grad_w1 = x.t().mm(grad_h)
#      # 每次更新梯度时需要将梯度清零。
#      # with torch.no_grad():
#      #    w1 -= learning_rate * w1.grad
#      #    w2 -= learning_rate * w2.grad
#      #    w1.grad.zero_()
#      #    w2.grad.zero_()
#      #  手动更新梯度。
#      # with torch.no_grad():
#      #     for param in model.parameters():
#      #         param -= learning_rate * param.grad
#      # 使用tensor去更新梯度。
#      optimizer.step()
#
# print((y_pred - y).sum().item())
# print(model[0].weight)

## torch.nn来做。
import torch
N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        y_pred = self.linear2(self.linear1(x).clamp(min=0))
        return y_pred


model = TwoLayerNet(D_in, H, D_out)
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(500):
     y_pred = model(x)
     # loss
     loss = loss_fn(y_pred, y)
     print(i, loss.item())
     # backward loss
     # compute the gradient
     model.zero_grad()
     loss.backward()
     optimizer.step()

print((y_pred - y).sum().item())
print(model[0].weight)