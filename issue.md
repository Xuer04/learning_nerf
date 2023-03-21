## 记录一下复现过程中出现的问题

### NeRF 相关
TODO

### Pytorch 相关

#### 问题 1:

报错:
```py
>>> x = torch.Tensor(..., device=torch.device('cuda:0'))
Error: legacy constructor expects device type: cpubut device type: cuda was passed
```

原因分析:

一般创建张量的方式为
```py
# method 1
x = torch.Tensor(...)

# method 2
x = torch.tensor(...)
```

方式一默认是在把张量放在 CPU（内存）中的。如果我们要使用 GPU 来训练模型，那么就还必须进行一步将张量复制到 GPU 上的操作，如果直接在 GPU 上创建张量是不支持的, 于是会得到上述报错, 但是方式二 `torch.tensor()` 是可以直接指定张量在 GPU 上创建的

```py
>>> y = torch.tensor([2, 3, 4], device=torch.device("cuda:0"))
>>> y
tensor([2, 3, 4], device='cuda:0')
>>> y.type()
'torch.cuda.LongTensor'
```

查询 Pytorch 的 [官方文档](https://pytorch.org/docs/stable/tensors.html) 会发现其中有一条:

> **torch.Tensor is an alias for the default tensor type (torch.FloatTensor).**

`torch.tensor` 是一个方法, 将输入参数转化为 `Tensor` 类型, `torch.Tensor` 是一个 class, `torch.tensor` 会从推断输入数据类型

推荐以后都使用 `torch.tensor` 的方式来创建张量

### NumPy 相关
TODO
