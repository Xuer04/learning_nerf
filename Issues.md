## 记录一下复现过程中出现的问题

**Todo 表示还没解决, Fixed 表示解决了**

### 框架相关

#### 问题 1: Fixed

训练中途结束后，如果要加载预训练模型继续训练时，会报错，报错位置为这个文件 [/lib/utils/net_utils.py](https://github.com/Xuer04/learning_nerf/blob/master/lib/utils/net_utils.py#L323)

报错信息:
```py
>>> optimizer.step()
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

原因分析:

当从与训练模型中加载 optimizer 的参数时，默认是在 CPU 上的，因此需要在`load_model`的时候把 optimizer 的参数传到 cuda 里，代码如下:
```py
optim.load_state_dict(pretrained_model['optim'])
for state in optim.state.values():
    for k, v in state.items():
        if torch.is_tensor(v):
            state[k] = v.cuda()
```

### NeRF 相关

#### 问题 1: Fixed

- 在光线采样的时候, 为了防止 OOM, 使用了`chunk_size`, `N_rays`的默认值是 1024, `chunk_size`的默认值是 4096, 好像没有起到 **batchify** 的作用?

> **batchify** 主要是在 evaluate 的时候做, 这个时候的`N_rays`应该是 `H * W`, 大于`chunk_size`

- 在光线上采样空间点得到 (N_rays * N_samples) 个空间点的时候应该做一下 **batchify**?

> 可以不用做, 在`batchify_rays`的时候做就可以了

#### 问题 2: Fixed

训练的时候 loss 一直下不去, 停在 0.4 左右不动

> rendering部分写的有问题, 在sampling的时候near和far写错了, 然后near和far一样, 导致sampling的结果一直为0

之前的代码:
```py
bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]
```

Fix 之后的代码:
```py
bounds = torch.reshape(ray_batch[..., [6, 8]], [-1, 1, 2])
near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]
```

### Pytorch 相关

#### 问题 1: Fixed

报错信息:
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

方式一默认是在把张量放在 CPU（内存）中的。如果我们要使用 GPU 来训练模型，那么就还必须进行一步将张量复制到 GPU 上的操作，如果直接在 GPU 上创建张量是不支持的, 于是会得到上述报错, 但是方式二`torch.tensor()`是可以直接指定张量在 GPU 上创建的

```py
>>> y = torch.tensor([2, 3, 4], device=torch.device("cuda:0"))
>>> y
tensor([2, 3, 4], device='cuda:0')
>>> y.type()
'torch.cuda.LongTensor'
```

查询 Pytorch 的 [官方文档](https://pytorch.org/docs/stable/tensors.html) 会发现其中有一条:

> **torch.Tensor is an alias for the default tensor type (torch.FloatTensor).**

`torch.tensor`是一个方法, 将输入参数转化为`Tensor`类型, `torch.Tensor`是一个 class, `torch.tensor`会从推断输入数据类型

推荐以后都使用`torch.tensor`的方式来创建张量

#### 问题 2: Fixed

在从 DataLoader 中读取 batch 的时候出现了这个错误

报错信息:

```py
AttributeError: 'str' object has no attribute 'to'
```

原因分析:

这里主要是用了 `to_cuda()` 这个函数导致的报错

`to_cuda()`函数的内容如下:

```py
def to_cuda(batch, device=torch.device('cuda:0')):
    if isinstance(batch, tuple) or isinstance(batch, list):
        #batch[k] = [b.cuda() for b in batch[k]]
        #batch[k] = [b.to(self.device) for b in batch[k]]
        batch = [to_cuda(b, device) for b in batch]
    elif isinstance(batch, dict):
        #batch[k] = {key: self.to_cuda(batch[k][key]) for key in batch[k]}
        batch_ = {}
        for key in batch:
            if key == 'meta':
                batch_[key] = batch[key]
            else:
                batch_[key] = to_cuda(batch[key], device)
        batch = batch_
    else:
        # batch[k] = batch[k].cuda()
        batch = batch.to(device)
    return batch
```

所以问题出现的原因是 `batch`中有变量的类型是 `string`

### NumPy 相关

#### 问题 1: Fixed

报错信息:

```py
python run.py --type evaluate --cfg_file configs/nerf/nerf.yaml
...
Error: Object of type float32 is not JSON serializable
```

原因分析:

Python 字典数据格式化写入 JSON 文件时不支持`np.float32`类型，类似还有`np.int32`、`np.array`等, 需要先转换为 Python 类型

```py
ret = {item: float(ret[item]) for item in ret}

# 另一种方法
ret = {item: ret[item].astype(float) for item in ret}
```
