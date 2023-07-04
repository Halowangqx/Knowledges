# Intro 2 pytorch(cs285)

![Screenshot from 2023-05-07 13-01-00](/home/wangqx/Pictures/Screenshot from 2023-05-07 13-01-00.png)

## auto-backward

A few things to watch out for:

- You can't do any in-place operations on a tensor that has `requires_grad=True`. (This prevents you from inadvertently mutating it in a way that isn't tracked for backprop purposes.)

- You also can't convert a tensor with `requires_grad=True` to numpy (for the same reason as above). Instead, you need to detach it first, e.g. `y.detach().numpy()`.

  这是因为pytorch tensor 和其生成的numpy共用内存。所以为了使tensor不变，必须要detach之后再tonumpy()

- Even though `y.detach()` returns a new tensor, that tensor occupies the same memory as `y`. Unfortunately, PyTorch lets you make changes to `y.detach()` or `y.detach.numpy()` which will affect `y` as well! If you want to safely mutate the detached version, you should use `y.detach().clone()` instead, which will create a tensor in new memory.

RL Connection: You would want to be doing simulator-related tasks with numpy, convert to torch when doing model-related tasks, and convert back to feed output into simulator.



env.reset

env.step

reward function



# Function Usage

## Generate data

**All functions below here have arguments: **

```python
dtype= device= requires_grad=
```

**data**

``` python
# generate a tensor with values form the interval [start, end), taken with common difference step
torch.arrange(start=0, end, step=1)
```

**random data**

```python
# Tensor形状为size 区间为[0,1) 均匀分布的float
torch.rand(size)

# Tensor [low, high) 均匀分布int
torch.randint(low=0, high, size)

# return a tensor has the same shape as input, randomized
torch.randint_like(input, low=0, high)

# return a tensor following standard normal distribution
torch.randn(size)
# 
torch.randn_like(input, low=0, high)

# returns a random permutation of integers from 0 to n - 1
torch.randperm(n)

```



## Tensor operate

```python
##### torch.max(input, dim)
# 所有第dim维不同，其他维度都相同的数字进行比较
torch.max(input, idm)

##### torch.gather(input, dim, index)
# 可以选取完整数据中多个且乱序的数值(索引替换)
# dim： index中的值是用来替换dim维度的索引
index = [
    [x1, x2, x3],
    [y1, y2, y3],
]
torch.gather(input, 1, index)
# the index of x1 is (0, 0) the dim here is 1
# x1 stands for getting input[0, x1]
# similarly, y2 stands for input[1, y2]
```

### torch.unsqueeze/squeeze

```
torch.unsqueeze(input, dim, out=None)
```

- **作用**：扩展维度 返回一个新的张量，在dim位置插入一个维度，如果是负数就加一

```
torch.squeeze(input, dim=None, out=None)
```

将输入张量形状中的1 去除并返回。 如果输入是形如(A×1×B×1×C×1×D)，那么输出形状就为： (A×B×C×D)

当给定dim时，那么挤压操作只在给定维度上。例如，输入形状为: (A×1×B), `squeeze(input, 0)` 将会保持张量不变，只有用 `squeeze(input, 1)`，形状会变成 (A×B)。

### torch.scatter

`scatter(dim, index, src)` 

- **dim：**沿着哪个维度进行索引
- **index：**用来 scatter 的元素索引
- **src：**用来 scatter 的源元素，可以是一个标量或一个张量

```
self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
```



## hook

A 'hook' is a user-defined function that can be attached to a tensor or a module, allowing you to intercept and manipulate gradients during backpropagation or output during forward propagation. Hooks provide a way to customize the behavior of your neural networks at various stages of computation, giving you more control and flexibility over the training process.

### forward hook

called during forward propagation



```python
hook(module, input, output) -> None or modified output
##### The hook will be called every time after forward() has computed an output.
register_forward_hook(hook, )
```



```python
hook(module, input, output) -> None or modified output

module.register_module_forward_hook(hook)
```



### backward hook

called during the backward pass

```python
cache = []
##### The 'hook function'
# the function will recieve the grad of the tensor hooked on 
def func(grad) -> Tensor or None
    cache.append(grad)
    ### if return a tensor, it will be the new grad
    return grad

tensor.register_hook(func)
```



## detach()

`detach()` 方法用于创建一个新的张量，该张量与原始张量共享相同的数据，但不参与原始张量的计算图和梯度计算。换句话说，`detach()` 方法会将新张量与原始张量的梯度计算过程分离。



# Data

## Dataset

```python
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, root, transform = None, target_transform = None):
        # describe where the file; which is content; the transform; which is labels;
        self.root = root
        self.transform = transform
        self.content = 
        self.labels = 
        
    def __len__(self):
        # return the length of the dataset(s' content)
        return len(self.content)
    
    ##### return a tuple --(content, label)
    def __getitem__(self,idx):
        # load the content
        mg_path = os.path.join(self.root, "picture", self.imgs[idx])
		label_path = os.path.join(self.root, "descript", self.labels[idx])
		img = Image.open(img_path).convert("RGB")
        
        # load the labels
		label_file=open(label_path,'r')
		label_list=[]
		for line in label_file:
			data_line=line.strip("\n").split()
			for i in data_line:
				label_list.append(float(i))
		label=torch.tensor(np.array(label_list))
        
        ##### transform
		if self.transform is not None:  
			img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
          
		return img , label
```

## transform

***APPLY ON DATASET***

- *ToTensor()*

- *Lambda Transforms*

  ```python
  target_transform = Lambda(lambda y: torch.zeros(
      10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
  ```

- *transforms.Compose()*

```python
# 用经验值归一化
std_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

# compose
trans = transforms.Compose([
        transforms.ToTensor(),
        std_normalize])
```



## Dataloader



# buid a VGG

```python
import torch
import torch.nn as nn

class VGGNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGGNet, self).__init__()

        # VGG-16 configuration: 'M' stands for max-pooling layer
        vgg_block = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

        self.features = self._make_layers(vgg_block)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self, vgg_block):
        layers = []
        in_channels = 3
        for v in vgg_block:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
```



danceability tempo duration_ms valence energy

# visdom

> Visdom is a flexible visualization tool designed for machine learning tasks. It allows users to visualize data, logs, and other information in real-time, making it a powerful tool for monitoring and debugging.



## Running Visdom Server

Before using Visdom, start the Visdom server:

```
visdom
```

This will start the server at `http://localhost:8097`. You can access the Visdom interface through this URL.

## Basic Usage

### Import and create a Visdom Instance

```python
from visdom import Visdom
viz = Visdom()
```

### Basic Plotting

#### Line Plot

Create a simple line plot:

```python
import numpy as np

x = np.arange(0, 10, 0.1)
y = np.sin(x)
viz.line(Y=y, X=x, opts=dict(title='y = sin(x)'))
```

#### Scatter Plot

Create a scatter plot:

```python
x = np.random.rand(100)
y = np.random.rand(100)
viz.scatter(X=np.column_stack((x, y)), opts=dict(title='Scatter Plot'))
```

#### Bar Plot

Create a bar plot:

```python
data = np.random.rand(10)
viz.bar(X=data, opts=dict(title='Bar Plot'))
```

### Updating Plots

#### Line Plot Update

Update a line plot by adding new data points:

```python
win = viz.line(Y=np.random.rand(10), opts=dict(title='Updating Line Plot'))

for i in range(10):
    viz.line(Y=np.random.rand(1), win=win, update='append')
```

#### Scatter Plot Update

Update a scatter plot by adding new data points:

```python
win = viz.scatter(X=np.random.rand(100, 2), opts=dict(title='Updating Scatter Plot'))

for i in range(10):
    viz.scatter(X=np.random.rand(1, 2), win=win, update='append')
```

## Visualize a Datasete

```python
from torchvision import datasets, transforms

# 加载数据集
train_loader = torch.utils.data.DataLoader(datasets.MNIST(
    r'mnist-data',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])),batch_size=128,shuffle=True)
sample=next(iter(train_loader)) # 通过迭代器获取样本
print(sample[0].shape)
print(sample[1].shape)
viz = Visdom(env='main') # 注意此时创建了新环境，请在界面中选择该环境
# sample[0]为样本数据，sample[1]为类别，nrow=16表示每行显示16张图像
viz.images(sample[0], nrow=16, win='mnist', opts=dict(title='mnist'))
```



### Saving and Loading

Save the current state of the Visdom server to a JSON file:

```python
viz.save(envs=['main'])
```

Load a previously saved state:

```
visdom -load main.json
```

