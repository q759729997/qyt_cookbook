# 多层感知器

- 多层感知器（multilayer perceptron，MLP）：分为输入层、隐藏层、输出层。隐藏层中的神经元和输入层中各个输入完全连接，输出层中的神经元和隐藏层中的各个神经元也完全连接。因此，多层感知机中的隐藏层和输出层都是全连接层（fully-connected layer，也叫稠密层dense layer）。

  ![多层感知器](imgs/%E5%A4%9A%E5%B1%82%E6%84%9F%E7%9F%A5%E5%99%A8.png)

## 损失函数

- 在机器学习里，将衡量误差的函数称为损失函数（loss function）。例如常见的平方误差函数也称为平方损失（square loss），*除以2是为了更方便的求导*：

$$
\ell^{(i)}(w, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2
$$

## 优化算法

- 当模型和损失函数形式较为简单时，误差最小化问题的解可以直接用公式表达出来。这类解叫作**解析解（analytical solution）**。然而，大多数深度学习模型并没有解析解，只能通过优化算法有限次迭代模型参数来尽可能降低损失函数的值。这类解叫作**数值解（numerical solution）**。
- **小批量随机梯度下降**（mini-batch stochastic gradient descent）：在每次迭代中，先随机均匀采样一个由固定数目训练数据样本所组成的小批量（mini-batch）$\mathcal{B}$，然后求小批量中数据样本的平均损失有关模型参数的导数（梯度），最后用此结果与预先设定的一个正数的乘积作为模型参数在本次迭代的减小量。

## 模型定义

- `torch.nn`模块定义了大量神经网络的层。“nn”是neural networks（神经网络）的缩写。它利用`autograd`来定义模型。。`nn`的核心数据结构是`Module`，既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络。`nn.Module`实例应该包含一些层以及返回输出的前向传播（forward）方法。

~~~python
class LinearNet(nn.Module):
    def __init__(self, n_feature):
    super(LinearNet, self).__init__()
    self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
    """定义前向传播"""
    y = self.linear(x)
    return y

net = LinearNet(n_feature=5)
print(net)  # 打印输出网络结构
"""输出
LinearNet(
(linear): Linear(in_features=5, out_features=1, bias=True)
)
"""
~~~

- 如下为使用`nn.Sequential`**搭建网络的三种方法**，`Sequential`是一个有序的容器，网络层将按照在传入`Sequential`的顺序依次被添加到计算图中。

~~~python
num_inputs = 5
# 写法一：module的name被自动设置为序号
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
)
print(net)
"""输出
Sequential(
(0): Linear(in_features=5, out_features=1, bias=True)
)
"""
# 写法二：add_module第一个参数为module的name
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......
print(net)
"""输出
Sequential(
(linear): Linear(in_features=5, out_features=1, bias=True)
)
"""
# 写法三
from collections import OrderedDict
net = nn.Sequential(
    OrderedDict([
        ('linear', nn.Linear(num_inputs, 1))
        # ......
    ])
)
print(net)
"""输出
Sequential(
(linear): Linear(in_features=5, out_features=1, bias=True)
)
"""
~~~

## 模型参数

- 通过`net.parameters()`来查看模型所有的可学习参数，此函数将返回一个生成器。

~~~python
net = nn.Sequential()
net.add_module('linear', nn.Linear(5, 1))
for param in net.parameters():
print(param)
"""输出
Parameter containing:
tensor([[-0.0567,  0.1161,  0.1954, -0.2397,  0.3248]], requires_grad=True)
Parameter containing:
tensor([-0.0782], requires_grad=True)
"""
~~~

- `net.named_parameters()`可以返回参数名称。

~~~python
for name, param in net.named_parameters():
print('name:{}, param:{}'.format(name, param))
"""
name:linear.weight, param:Parameter containing:
tensor([[-0.3299, -0.2503,  0.1922, -0.3915, -0.2623]], requires_grad=True)
name:linear.bias, param:Parameter containing:
tensor([-0.4374], requires_grad=True)
"""
~~~



