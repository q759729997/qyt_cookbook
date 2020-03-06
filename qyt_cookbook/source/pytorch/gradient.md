# 梯度计算

- PyTorch提供的[autograd](https://pytorch.org/docs/stable/autograd.html)包能够根据输入和前向传播过程自动构建计算图，并执行反向传播。

## 梯度自动计算

- 将Tensor设置`requires_grad=True`，它将开始追踪(track)在其上的所有操作（这样就可以利用链式法则进行梯度传播了）。完成计算后，可以调用`.backward()`来完成所有梯度计算。此`Tensor`的梯度将累积到`.grad`属性中。

~~~python
x = torch.ones(2)  # # 缺失情况下默认 requires_grad = False
print(x, x.requires_grad)  # tensor([1., 1.]) False
x.requires_grad_(True)  # 通过.requires_grad_()用in-place的方式改变requires_grad属性
print(x, x.requires_grad)  # tensor([1., 1.], requires_grad=True) True
x = torch.ones(2, requires_grad=True)
print(x, x.requires_grad)  # tensor([1., 1.], requires_grad=True) True
~~~

- **注意：在`y.backward()`时**，如果`y`是标量，则不需要为`backward()`传入任何参数；否则，需要传入一个与`y`同形的`Tensor`。参考：backward为什么有一个grad_variables参数<https://zhuanlan.zhihu.com/p/29923090>

~~~python
x = torch.ones(2, requires_grad=True)
y = x * x
print(y, y.grad_fn)  # tensor([1., 1.], grad_fn=<MulBackward0>) <MulBackward0 object at 0x000001521DC5BF60>
y = y.sum()
print(y, y.grad_fn)  # tensor(2., grad_fn=<SumBackward0>) <SumBackward0 object at 0x000001521DC5BF60>
print(x.grad)  # None
print(y.backward())  # None 等价于 y.backward(torch.tensor(1.))
print(x.grad)  # tensor([2., 2.])
~~~

- 调用`.detach()`将其从追踪记录中分离出来，这样就可以防止将来的计算被追踪，这样梯度就传不过去了。此外，还可以用`with torch.no_grad()`将不想被追踪的操作代码块包裹起来，这种方法在**评估模型的时候很常用**，因为在评估模型时，我们并不需要计算可训练参数（`requires_grad=True`）的梯度。

~~~python
x = torch.ones(2, requires_grad=True)
y = (x * x).sum()
print(y, y.grad_fn)  # tensor(2., grad_fn=<SumBackward0>) <SumBackward0 object at 0x000001A08211D1D0>
with torch.no_grad():  # 中断梯度追踪
    y = (x * x).sum()
    print(y, y.grad_fn)  # tensor(2.) None
~~~

- 如果我们想要修改`tensor`的数值，但是又不希望被`autograd`记录（即不会影响反向传播），那么我么可以对`tensor.data`进行操作。**`requires_grad=True`的叶子节点，不能进行`in-place`操作。**

~~~
x = torch.ones(1, requires_grad=True)
print(x.data)  # tensor([1.]),data也是一个tensor
print(x.data.requires_grad)  # False，已经是独立于计算图之外
y = 2 * x
x.data *= 100  # 只改变了值，不会记录在计算图，所以不会影响梯度传播
y.backward()
print(x)  # tensor([100.], requires_grad=True)，更改data的值也会影响tensor的值
print(x.grad)  # tensor([2.])
print('{} 不使用data，直接操作tensor {}'.format('-'*15, '-'*15))
# x = torch.ones(1, requires_grad=True)
# y = 2 * x
# x *= 100  # RuntimeError: a leaf Variable that requires grad has been used in an in-place operation.
~~~

- `Function`是另外一个很重要的类。`Tensor`和`Function`互相结合就可以构建一个记录有整个计算过程的有向无环图（DAG）。每个`Tensor`都有一个`.grad_fn`属性，该属性即创建该`Tensor`的`Function`, 就是说该`Tensor`是不是通过某些运算得到的，若是，则`grad_fn`返回一个与这些运算相关的对象，否则是None。像x这种直接创建的称为叶子节点，叶子节点对应的`grad_fn`是`None`。

~~~python
x = torch.ones(2)
print(x.grad_fn)  # None
y = x * 2
print(y)  # tensor([2., 2.])
print(x.grad_fn, y.grad_fn)  # None None
print('{} 设置requires_grad=True {}'.format('-'*15, '-'*15))
x = torch.ones(2, requires_grad=True)
print(x.grad_fn)  # None 直接创建的Tensor没有grad_fn，被称为叶子节点。
y = x * 2
print(x.is_leaf, y.is_leaf)  # True False
print(y)  # tensor([2., 2.], grad_fn=<MulBackward0>)
print(x.grad_fn, y.grad_fn)  # None <MulBackward0 object at 0x00000282A67BCF28>
~~~

### 雅克比矩阵

- 数学上，如果有一个函数值和自变量都为向量的函数  $\vec{y} =f(\vec{x})$, 那么 $\vec{y}$关于 $\vec{x}$ 的梯度就是一个雅可比矩阵`（Jacobian matrix）`:

$$
J=(\frac{\partial y_{1}}{\partial x_{1}}⋯\frac{\partial y_{1}}{\partial x_{n}} ⋮⋱⋮ \frac{\partial y_{m}}{\partial x_{1}}⋯\frac{\partial y_{m}}{\partial x_{n}})
$$

- 在向量分析中，**雅可比矩阵**是函数的一阶偏导数以一定方式排列成的矩阵，其行列式称为**雅可比行列式**。`Jacobian`可以发音为`[ja ˈko bi ən]`。假设$F:\mathbb{R}_{n}\rightarrow \mathbb{R}_{m}$ 是一个从n维欧式空间映射到m维欧式空间的函数。这个函数由m个实数组成：$y_{1}(x_{1},\cdots ,x_{n}),\cdots,y_{m}(x_{1},\cdots ,x_{n})$ 。这些函数的偏导数（如果存在）可以组成一个m行n列的矩阵，这个矩阵就是所谓的雅克比矩阵：

$$
\begin{bmatrix}
\frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\ 
\vdots  & \ddots  & \vdots\\ 
\frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
\end{bmatrix}
$$

- 而`torch.autograd`这个包就是用来计算一些雅克比矩阵的乘积的。例如，如果 $v$ 是一个标量函数的 $l=g(\vec{y} )$ 的梯度：

$$
v=(\frac{\partial l}{\partial y_{1}}⋯\frac{\partial l}{\partial y_{m}})
$$

- 那么根据链式法则我们有 $l$ 关于 $\vec{x}$ 的雅克比矩阵就为:

$$
vJ=(\frac{\partial l}{\partial y_{1}}⋯\frac{\partial l}{\partial y_{m}})(\frac{\partial y_{1}}{\partial x_{1}}⋯\frac{\partial y_{1}}{\partial x_{n}} ⋮⋱⋮ \frac{\partial y_{m}}{\partial x_{1}}⋯\frac{\partial y_{m}}{\partial x_{n}})=(\frac{\partial _{l}}{\partial x_{1}}⋯\frac{\partial l}{\partial x_{n}})
$$

- 注意：grad在反向传播过程中是累加的(`accumulated`)，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以一般在反向传播之前需把梯度清零。

~~~python
x = torch.ones(2, requires_grad=True)
y = (x * x).sum()
y.backward()
print(x.grad)  # tensor([2., 2.])
y = (x * x).sum()
y.backward()
print(x.grad)  # tensor([4., 4.])；梯度累加
x.grad.data.zero_()  # 梯度清零
y = (x * x).sum()
y.backward()
print(x.grad)  # tensor([2., 2.])
~~~

