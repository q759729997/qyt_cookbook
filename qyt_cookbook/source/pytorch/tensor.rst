==================
Tensor数据操作
==================

-  ``Tensor``\ 这个单词一般可译作“张量”，张量可以看作是一个多维数组。标量可以看作是0维张量，向量可以看作1维张量，矩阵可以看作是二维张量。

创建Tensor
######################

-  创建\ ``Tensor``\ 函数列表：

=================================  =========================
函数                               功能
=================================  =========================
Tensor(\*sizes)                     基础构造函数
tensor(data,)                      类似np.array的构造函数
empty(\*sizes)                      未初始化的Tensor
ones(\*sizes)                       全1的Tensor
zeros(\*sizes)                      全0的Tensor
eye(\*sizes)                        对角线为1，其他为0
arange(s,e,step)                   从s到e，步长为step
linspace(s,e,steps)                从s到e，均匀切分成steps份
rand/randn(\*sizes)                 均匀/标准分布
normal(mean,std)/uniform(from,to)  正态分布/均匀分布
randperm(m)                        随机排列
=================================  =========================

-  重要参数 ``dtype`` ，\ **默认数据类型**\ 为 ``torch.float`` ，32位浮点数。常用数据类型。

============== ============================= ==================
数据类型       dtype                         CPU tensor
============== ============================= ==================
32位浮点数     torch.float32 或 torch.float  torch.FloatTensor
64位浮点数     torch.float64 或 torch.double torch.DoubleTensor
32位带符号整数 torch.int32 或 torch.int      torch.IntTensor
64位带符号整数 torch.int64 或 torch.long     torch.LongTensor
布尔型         torch.bool                    torch.BoolTensor
============== ============================= ==================

.. code:: python

    x = torch.ones(1, 2)
    print('x:{}, dtype:{}'.format(x, x.dtype))
    # 输出：x:tensor([[1., 1.]]), dtype:torch.float32

-  通过现有的 ``Tensor`` 来创建，此方法会默认重用输入 ``Tensor`` 的一些属性，例如数据类型，除非自定义数据类型。

.. code:: python

    x = torch.tensor([1, 2, 3])
    print('x:{}, dtype:{}'.format(x, x.dtype))
    # 输出 x:tensor([1, 2, 3]), dtype:torch.int64
    x = x.new_empty(1, 5)
    print('x:{}, dtype:{}'.format(x, x.dtype))
    # 输出 x:tensor([[0, 0, 0, 0, 0]]), dtype:torch.int64
    x = x.new_empty(1, 5, dtype=torch.double)
    print('x:{}, dtype:{}'.format(x, x.dtype))
    # x:tensor([[0., 0., 0., 0., 0.]], dtype=torch.float64), dtype:torch.float64

常用属性
######################

-  ``Tensor`` 形状获取，通过 ``shape`` 或者 ``size()`` 来获取 ``Tensor`` 的形状。返回的 ``torch.Size`` 其实相当于一个 ``tuple`` , 支持所有 ``tuple`` 的操作。

.. code:: python

    x = torch.randn(8, 28, 28)
    print(x.shape)  # 输出 torch.Size([8, 28, 28])
    print(type(x.shape))  # 输出 <class 'torch.Size'>
    print(x.size())  # 输出 torch.Size([8, 28, 28])
    print(type(x.size()))  # 输出 <class 'torch.Size'>
    print(x.size()[1])  # 输出 28
    # x.shape[1] += 1，TypeError: 'torch.Size' object does not support item assignment

数学计算
######################

-  几种不同形式的相加运算，计算结果都一致（\ **计算时需要注意数据类型**\ ）。注：PyTorch操作inplace版本都有后缀\ ``_`` , 例如\ ``x.copy_(y), x.t_()`` 。

.. code:: python

    x = torch.tensor([1, 2, 3])
    y = torch.tensor([0.1, 0.2, 0.3])
    # 形式一：直接相加
    print(x + y)  # tensor([1.1000, 2.2000, 3.3000])
    # 形式二：使用torch.add函数相加
    print(torch.add(x, y))  # tensor([1.1000, 2.2000, 3.3000])
    # 形式三：使用torch.add函数相加，将结果输出至指定的tensor
    result = torch.empty_like(y)
    print(torch.add(x, y, out=result))  # tensor([1.1000, 2.2000, 3.3000])
    print(result)  # tensor([1.1000, 2.2000, 3.3000])
    # 形式四：inplace版本，y的值会变化
    print(y.add_(x))  # tensor([1.1000, 2.2000, 3.3000])
    print(y)  # tensor([1.1000, 2.2000, 3.3000])

-  **广播机制**\ ：当对两个形状不同的\ ``Tensor``\ 按元素运算时，可能会触发广播（broadcasting）机制：先适当复制元素使这两个\ ``Tensor``\ 形状相同后再按元素运算。

.. code:: python

    x = torch.eye(2, 2)
    print(x)  # tensor([[1., 0.], [0., 1.]])
    print(x + torch.ones(1))  # tensor([[2., 1.], [1., 2.]])
    print(x + torch.ones(1, 2))  # tensor([[2., 1.], [1., 2.]])
    print(x + torch.ones(2, 1))  # tensor([[2., 1.], [1., 2.]])
    print(x + torch.ones(2, 2))  # tensor([[2., 1.], [1., 2.]])

- ``max`` 函数。传入一个tensor时，返回其最大的值，传入两个时，则进行元素比较，每个元素取较大的，形状不同时要能够进行广播计算。

.. code-block:: python

    x = torch.tensor([-1, 0, 1])
    print(x)
    print(torch.max(x))  # tensor(1)
    print(torch.max(x, torch.tensor(0)))  # tensor([0, 0, 1])

-  一些线性代数函数：

================================= =================================
函数                              功能
================================= =================================
trace                             对角线元素之和(矩阵的迹)
diag                              对角线元素
triu/tril                         矩阵的上三角/下三角，可指定偏移量
mm/bmm                            矩阵乘法，batch的矩阵乘法
addmm/addbmm/addmv/addr/baddbmm.. 矩阵运算
t                                 转置
dot/cross                         内积/外积
inverse                           求逆矩阵
svd                               奇异值分解
================================= =================================

按维度计算
***************************

-  对多维\ ``Tensor``\ 按维度操作。可以只对其中同一列（\ ``dim=0``\ ）或同一行（\ ``dim=1``\ ）的元素求和，并在结果中保留行和列这两个维度（\ ``keepdim=True``\ ）。求和操作中，被计算的dim最后变为size=1。

.. code:: python

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(x.shape)  # torch.Size([2, 3])
    print(x.sum(dim=0, keepdim=True))  # tensor([[5, 7, 9]])
    print(x.sum(dim=0, keepdim=True).shape)  # torch.Size([1, 3])
    print(x.sum(dim=1, keepdim=True))  # tensor([[ 6], [15]])
    print(x.sum(dim=1, keepdim=True).shape)  # torch.Size([2, 1])

点乘与矩阵乘
***************************

- **矩阵乘** ，行乘列求和作为输出元素, (1, 2)矩阵乘(2, 3)变为(1, 3)
- **点乘** ，输入tensor形状一致，对应位置的元素相乘

.. code-block:: python

    # 矩阵乘是矩阵的运算，点乘是矩阵中元素的运算
    x = torch.eye(1, 2)
    print(x)  # tensor([[1., 0.]])
    y = torch.ones(2, 3)
    print(y)
    # 矩阵乘，行乘列求和作为输出元素, (1, 2)矩阵乘(2, 3)变为(1, 3)
    print(torch.matmul(x, y))  # tensor([[1., 1., 1.]])
    y = torch.tensor([2, 2])
    print(y)  # tensor([2, 2])
    # 点乘，输入tensor形状一致，对应位置的元素相乘
    print(x * y)  # tensor([[2., 0.]])

小批量乘法
***************************

- 我们可以使用小批量乘法运算 ``bmm`` 对两个小批量中的矩阵一一做乘法。假设第一个小批量中包含 :math:`n` 个形状为 :math:`a\times b` 的矩阵 :math:`\boldsymbol{X}_1, \ldots, \boldsymbol{X}_n` ，第二个小批量中包含 :math:`n` 个形状为 :math:`b\times c` 的矩阵 :math:`\boldsymbol{Y}_1, \ldots, \boldsymbol{Y}_n` 。这两个小批量的矩阵乘法输出为 :math:`n` 个形状为 :math:`a\times c` 的矩阵 :math:`\boldsymbol{X}_1\boldsymbol{Y}_1, \ldots, \boldsymbol{X}_n\boldsymbol{Y}_n` 。因此，给定两个形状分别为( :math:`n` ,  :math:`a` ,  :math:`b` )和( :math:`n` ,  :math:`b` ,  :math:`c` )的`Tensor`，小批量乘法输出的形状为( :math:`n` ,  :math:`a` ,  :math:`c` )。

.. code-block:: python

    x = torch.ones(8, 2, 3)
    y = torch.zeros(8, 3, 4)
    print(torch.bmm(x, y).shape)  # torch.Size([8, 2, 4])

索引操作
######################

-  使用类似NumPy的索引操作来访问\ ``Tensor``\ 的一部分，需要注意的是：\ **索引出来的结果与原数据共享内存，即修改一个，另一个会跟着修改。**

.. code:: python

    x = torch.eye(2, 2)
   print(x)  # tensor([[1., 0.], [0., 1.]])
    y = x[0, :]  # 取第一维位置0，第二维全部
    print(y)  # tensor([1., 0.])
    y += 3  # 源tensor，也就是x也随之改变
    print(y)  # tensor([4., 3.])
    print(x)  # tensor([[4., 3.], [0., 1.]])

-  高级索引选择函数：

+---------------------------------+-----------------------------------+
| 函数                            | 功能                              |
+=================================+===================================+
| index_select(input, dim, index) | 在指定维度                        |
|                                 | dim上选取，比如选取某些行、某些列 |
+---------------------------------+-----------------------------------+
| masked_select(input, mask)      | 例子如上                          |
|                                 | ，a[a>0]，使用ByteTensor进行选取  |
+---------------------------------+-----------------------------------+
| nonzero(input)                  | 非0元素的下标                     |
+---------------------------------+-----------------------------------+
| gather(input, dim, index)       | 根据index，在dim维度              |
|                                 | 上选取数据，输出的size与index一样 |
+---------------------------------+-----------------------------------+

-  ``gather``,根据index，在dim维度上选取数据,输出的size与index一样。

.. code:: python

    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    print(y_hat.shape)  # torch.Size([2, 3])
    index = torch.LongTensor([0, 2]).view(-1, 1)
    print(index.shape)  # torch.Size([2, 1])
    print(index)
    """
    tensor([[0],
    [2]])
    """
    print(y_hat.gather(dim=1, index=index).shape)  # torch.Size([2, 1])
    print(y_hat.gather(dim=1, index=index))
    """
    tensor([[0.1000],
    [0.5000]])
    """

-  ``y_hat.argmax(dim=1)``\ 返回矩阵\ ``y_hat``\ 每行中最大元素的索引。

.. code:: python

    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    print(y_hat.shape)  # torch.Size([2, 3])
    print(y_hat.argmax(dim=1).shape)  # torch.Size([2])
    print(y_hat.argmax(dim=1))  # tensor([2, 2])

形状改变操作
######################

-  用\ ``view()``\ 来改变\ ``Tensor``\ 的形状。注意\ ``view()``\ 返回的新\ ``Tensor``\ 与源\ ``Tensor``\ 虽然可能有不同的\ ``size``\ ，但是是共享\ ``data``\ 的，也即更改其中的一个，另外一个也会跟着改变。(顾名思义，view仅仅是改变了对这个张量的观察角度，内部数据并未改变)

.. code:: python

    x = torch.zeros(10)
    print(x.shape)  # torch.Size([10])
    y = x.view(2, 5)
    print(y.shape)  # torch.Size([2, 5])
    y = x.view(2, -1)  # -1所指的维度可以根据其他维度的值推出来
    print(y.shape)  # torch.Size([2, 5])
    y = x.view(-1, 5)
    print(y.shape)  # torch.Size([2, 5])
    print('{} 共享数据 {}'.format('-'*15, '-'*15))
    y += 3
    print(x)  # tensor([3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])

-  使用\ ``clone``\ 拷贝tensor，创建一个副本 ，使其不共享\ ``data``\ 。使用\ ``clone``\ 还有一个好处是会被记录在计算图中，即梯度回传到副本时也会传到源\ ``Tensor``\ 。\ *Pytorch还提供了一个reshape() 可以改变形状，但是此函数并不能保证返回的是其拷贝，所以不推荐使用。*

.. code:: python

    x = torch.zeros(10)
    y = x.clone().view(-1, 5)
    y += 3
    print(y)  # tensor([[3., 3., 3., 3., 3.], [3., 3., 3., 3., 3.]])
    print(x)  # tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

-  ``unsqueeze``\ **增加维度**\ ，参数\ ``dim``\ 表示在哪个维度位置增加一个维度。

.. code:: python

    # torch.Size([2]) dim=0 ==》torch.Size([1, 2])
    # torch.Size([2]) dim=1 ==》torch.Size([2, 1])
    x = torch.tensor([1, 2])
    print(x.shape)  # torch.Size([2])
    print(torch.unsqueeze(x, dim=0).shape)  # torch.Size([1, 2])
    print(torch.unsqueeze(x, dim=0))  # tensor([[1, 2]])
    print(torch.unsqueeze(x, dim=1).shape)  # torch.Size([2, 1])
    print(torch.unsqueeze(x, dim=1))
    """
    tensor([[1],
        [2]])
    """

-  ``squeeze``\ **减少维度**\ ，默认去掉所有size为1的维度，可以使用\ ``dim``\ 参数指定某一个待移除的位置。若指定位置size不为1，则不进行任何操作。

.. code:: python

    x = torch.ones(1, 2, 1, 3, 1, 4)
    print(x.shape)  # torch.Size([1, 2, 1, 3, 1, 4])
    print(torch.squeeze(x).shape)  # torch.Size([2, 3, 4])
    print(torch.squeeze(x, dim=0).shape)  # torch.Size([2, 1, 3, 1, 4])
    print(torch.squeeze(x, dim=1).shape)  # torch.Size([1, 2, 1, 3, 1, 4])

-  ``cat`` 张量连接（拼接）。除了参数中指定的维度 ``dim=0`` ，其他位置的形状必须相同。指定dim的size进行加和。

.. code:: python

    x = torch.ones(2, 3, 4)
    y = torch.ones(2, 1, 4)
    z = torch.cat((x, y), dim=1)
    print(z.shape)  # torch.Size([2, 4, 4])

- ``stack`` 沿新维度连接张量序列。所有的张量必须是相同的大小。

.. code:: python

    x = torch.ones(2, 3)
    y = torch.zeros(2, 3)
    z = torch.stack((x, y), dim=1)
    print(z.shape)  # torch.Size([2, 2, 3])
    print(z)
    """
    tensor([[[1., 1., 1.],
     [0., 0., 0.]],
    [[1., 1., 1.],
     [0., 0., 0.]]])
    """
    x = torch.ones(2, 3)
    y = torch.zeros(2, 3)
    z = torch.stack((x, y), dim=0)
    print(z.shape)  # torch.Size([2, 2, 3])
    print(z)
    """
    tensor([[[1., 1., 1.],
     [1., 1., 1.]],
    [[0., 0., 0.],
     [0., 0., 0.]]])
    """

维度交换
######################

- ``permute`` 维度交换。

.. code:: python

    img = torch.ones(3, 32, 32)
    print(img.shape)  # torch.Size([3, 32, 32])
    img = img.permute(1, 2, 0)
    print(img.shape)  # torch.Size([32, 32, 3])

    >>> x = torch.randn(2, 3, 5)
    >>> x.size()
    torch.Size([2, 3, 5])
    >>> x.permute(2, 0, 1).size()
    torch.Size([5, 2, 3])

Tensor与Python数据转换
######################

-  ``item()`` ,它可以将一个标量\ ``Tensor``\ 转换成一个\ ``Python number``\ ：

.. code:: python

    x = torch.tensor([3])
    print(x)  # tensor([3])
    print(x.shape)  # torch.Size([1])
    print(x.item())  # 3

-  **Tensor转numpy**\ ：使用\ ``numpy()``\ 将\ ``Tensor``\ 转换成NumPy数组，二者\ **共享内存**\ ，转换速度很快。改变其中一个另一个也变。所有在CPU上的\ ``Tensor``\ （除了\ ``CharTensor``\ ）都支持与NumPy数组相互转换。

.. code:: python

    x = torch.zeros(3)
    y = x.numpy()
    print(x, '\t', y)  # tensor([0., 0., 0.]) [0. 0. 0.]
    x += 1
    print(x, '\t', y)  # tensor([1., 1., 1.]) [1. 1. 1.]
    y += 1
    print(x, '\t', y)  # tensor([2., 2., 2.]) [2. 2. 2.]

-  **numpy转Tensor**\ ：使用\ ``from_numpy()``\ 将NumPy数组转换成\ ``Tensor``\ ，二者\ **共享内存**\ ，转换速度很快。改变其中一个另一个也变。\ ``torch.tensor()``\ 会进行数据拷贝（就会消耗更多的时间和空间），所以返回的\ ``Tensor``\ 和原来的数据不再共享内存。

.. code:: python

    x = np.zeros(3)
    y = torch.from_numpy(x)
    print(x, '\t', y)  # [0. 0. 0.] tensor([0., 0., 0.], dtype=torch.float64)
    x += 1
    print(x, '\t', y)  # [1. 1. 1.] tensor([1., 1., 1.], dtype=torch.float64)
    y += 2
    print(x, '\t', y)  # [3. 3. 3.] tensor([3., 3., 3.], dtype=torch.float64)
    print('{} 不共享内存 {}'.format('-'*15, '-'*15))
    x = np.zeros(3)
    y = torch.tensor(x)
    print(x, '\t', y)  # [0. 0. 0.] tensor([0., 0., 0.], dtype=torch.float64)
    x += 1
    print(x, '\t', y)  # [1. 1. 1.] tensor([0., 0., 0.], dtype=torch.float64)
    y += 2
    print(x, '\t', y)  # [1. 1. 1.] tensor([2., 2., 2.], dtype=torch.float64)


设备间移动
######################

-  用方法\ ``to()``\ 可以将\ ``Tensor``\ 在CPU和GPU（需要硬件支持）之间相互移动。
-  GPU环境下操作如下，\ ``torch.cuda.is_available()``\ **用于判断cuda是否可用**\ ：

.. code:: python

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)  # cuda
    x = torch.ones(3)
    print(x)  # tensor([1., 1., 1.])
    print(x.to(device))  # tensor([1., 1., 1.], device='cuda:0')
    print(x)  # tensor([1., 1., 1.])
    print(x.to(device, dtype=torch.int))  # tensor([1, 1, 1], device='cuda:0', dtype=torch.int32)

-  CPU环境下操作如下：

.. code:: python

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)  # cpu
    x = torch.ones(3)
    print(x)  # tensor([1., 1., 1.])
    print(x.to(device))  # tensor([1., 1., 1.])
    print(x.to(device, dtype=torch.int))  # tensor([1, 1, 1], dtype=torch.int32)

-  **Tensor运算需要保证都在相同的设备上**\ ，否则会报错：\ ``RuntimeError: expected device cuda:0 but got device cpu``
-  Tensor转Python数据类型的操作（如\ ``.numpy()``\ ），若Tensor在cuda设备上，需要先将其转移至cpu上，再进行操作。否则会报错：\ ``TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.``
