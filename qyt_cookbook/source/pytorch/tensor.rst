.. _header-n479:

Tensor数据操作
==============

-  ``Tensor``\ 这个单词一般可译作“张量”，张量可以看作是一个多维数组。标量可以看作是0维张量，向量可以看作1维张量，矩阵可以看作是二维张量。

.. _header-n484:

创建Tensor
----------

-  创建\ ``Tensor``\ 函数列表：

================================= =========================
函数                              功能
================================= =========================
Tensor(*sizes)                    基础构造函数
tensor(data,)                     类似np.array的构造函数
empty(*sizes)                     未初始化的Tensor
ones(*sizes)                      全1的Tensor
zeros(*sizes)                     全0的Tensor
eye(*sizes)                       对角线为1，其他为0
arange(s,e,step)                  从s到e，步长为step
linspace(s,e,steps)               从s到e，均匀切分成steps份
rand/randn(*sizes)                均匀/标准分布
normal(mean,std)/uniform(from,to) 正态分布/均匀分布
randperm(m)                       随机排列
================================= =========================

-  重要参数 ``dtype``\ ，\ **默认数据类型**\ 为 ``torch.float``
   ，32位浮点数。常用数据类型。

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

-  通过现有的 ``Tensor`` 来创建，此方法会默认重用输入 ``Tensor``
   的一些属性，例如数据类型，除非自定义数据类型。

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

.. _header-n558:

Tensor常用属性
--------------

-  ``Tensor`` 形状获取，通过 ``shape`` 或者 ``size()`` 来获取 ``Tensor``
   的形状。返回的 ``torch.Size`` 其实相当于一个 ``tuple`` , 支持所有
   ``tuple`` 的操作。

.. code:: python

   x = torch.randn(8, 28, 28)
   print(x.shape)  # 输出 torch.Size([8, 28, 28])
   print(type(x.shape))  # 输出 <class 'torch.Size'>
   print(x.size())  # 输出 torch.Size([8, 28, 28])
   print(type(x.size()))  # 输出 <class 'torch.Size'>
   print(x.size()[1])  # 输出 28
   # x.shape[1] += 1，TypeError: 'torch.Size' object does not support item assignment
