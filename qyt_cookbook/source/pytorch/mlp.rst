==================
多层感知器
==================

-  多层感知器（multilayer perceptron，MLP）：分为输入层、隐藏层、输出层。隐藏层中的神经元和输入层中各个输入完全连接，输出层中的神经元和隐藏层中的各个神经元也完全连接。因此，多层感知机中的隐藏层和输出层都是全连接层（fully-connected layer，也叫稠密层dense layer）。

.. image:: ./mlp.assets/mlp.png
    :alt: 多层感知器
    :align: center

-  多层感知机就是含有至少一个隐藏层的由全连接层组成的神经网络，且每个隐藏层的输出通过激活函数进行变换。多层感知机的层数和各隐藏层中隐藏单元个数都是超参数。多层感知机按以下方式计算输出：

.. math::

    \begin{aligned}
    \boldsymbol{H} &= \phi(\boldsymbol{X} \boldsymbol{W}_h + \boldsymbol{b}_h),\\
    \boldsymbol{O} &= \boldsymbol{H} \boldsymbol{W}_o + \boldsymbol{b}_o,
    \end{aligned}

-  其中\ :math:`\phi`\ 表示激活函数。在分类问题中，我们可以对输出\ :math:`\boldsymbol{O}`\ 做softmax运算，并使用softmax回归中的交叉熵损失函数。在回归问题中，我们将输出层的输出个数设为1，并将输出\ :math:`\boldsymbol{O}`\ 直接提供给线性回归中使用的平方损失函数。

激活函数
######################

-  激活函数（activation function）：全连接层只是对数据做仿射变换（affine transformation），而多个仿射变换的叠加仍然是一个仿射变换。解决问题的一个方法是引入非线性变换，例如对隐藏变量使用按元素运算的非线性函数进行变换，然后再作为下一个全连接层的输入。这个\ **非线性函数被称为激活函数**\ 。

ReLU
***************************

-  ReLU（rectified linear unit）函数提供了一个很简单的非线性变换。给定元素\ :math:`x`\ ，该函数定义为

.. math:: \text{ReLU}(x) = \max(x, 0)

.. image:: ./mlp.assets/image-20200316225821002.png
    :alt:
    :align: center
    :scale: 67

-  当输入为负数时，ReLU函数的导数为0；当输入为正数时，ReLU函数的导数为1。尽管输入为0时ReLU函数不可导，但是我们可以取此处的导数为0。

.. image:: ./mlp.assets/image-20200316230159193.png
    :alt:
    :align: center
    :scale: 67

sigmoid
***************************

-  sigmoid函数可以将元素的值变换到0和1之间：

.. math:: \text{sigmoid}(x) = \frac{1}{1 + \exp(-x)}

.. image:: ./mlp.assets/image-20200316231711688.png
    :alt:
    :align: center
    :scale: 67

-  依据链式法则，sigmoid函数的导数

.. math:: \text{sigmoid}'(x) = \text{sigmoid}(x)\left(1-\text{sigmoid}(x)\right)

-  当输入为0时，sigmoid函数的导数达到最大值0.25；当输入越偏离0时，sigmoid函数的导数越接近0。

.. image:: ./mlp.assets/image-20200316231929192.png
    :alt:
    :align: center
    :scale: 67

tanh
***************************

-  tanh（双曲正切）函数可以将元素的值变换到-1和1之间：

   .. math:: \text{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}

.. image:: ./mlp.assets/image-20200316232255834.png
    :alt:
    :align: center
    :scale: 67

-  当输入为0时，tanh函数的导数达到最大值1；当输入越偏离0时，tanh函数的导数越接近0。

.. image:: ./mlp.assets/image-20200316232411001.png
    :alt:
    :align: center
    :scale: 67

损失函数
######################

-  在机器学习里，将衡量误差的函数称为损失函数（loss function）。例如常见的平方误差函数也称为平方损失（square loss），\ *除以2是为了更方便的求导*\ ：

.. math:: \ell^{(i)}(w, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2

- 在一个深度学习问题中，我们通常会预先定义一个损失函数。有了损失函数以后，我们就可以使用优化算法试图将其最小化。在优化中，这样的损失函数通常被称作优化问题的目标函数（objective function）。依据惯例，优化算法通常只考虑最小化目标函数。其实，任何最大化问题都可以很容易地转化为最小化问题，只需令目标函数的相反数为新的目标函数即可。
-  PyTorch在\ ``nn``\ 模块中提供了各种损失函数，这些损失函数可看作是一种特殊的层，PyTorch也将这些损失函数实现为\ ``nn.Module``\ 的子类。

.. code:: python

    loss = nn.MSELoss()  # 均方误差损失
    print(loss)  # MSELoss()
    pred_y = torch.tensor([-1, -1], dtype=torch.float)
    y = torch.tensor([1, 1], dtype=torch.float)
    print(loss(pred_y, y))  # tensor(4.),数据类型不能为int

局部最小值
***************************

- 局部最小值（local minimum）:对于目标函数 :math:`f(x)` ，如果 :math:`f(x)` 在 :math:`x` 上的值比在 :math:`x` 邻近的其他点的值更小，那么 :math:`f(x)` 可能是一个局部最小值（local minimum）。如果 :math:`f(x)` 在 :math:`x` 上的值是目标函数在整个定义域上的最小值，那么 :math:`f(x)` 是全局最小值（global minimum）。举个例子，给定函数

.. math::

    f(x) = x \cdot \text{cos}(\pi x), \qquad -1.0 \leq x \leq 2.0,

- 我们可以大致找出该函数的局部最小值和全局最小值的位置。需要注意的是，图中箭头所指示的只是大致位置。

.. image:: ./mlp.assets/local_minimum_20200329144322.png
    :alt:
    :align: center
    :scale: 67

鞍点
***************************

- 鞍点（saddle point）:也会导致梯度接近或变成零。由于深度学习模型参数通常都是高维的，目标函数的鞍点通常比局部最小值更常见。
- 下图为函数 :math:`f(x) = x^3` 的示例：

.. image:: ./mlp.assets/saddle_point_20200329145247.png
    :alt:
    :align: center
    :scale: 67

- 下图为二维空间的函数 :math:`f(x, y) = x^2 - y^2` 的示例。该函数看起来像一个马鞍，而鞍点恰好是马鞍上可坐区域的中心。

.. image:: ./mlp.assets/saddle_point_3d_20200329145247.png
    :alt:
    :align: center
    :scale: 80

- 在图的鞍点位置，目标函数在 :math:`x` 轴方向上是局部最小值，但在 :math:`y` 轴方向上是局部最大值。假设一个函数的输入为 :math:`k` 维向量，输出为标量，那么它的海森矩阵（Hessian matrix）有 :math:`k` 个特征值。该函数在梯度为0的位置上可能是局部最小值、局部最大值或者鞍点。

    - 当函数的海森矩阵在梯度为零的位置上的特征值全为正时，该函数得到局部最小值。
    - 当函数的海森矩阵在梯度为零的位置上的特征值全为负时，该函数得到局部最大值。
    - 当函数的海森矩阵在梯度为零的位置上的特征值有正有负时，该函数得到鞍点。

- 随机矩阵理论告诉我们，对于一个大的高斯随机矩阵来说，任一特征值是正或者是负的概率都是0.5。那么，以上第一种情况的概率为  :math:`0.5^k` 。由于深度学习模型参数通常都是高维的（ :math:`k` 很大），目标函数的鞍点通常比局部最小值更常见。
- 参考文献： Wigner, E. P. (1958). On the distribution of the roots of certain symmetric matrices. Annals of Mathematics, 325-327.

海森矩阵
***************************

- 海森矩阵（Hessian matrix）:又译作黑塞矩阵、海瑟矩阵、海塞矩阵等，是一个多元函数的二阶偏导数构成的方阵，描述了函数的局部曲率。

.. image:: ./mlp.assets/hessian_matrix_20200329152816.png
    :alt:
    :align: center
    :scale: 70

优化算法
######################

-  当模型和损失函数形式较为简单时，误差最小化问题的解可以直接用公式表达出来。这类解叫作\ **解析解（analytical solution）**\ 。然而，大多数深度学习模型并没有解析解，只能通过优化算法有限次迭代模型参数来尽可能降低损失函数的值。这类解叫作\ **数值解（numerical solution）**\ 。
-  **小批量随机梯度下降**\ （mini-batch stochastic gradient descent）：在每次迭代中，先随机均匀采样一个由固定数目训练数据样本所组成的小批量（mini-batch）\ :math:`\mathcal{B}`\ ，然后求小批量中数据样本的平均损失有关模型参数的导数（梯度），最后用此结果与预先设定的一个正数的乘积作为模型参数在本次迭代的减小量。
-  ``torch.optim``\ 模块提供了很多常用的优化算法比如SGD、Adam和RMSProp等。

.. code:: python

    net = nn.Sequential()
    net.add_module('linear', nn.Linear(5, 1))
    optimizer = torch.optim.SGD(net.parameters(), lr=0.03)  # 小批量随机梯度下降,lr为必须参数
    print(optimizer)
    """
    SGD (
    Parameter Group 0
   	dampening: 0
   	lr: 0.03
   	momentum: 0
   	nesterov: False
   	weight_decay: 0
    )
    """

-  为不同子网络设置不同的学习率，这\ **在finetune时经常用到**:

.. code:: python

    optimizer = optim.SGD([
                    # 如果对某个参数不指定学习率，就使用最外层的默认学习率
                    {'params': net.subnet1.parameters()}, # lr=0.03
                    {'params': net.subnet2.parameters(), 'lr': 0.01}
                ], lr=0.03)

-  调整学习率：要有两种做法。一种是修改\ ``optimizer.param_groups``\ 中对应的学习率，另一种是更简单也是较为推荐的做法——新建优化器，由于optimizer十分轻量级，构建开销很小，故而可以构建新的optimizer。但是后者对于使用动量的优化器（如Adam），会丢失动量等状态信息，可能会造成损失函数的收敛出现震荡等情况。

.. code:: python

    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1 # 学习率为之前的0.1倍

模型定义
######################

-  ``torch.nn``\ 模块定义了大量神经网络的层。“nn”是neural networks（神经网络）的缩写。它利用\ ``autograd``\ 来定义模型。。\ ``nn``\ 的核心数据结构是\ ``Module``\ ，既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络。\ ``nn.Module``\ 实例应该包含一些层以及返回输出的前向传播（forward）方法。
-   ``nn.Module`` 构造的网络，无须定义反向传播函数。系统将通过自动求梯度而自动生成反向传播所需的 ``backward`` 函数。

.. code:: python

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

-  如下为使用\ ``nn.Sequential``\ **搭建网络的三种方法**\ ，\ ``Sequential``\ 是一个有序的容器，网络层将按照在传入\ ``Sequential``\ 的顺序依次被添加到计算图中。

.. code:: python

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

-  注意：\ ``torch.nn``\ 仅支持输入一个batch的样本不支持单个样本输入，如果\ **只有单个样本**\ ，可使用\ ``input.unsqueeze(0)``\ 来添加一维。
-  Sequential与ModuleList区别：ModuleList仅仅是一个储存各种模块的列表，这些模块之间没有联系也没有顺序（所以不用保证相邻层的输入输出维度匹配），而且没有实现forward功能需要自己实现，所以上面执行net(x)会报NotImplementedError；而Sequential内的模块需要按照顺序排列，要保证相邻层的输入输出大小相匹配，内部forward功能已经实现。
- ``ModuleList`` 不同于一般的Python的list，加入到ModuleList里面的所有模块的 **参数会被自动添加** 到整个网络中。 ``ModuleDict`` 与其功能类似。

模型参数
######################

-  通过\ ``net.parameters()``\ 来查看模型所有的可学习参数，此函数将返回一个生成器（迭代器）。

.. code:: python

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

-  ``net.named_parameters()``\ 可以返回参数名称。

.. code:: python

    for name, param in net.named_parameters():
        print('name:{}, param:{}'.format(name, param))
    """
    name:linear.weight, param:Parameter containing:
    tensor([[-0.3299, -0.2503,  0.1922, -0.3915, -0.2623]], requires_grad=True)
    name:linear.bias, param:Parameter containing:
    tensor([-0.4374], requires_grad=True)
    """

-  param的类型为torch.nn.parameter.Parameter，和Tensor不同的是如果一个Tensor是Parameter，那么它会自动被添加到模型的参数列表里。

初始化模型参数
***************************

- PyTorch中nn.Module的模块参数都采取了较为合理的初始化策略，因此一般不用我们考虑。
-  在使用\ ``net``\ 前，我们需要初始化模型参数。PyTorch在\ ``init``\ 模块中提供了多种参数初始化方法。这里的\ ``init``\ 是\ ``initializer``\ 的缩写形式。
-  通过\ ``init.normal_``\ 将权重参数每个元素初始化为随机采样于均值为0、标准差为0.01的正态分布。偏差会初始化为零。

.. code:: python

    net = nn.Sequential()
    net.add_module('linear', nn.Linear(5, 1))
    print('初始化前')
    for param in net.parameters():
        print(param)
    """输出
    Parameter containing:
    tensor([[-0.0567,  0.1161,  0.1954, -0.2397,  0.3248]], requires_grad=True)
    Parameter containing:
    tensor([-0.0782], requires_grad=True)
    """
    nn.init.normal_(net[0].weight, mean=0, std=0.01)
    nn.init.constant_(net[0].bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)
    print('初始化后')
    for param in net.parameters():
        print(param)
    """
    Parameter containing:
    tensor([[0.0037, 0.0178, 0.0186, 0.0216, 0.0020]], requires_grad=True)
    Parameter containing:
    tensor([0.], requires_grad=True)
    """

-  如果需要使用name定位某一层时，则\ ``net[0].weight``\ 应改为\ ``net.linear.weight``\ ，\ ``bias``\ 亦然。因为\ ``net[0]``\ 这样根据下标访问子模块的写法只有当\ ``net``\ 是个\ ``ModuleList``\ 或者\ ``Sequential``\ 实例时才可以。

-  常用的还有\ ``xavier_normal_``\ 。Xavier随机初始化，假设某全连接层的输入个数为 :math:`a` ，输出个数为 :math:`b` ，Xavier随机初始化将使该层中权重参数的每个元素都随机采样于均匀分布,它的设计主要考虑到，模型参数初始化后，每层输出的方差不该受该层输入个数影响，且每层梯度的方差也不该受该层输出个数影响。

.. math::

    U\left(-\sqrt{\frac{6}{a+b}}, \sqrt{\frac{6}{a+b}}\right)

训练模型
######################

-  构造数据=》加载数据=》定义模型=》定义优化器=》定义损失函数=》进行训练。
-  通过调用\ ``optim``\ 实例的\ ``step``\ 函数来迭代模型参数。训练时注意\ ``optimizer.zero_grad()``\ 梯度清零，防止梯度一直累加。

.. code:: python

    # 构造数据
    num_samples = 200  # 样本个数
    num_inputs = 2  # 特征个数
    features = torch.randn(num_samples, num_inputs)
    print('features shape:{}, dtype:{}'.format(features.shape, features.dtype))  # features shape:torch.Size([200, 2]), dtype:torch.float32
    label_weight = [2.0, 5.0]  # 定义一个线性函数
    label_bias = 6.0
    labels = torch.randn(num_samples)
    labels += label_weight[0] * features[:, 0] + label_weight[1] * features[:, 1] + label_bias
    print('labels shape:{}, dtype:{}'.format(labels.shape, labels.dtype))  # labels shape:torch.Size([200]), dtype:torch.float32
    # 加载数据
    batch_size = 8
    dataset = torch.utils.data.TensorDataset(features, labels)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    print('data_iter len:{}'.format(len(data_iter)))
    # for X, y in data_iter:
    #     print(X, y)
    #     break
    # 定义模型
    net = nn.Sequential()
    net.add_module('linear', nn.Linear(num_inputs, 1))
    print(net)
    """
    Sequential(
    (linear): Linear(in_features=2, out_features=1, bias=True)
    )
    """
    # 定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=0.03)
    # 定义损失函数
    loss = nn.MSELoss()
    # 进行训练
    num_epochs = 8
    for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)  # 模型前向传播
        loss_value = loss(output, y.view(-1, 1))  # 计算loss
        optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
        loss_value.backward()  # 反向传播
        optimizer.step()  # 迭代模型参数
    print('epoch %d, loss: %f' % (epoch, loss_value.item()))
    # 输出训练后的结果
    print(label_weight, net[0].weight.data)  # [2.0, 5.0] tensor([[2.0171, 4.9683]])
    print(label_bias, net[0].bias.data)  # 6.0 tensor([6.0194])
    """
    epoch 1, loss: 5.885800
    epoch 2, loss: 0.424021
    epoch 3, loss: 0.963439
    epoch 4, loss: 1.011478
    epoch 5, loss: 1.178113
    epoch 6, loss: 0.847684
    epoch 7, loss: 0.644298
    epoch 8, loss: 0.848485
    """

模型调优
######################

- 权重衰减（weight decay）等价于 :math:`L_2`  范数正则化（regularization）。正则化通过为模型损失函数添加惩罚项使学出的模型参数值较小，是应对过拟合的常用手段。权重衰减可以通过优化器中的 ``weight_decay`` 超参数来指定。

- 丢弃法（dropout）常常被用来应对过拟合问题。 **丢弃法不改变其输入的期望值。** 被丢弃的隐藏单元相关的权重的梯度均为0。由于在训练中隐藏层神经元的丢弃是随机的，输出层的计算无法过度依赖隐藏层中的任一个，从而在训练模型时起到正则化的作用，并可以用来应对过拟合。在测试模型时，我们为了拿到更加确定性的结果，一般不使用丢弃法。 **丢弃法只在训练模型时使用。** 参考文献：Dropout: a simple way to prevent neural networks from overfitting. JMLR

.. image:: ./mlp.assets/dropout_20200319212355.png
    :alt:
    :align: center

- 在PyTorch中，我们只需要在全连接层后添加Dropout层并指定丢弃概率。在训练模型时，Dropout层将以指定的丢弃概率随机丢弃上一层的输出元素；在测试模型时（即model.eval()后），Dropout层并不发挥作用。  ``nn.Dropout(p=0.2)`` p表示被丢弃的概率。

批量归一化
***************************

- 批量归一化（batch normalization）层，它能让较深的神经网络的训练变得更加容易。批量归一化的提出正是为了应对深度模型训练的挑战。在模型训练时，批量归一化利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。
- **对全连接层做批量归一化** 我们将批量归一化层置于全连接层中的仿射变换和激活函数之间。利用均值和方差进行归一化。参考： https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter05_CNN/5.10_batch-norm
- **对卷积层做批量归一化** 对卷积层来说，批量归一化发生在卷积计算之后、应用激活函数之前。如果卷积计算输出多个通道，我们需要对这些通道的输出分别做批量归一化，且 **每个通道都拥有独立的拉伸和偏移参数，并均为标量** 。设小批量中有 :math:`m` 个样本。在单个通道上，假设卷积计算输出的高和宽分别为 :math:`p` 和 :math:`q` 。我们需要对该通道中 :math:`m \times p \times q`个元素同时做批量归一化。对这些元素做标准化计算时，我们使用相同的均值和方差，即该通道中 :math:`m \times p \times q` 个元素的均值和方差。
- **预测时的批量归一化** 使用批量归一化训练时，我们可以将批量大小设得大一点，从而使批量内样本的均值和方差的计算都较为准确。将训练好的模型用于预测时，我们希望模型对于任意输入都有确定的输出。因此，单个样本的输出不应取决于批量归一化所需要的随机小批量中的均值和方差。一种常用的方法是通过移动平均估算整个训练数据集的样本均值和方差，并在预测时使用它们得到确定的输出。可见，和丢弃层一样，批量归一化层在训练模式和预测模式下的计算结果也是不一样的。
- ``torch.nn.BatchNorm1d、BatchNorm2d`` 实现批量归一化层。重要参数： ``num_features –  通道数(N, C, H, W)``

.. code-block:: python

    # 卷积
    nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
    nn.BatchNorm2d(6),
    # 全连接层
    nn.Linear(120, 84),
    nn.BatchNorm1d(84),

- 参考文献：Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167.

读取和存储
######################

读写模型
***************************

- ``state_dict`` 是一个从参数名称隐射到参数Tesnor的字典对象。

.. code:: python

    net = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    print(net.state_dict())
    """输出
    OrderedDict([('0.weight', tensor([[-0.6398, -0.0105],
    [ 0.2083, -0.5284],
    [-0.1384,  0.0481]])), ('0.bias', tensor([ 0.0495,  0.1969, -0.5676])), ('1.weight', tensor([[-0.5332, -0.3395,  0.2963]])), ('1.bias', tensor([0.3041]))])
    """

- 注意，只有具有可学习参数的层(卷积层、线性层等)才有state_dict中的条目。优化器(optim)也有一个state_dict，其中包含关于优化器状态以及所使用的超参数的信息。

.. code:: python

    optimizer = torch.optim.RMSprop(net.parameters())
    print(optimizer.state_dict())
    """输出
    {'state': {}, 'param_groups': [{'lr': 0.01, 'momentum': 0, 'alpha': 0.99, 'eps': 1e-08, 'centered': False, 'weight_decay': 0, 'params': [2209963174936, 2209963175008, 2209963175224, 2209963175296]}]}
    """

- 保存和加载state_dict(推荐方式):

.. code:: python

    net = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    torch.save(net.state_dict(), './data/save/state_dict.pt')
    net_state_dict = torch.load('./data/save/state_dict.pt')
    print(net_state_dict)
    """
    OrderedDict([('0.weight', tensor([[-0.1572, -0.5445],
    [-0.0474,  0.6642],
    [-0.3742,  0.4575]])), ('0.bias', tensor([ 0.3841, -0.3620,  0.0496])), ('1.weight', tensor([[-0.4403,  0.0146,  0.0514]])), ('1.bias', tensor([0.1762]))])
    """
    net2 = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    print(net2.state_dict())
    """模型2随机初始化的参数，与模型1明显不同
    OrderedDict([('0.weight', tensor([[-0.4988,  0.6664],
    [ 0.4392,  0.1901],
    [ 0.7048,  0.6054]])), ('0.bias', tensor([-0.4389, -0.6592,  0.1810])), ('1.weight', tensor([[-0.3644, -0.1919, -0.2438]])), ('1.bias', tensor([0.2472]))])
    """
    net2.load_state_dict(net_state_dict)
    print(net2.state_dict())
    """使用模型1的参数初始化后，模型2的参数变成与模型1一致
    OrderedDict([('0.weight', tensor([[-0.1572, -0.5445],
    [-0.0474,  0.6642],
    [-0.3742,  0.4575]])), ('0.bias', tensor([ 0.3841, -0.3620,  0.0496])), ('1.weight', tensor([[-0.4403,  0.0146,  0.0514]])), ('1.bias', tensor([0.1762]))])
    """

- 保存和加载整个模型。

.. code:: python

    net = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    print(net.state_dict())
    """
    OrderedDict([('0.weight', tensor([[-0.1631, -0.0345],
    [ 0.3992, -0.1971],
    [-0.2313, -0.2398]])), ('0.bias', tensor([-0.1279,  0.0706,  0.7025])), ('1.weight', tensor([[-0.3476,  0.0543,  0.4400]])), ('1.bias', tensor([-0.3389]))])
    """
    torch.save(net, './data/save/whole_model.pt')
    net2 = torch.load('./data/save/whole_model.pt')
    print(net2)
    """
    Sequential(
    (0): Linear(in_features=2, out_features=3, bias=True)
    (1): Linear(in_features=3, out_features=1, bias=True)
    )
    """
    print(net2.state_dict())
    """与保存前参数一致
    OrderedDict([('0.weight', tensor([[-0.1631, -0.0345],
    [ 0.3992, -0.1971],
    [-0.2313, -0.2398]])), ('0.bias', tensor([-0.1279,  0.0706,  0.7025])), ('1.weight', tensor([[-0.3476,  0.0543,  0.4400]])), ('1.bias', tensor([-0.3389]))])
    """

读写tensor
***************************

-  可以直接使用 ``save`` 函数和 ``load`` 函数分别存储和读取Tensor。save使用Python的pickle实用程序将对象进行序列化，然后将序列化的对象保存到disk，使用save可以保存各种对象,包括模型、张量和字典等。而load使用pickle unpickle工具将pickle的对象文件反序列化为内存。

.. code:: python

    x = torch.ones(1)
    torch.save(x, './data/save/x1.pt')
    x1 = torch.load('./data/save/x1.pt')
    print(x1)  # tensor([1.])
    # 存储列表
    torch.save([x, torch.ones(2)], './data/save/x2.pt')
    x2 = torch.load('./data/save/x2.pt')
    print(x2)  # [tensor([1.]), tensor([1., 1.])]
    # 存储字典
    torch.save({'x': x, 'y': torch.ones(3)}, './data/save/x3.pt')
    x3 = torch.load('./data/save/x3.pt')
    print(x3)  # {'x': tensor([1.]), 'y': tensor([1., 1., 1.])}

GPU计算
######################

- 通过 ``nvidia-smi`` 命令来查看显卡信息。
- 用 ``torch.cuda.is_available()`` 查看GPU是否可用。

.. code:: python

    torch.cuda.is_available() # 查看GPU是否可用，输出 True
    torch.cuda.device_count() # 查看GPU数量，输出 1
    torch.cuda.current_device()  # 查看当前GPU索引号，索引号从0开始
    torch.cuda.get_device_name(0)  # 根据索引号查看GPU名字，输出 'GeForce GTX xxx'

- 使用 ``.cuda()`` 可以将CPU上的Tensor转换（复制）到GPU上。如果有多块GPU，我们用.cuda(i)来表示第 ii 块GPU及相应的显存（ii从0开始）且cuda(0)和cuda()等价。通过Tensor的device属性来查看该Tensor所在的设备。

- 创建tensor的时候就指定设备。

.. code:: python

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor([1, 2, 3], device=device)
    x = torch.tensor([1, 2, 3]).to(device)

- 需要注意的是， **存储在不同位置中的数据是不可以直接进行计算的** 。即存放在CPU上的数据不可以直接与存放在GPU上的数据进行运算，位于不同GPU上的数据也是不能直接进行计算的。
- 同Tensor类似，PyTorch模型也可以通过.cuda转换到GPU上。我们可以通过检查模型的参数的device属性来查看存放模型的设备。
- 前向传播时需要保证模型输入的Tensor和模型都在同一设备上，否则会报错。PyTorch要求计算的所有输入数据都在内存或同一块显卡的显存上。

多GPU计算
***************************

- 要想使用PyTorch进行多GPU计算，最简单的方法是直接用 ``torch.nn.DataParallel`` 将模型wrap一下即可：

.. code-block:: python

    net = torch.nn.Linear(10, 1).cuda()
    net = torch.nn.DataParallel(net)
    print(net)
    """
    DataParallel(
      (module): Linear(in_features=10, out_features=1, bias=True)
    )
    """

- 这时，默认所有存在的GPU都会被使用。如果我们机子中有很多GPU(例如上面显示我们有4张显卡，但是只有第0、3块还剩下一点点显存)，但我们只想使用0、3号显卡，那么我们可以用参数device_ids指定即可: ``torch.nn.DataParallel(net, device_ids=[0, 3])`` 。
- DataParallel也是一个nn.Module，只是这个类其中有一个module就是传入的实际模型。因此当我们调用DataParallel后，模型结构变了（在外面加了一层而已）。所以直接加载肯定会报错的，因为模型结构对不上。所以正确的方法是保存的时候只保存net.module:

.. code-block:: python

    torch.save(net.module.state_dict(), "./DataParallel_model.pt")
    new_net = torch.nn.Linear(10, 1)
    new_net.load_state_dict(torch.load("./DataParallel_model.pt")) # 加载成功
