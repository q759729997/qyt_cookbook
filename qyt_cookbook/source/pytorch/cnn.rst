==================
卷积神经网络
==================

- 卷积神经网络（convolutional neural network）是含有卷积层（convolutional layer）的神经网络。

二维卷积
######################

- 虽然卷积层得名于卷积（convolution）运算，但我们通常在卷积层中使用更加直观的互相关（cross-correlation）运算。在二维卷积层中，一个二维输入数组和一个二维核（kernel）数组（又称卷积核或过滤器（filter））通过互相关运算输出一个二维数组。图中的阴影部分为第一个输出元素及其计算所使用的输入和核数组元素： :math:`0\times0+1\times1+3\times2+4\times3=19` 。

.. image:: ./cnn.assets/cnn_example_20200321205616.png
    :alt:
    :align: center

- 二维卷积层将输入和卷积核做互相关运算，并加上一个标量偏差来得到输出。 **卷积层的模型参数包括了卷积核和标量偏差** 。在训练模型的时候，通常我们先对卷积核随机初始化，然后不断迭代卷积核和偏差。

- **填充（padding）** 是指在输入高和宽的两侧填充元素（通常是0元素）。下图里我们在原输入高和宽的两侧分别添加了值为0的元素，使得输入高和宽从3变成了5。填充可以增加输出的高和宽。这常用来使输出与输入具有相同的高和宽。

.. image:: ./cnn.assets/padding_20200321211422.png
    :alt:
    :align: center

- **步幅（stride）** 即为每次滑动的行数和列数。下图展示了在高上步幅为3、在宽上步幅为2的二维互相关运算。步幅可以减小输出的高和宽。

.. image:: ./cnn.assets/stride_20200321211847.png
    :alt:
    :align: center

输出形状计算
***************************

- 假设输入形状是 :math:`n_h\times n_w` ，卷积核窗口形状是 :math:`k_h\times k_w` ；在高的两侧一共填充 :math:`p_h` 行，在宽的两侧一共填充 :math:`p_w` 列；当高上步幅为 :math:`s_h` ，宽上步幅为 :math:`s_w` 时，输出形状为（除不尽时向下取整）：

.. math::

	\lfloor(n_h-k_h+p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+p_w+s_w)/s_w\rfloor
