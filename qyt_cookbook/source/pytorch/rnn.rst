==================
循环神经网络
==================

- 循环神经网络（recurrent neural network）。循环神经网络的隐藏状态可以捕捉截至当前时间步的序列的历史信息。循环神经网络模型参数的数量不随时间步的增加而增长。
- 下图展示了循环神经网络在3个相邻时间步的计算逻辑。在时间步 :math:`t` ，隐藏状态的计算可以看成是将输入 :math:`\boldsymbol{X}_t` 和前一时间步隐藏状态 :math:`\boldsymbol{H}_{t-1}` 连结后输入一个激活函数为 :math:`\phi` 的全连接层。该全连接层的输出就是当前时间步的隐藏状态 :math:`\boldsymbol{H}_t` ，且模型参数为 :math:`\boldsymbol{W}_{xh}` 与 :math:`\boldsymbol{W}_{hh}` 的连结，偏差为 :math:`\boldsymbol{b}_h` 。当前时间步 :math:`t` 的隐藏状态 :math:`\boldsymbol{H}_t` 将参与下一个时间步 :math:`t+1` 的隐藏状态 :math:`\boldsymbol{H}_{t+1}` 的计算，并输入到当前时间步的全连接输出层。

.. image:: ./rnn.assets/rnn_20200326232143.png
    :alt:
    :align: center

- 隐藏状态中 :math:`\boldsymbol{X}_t \boldsymbol{W}_{xh} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hh}` 的计算等价于 :math:`\boldsymbol{X}_t` 与 :math:`\boldsymbol{H}_{t-1}` 连结后的矩阵乘以 :math:`\boldsymbol{W}_{xh}` 与 :math:`\boldsymbol{W}_{hh}` 连结后的矩阵。
- 可以基于字符级循环神经网络来创建语言模型。因为每个输入词是一个字符，因此这个模型被称为字符级循环神经网络（character-level recurrent neural network）。

.. image:: ./rnn.assets/character_level_lm__20200326232923.png
    :alt:
    :align: center