==================
经典卷积模型
==================

LeNet模型
######################

- 它是早期用来识别手写数字图像的卷积神经网络。这个名字来源于LeNet论文的第一作者Yann LeCun。LeNet展示了通过梯度下降训练卷积神经网络可以达到手写数字识别在当时最先进的结果。这个奠基性的工作第一次将卷积神经网络推上舞台，为世人所知。LeNet的网络结构如下图所示。

.. image:: ./cnnModels.assets/lenet_20200322145155.png
    :alt:
    :align: center

- LeNet分为卷积层块和全连接层块两个部分。使用sigmoid激活函数。参数量： ``total: 44426, trainable: 44426`` ,模型结构：

.. code:: python

	class LeNet(nn.Module):
	    """早期用来识别手写数字图像的卷积神经网络"""
	    def __init__(self):
	        super().__init__()
	        self.conv = nn.Sequential(
	            nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
	            nn.Sigmoid(),
	            nn.MaxPool2d(2, 2),  # kernel_size, stride
	            nn.Conv2d(6, 16, 5),
	            nn.Sigmoid(),
	            nn.MaxPool2d(2, 2))
	        # input_shape = (64, 1, 28, 28)  # 批量大小, 通道, 高, 宽
	        self.fc = nn.Sequential(
	            nn.Linear(16 * 4 * 4, 120),
	            nn.Sigmoid(),
	            nn.Linear(120, 84),
	            nn.Sigmoid(),
	            nn.Linear(84, 10))

	    def forward(self, img):
	        feature = self.conv(img)
	        output = self.fc(feature.view(img.shape[0], -1))
	        return output

- 卷积层输出形状计算：

.. code:: python

    """
    input_shape:(64, 1, 28, 28)
    layer:1 conv:(1, 6, 5), output_shape:(64, 6, 24, 24)
    layer:2 maxpool:(2, 2), output_shape:(64, 6, 12, 12)
    layer:3 conv:(6, 16, 5), output_shape:(64, 16, 8, 8)
    layer:4 maxpool:(2, 2), output_shape:(64, 16, 4, 4)
    output_shape:(64, 16, 4, 4)
    """

- 参考文献：LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.