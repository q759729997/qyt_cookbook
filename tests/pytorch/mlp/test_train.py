"""
    main_module - 模型训练测试，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest

import numpy as np
import torch
from torch import nn


class TestTrain(unittest.TestCase):
    """模型训练测试.

    Main methods:
        test_train - 模型训练测试.
    """
    # @unittest.skip('debug')
    def test_train(self):
        """模型训练测试.
        """
        print('{} test_train {}'.format('-'*15, '-'*15))
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


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
