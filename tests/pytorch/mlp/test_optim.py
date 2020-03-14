"""
    main_module - 优化算法测试，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest

import torch
from torch import nn


class TestOptim(unittest.TestCase):
    """优化算法测试.

    Main methods:
        test_SGD - 小批量随机梯度下降.
    """
    # @unittest.skip('debug')
    def test_SGD(self):
        """小批量随机梯度下降.
        """
        print('{} test_SGD {}'.format('-'*15, '-'*15))
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


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
