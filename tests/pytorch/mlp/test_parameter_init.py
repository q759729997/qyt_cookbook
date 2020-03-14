"""
    main_module - 模型参数初始化测试，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest

from torch import nn


class TestParameterInit(unittest.TestCase):
    """模型参数初始化测试.

    Main methods:
        test_normal - 正态分布初始化测试.
    """
    # @unittest.skip('debug')
    def test_normal(self):
        """正态分布初始化测试.
        """
        print('{} test_normal {}'.format('-'*15, '-'*15))
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


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
