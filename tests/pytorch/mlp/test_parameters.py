"""
    main_module - 模型参数测试，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest

from torch import nn


class TestParameters(unittest.TestCase):
    """模型参数测试.

    Main methods:
        test_parameters - 模型参数测试.
    """
    # @unittest.skip('debug')
    def test_parameters(self):
        """模型参数测试.
        """
        print('{} test_parameters {}'.format('-'*15, '-'*15))
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
        for name, param in net.named_parameters():
            print('name:{}, param:{}'.format(name, param))
        """
        name:linear.weight, param:Parameter containing:
        tensor([[-0.3299, -0.2503,  0.1922, -0.3915, -0.2623]], requires_grad=True)
        name:linear.bias, param:Parameter containing:
        tensor([-0.4374], requires_grad=True)
        """


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
