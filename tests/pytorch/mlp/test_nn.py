"""
    main_module - pytorch定义模型测试，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest

from torch import nn


class TestNN(unittest.TestCase):
    """pytorch定义模型测试.

    Main methods:
        test_module_example - module继承示例.
        test_sequential - sequential搭建网络方法.
    """
    @unittest.skip('debug')
    def test_module_example(self):
        """module继承示例.
        """
        print('{} test_module_example {}'.format('-'*15, '-'*15))

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

    # @unittest.skip('debug')
    def test_sequential(self):
        """sequential搭建网络方法.
        """
        print('{} test_sequential {}'.format('-'*15, '-'*15))
        num_inputs = 5
        # 写法一
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
        # 写法二
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


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
