"""
    main_module - Tensor常用属性测试，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest

import torch


class TestTensorAttr(unittest.TestCase):
    """Tensor常用属性测试.

    Main methods:
        test_shape - Tensor形状获取测试.
    """
    # @unittest.skip('debug')
    def test_shape(self):
        """Tensor形状获取测试.
        """
        print('{} test_shape {}'.format('-'*15, '-'*15))
        x = torch.randn(8, 28, 28)
        print(x.shape)  # 输出 torch.Size([8, 28, 28])
        print(type(x.shape))  # 输出 <class 'torch.Size'>
        print(x.size())  # 输出 torch.Size([8, 28, 28])
        print(type(x.size()))  # 输出 <class 'torch.Size'>
        print(x.size()[1])  # 输出 28
        # x.shape[1] += 1，TypeError: 'torch.Size' object does not support item assignment


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
