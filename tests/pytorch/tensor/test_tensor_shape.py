"""
    main_module - Tensor形状改变操作测试，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest

import torch


class TestTensorShape(unittest.TestCase):
    """Tensor形状改变操作测试.

    Main methods:
        test_view - view函数测试.
        test_clone - clone函数测试.
    """
    @unittest.skip('debug')
    def test_view(self):
        """view函数测试.
        """
        print('{} test_view {}'.format('-'*15, '-'*15))
        x = torch.zeros(10)
        print(x.shape)  # torch.Size([10])
        y = x.view(2, 5)
        print(y.shape)  # torch.Size([2, 5])
        y = x.view(2, -1)  # -1所指的维度可以根据其他维度的值推出来
        print(y.shape)  # torch.Size([2, 5])
        y = x.view(-1, 5)
        print(y.shape)  # torch.Size([2, 5])
        print('{} 共享数据 {}'.format('-'*15, '-'*15))
        y += 3
        print(x)  # tensor([3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])

    # @unittest.skip('debug')
    def test_clone(self):
        """clone函数测试.
        """
        print('{} test_clone {}'.format('-'*15, '-'*15))
        x = torch.zeros(10)
        y = x.clone().view(-1, 5)
        y += 3
        print(y)  # tensor([[3., 3., 3., 3., 3.], [3., 3., 3., 3., 3.]])
        print(x)  # tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
