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
        test_unsqueeze - unsqueeze增加维度函数测试.
        test_squeeze - squeeze减少维度函数测试.
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

    @unittest.skip('debug')
    def test_clone(self):
        """clone函数测试.
        """
        print('{} test_clone {}'.format('-'*15, '-'*15))
        x = torch.zeros(10)
        y = x.clone().view(-1, 5)
        y += 3
        print(y)  # tensor([[3., 3., 3., 3., 3.], [3., 3., 3., 3., 3.]])
        print(x)  # tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    @unittest.skip('debug')
    def test_unsqueeze(self):
        """unsqueeze增加维度函数测试.
        """
        print('{} test_unsqueeze {}'.format('-'*15, '-'*15))
        x = torch.tensor([1, 2])
        print(x.shape)  # torch.Size([2])
        print(torch.unsqueeze(x, dim=0).shape)  # torch.Size([1, 2])
        print(torch.unsqueeze(x, dim=0))  # tensor([[1, 2]])
        print(torch.unsqueeze(x, dim=1).shape)  # torch.Size([2, 1])
        print(torch.unsqueeze(x, dim=1))
        """
        tensor([[1],
        [2]])
        """

    # @unittest.skip('debug')
    def test_squeeze(self):
        """squeeze减少维度函数测试.
        """
        print('{} test_squeeze {}'.format('-'*15, '-'*15))
        x = torch.ones(1, 2, 1, 3, 1, 4)
        print(x.shape)  # torch.Size([1, 2, 1, 3, 1, 4])
        print(torch.squeeze(x).shape)  # torch.Size([2, 3, 4])
        print(torch.squeeze(x, dim=0).shape)  # torch.Size([2, 1, 3, 1, 4])
        print(torch.squeeze(x, dim=1).shape)  # torch.Size([1, 2, 1, 3, 1, 4])


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
