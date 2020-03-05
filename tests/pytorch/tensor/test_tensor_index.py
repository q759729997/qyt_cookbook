"""
    main_module - Tensor索引操作测试，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest

import torch


class TestTensorIndex(unittest.TestCase):
    """Tensor索引操作测试.

    Main methods:
        test_index - Tensor索引操作测试.
    """
    # @unittest.skip('debug')
    def test_index(self):
        """Tensor索引操作测试.
        """
        print('{} test_index {}'.format('-'*15, '-'*15))
        x = torch.eye(2, 2)
        print(x)  # tensor([[1., 0.], [0., 1.]])
        y = x[0, :]  # 取第一维位置0，第二维全部
        print(y)  # tensor([1., 0.])
        y += 3  # 源tensor，也就是x也随之改变
        print(y)  # tensor([4., 3.])
        print(x)  # tensor([[4., 3.], [0., 1.]])


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
