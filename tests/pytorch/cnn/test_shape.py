"""
    main_module - 输出形状计算，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest

import torch
from torch import nn


class TestShape(unittest.TestCase):
    """输出形状计算.

    Main methods:
        test_Conv2d - 二维卷积核.
    """
    # @unittest.skip('debug')
    def test_Conv2d(self):
        """二维卷积核.
        """
        print('{} test_Conv2d {}'.format('-'*15, '-'*15))
        net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3)
        )
        x = torch.ones(8, 1, 28, 28)  # 批量大小, 通道, 高, 宽
        y = net(x)
        print(y.shape)  # torch.Size([8, 2, 26, 26])


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
