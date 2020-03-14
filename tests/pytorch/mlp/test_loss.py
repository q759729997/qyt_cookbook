"""
    main_module - 损失函数测试，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest

import torch
from torch import nn


class TestLoss(unittest.TestCase):
    """损失函数测试.

    Main methods:
        test_MSELoss - 均方误差测试.
    """
    # @unittest.skip('debug')
    def test_MSELoss(self):
        """均方误差测试.
        """
        print('{} test_MSELoss {}'.format('-'*15, '-'*15))
        loss = nn.MSELoss()  # 均方误差损失
        print(loss)  # MSELoss()
        pred_y = torch.tensor([-1, -1], dtype=torch.float)
        y = torch.tensor([1, 1], dtype=torch.float)
        print(loss(pred_y, y))  # tensor(4.)，数据类型不能为int


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
