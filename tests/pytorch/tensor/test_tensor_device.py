"""
    main_module - Tensor设备间移动测试，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest

import torch


class TestTensorDevice(unittest.TestCase):
    """Tensor设备间移动测试.

    Main methods:
        test_device - Tensor设备移动测试.
    """
    # @unittest.skip('debug')
    def test_device(self):
        """Tensor设备移动测试.
        """
        print('{} test_device {}'.format('-'*15, '-'*15))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)  # cpu
        x = torch.ones(3)
        print(x)  # tensor([1., 1., 1.])
        print(x.to(device))  # tensor([1., 1., 1.])
        print(x.to(device, dtype=torch.int))  # tensor([1, 1, 1], dtype=torch.int32)


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
