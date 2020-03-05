"""
    main_module - 数据转换测试，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest

import numpy as np
import torch


class TestDataTransform(unittest.TestCase):
    """数据转换测试.

    Main methods:
        test_item - item函数测试.
        test_tensor_to_numpy - Tensor转NumPy.
        test_numpy_to_tensor - NumPy转Tensor.
    """
    @unittest.skip('debug')
    def test_item(self):
        """item函数测试.
        """
        print('{} test_item {}'.format('-'*15, '-'*15))
        x = torch.tensor([3])
        print(x)  # tensor([3])
        print(x.shape)  # torch.Size([1])
        print(x.item())  # 3

    @unittest.skip('debug')
    def test_tensor_to_numpy(self):
        """Tensor转NumPy.
        """
        print('{} test_tensor_to_numpy {}'.format('-'*15, '-'*15))
        x = torch.zeros(3)
        y = x.numpy()
        print(x, '\t', y)  # tensor([0., 0., 0.])     [0. 0. 0.]
        x += 1
        print(x, '\t', y)  # tensor([1., 1., 1.])     [1. 1. 1.]
        y += 1
        print(x, '\t', y)  # tensor([2., 2., 2.])     [2. 2. 2.]

    # @unittest.skip('debug')
    def test_numpy_to_tensor(self):
        """NumPy转Tensor.
        """
        print('{} test_numpy_to_tensor {}'.format('-'*15, '-'*15))
        x = np.zeros(3)
        y = torch.from_numpy(x)
        print(x, '\t', y)  # [0. 0. 0.]       tensor([0., 0., 0.], dtype=torch.float64)
        x += 1
        print(x, '\t', y)  # [1. 1. 1.]       tensor([1., 1., 1.], dtype=torch.float64)
        y += 2
        print(x, '\t', y)  # [3. 3. 3.]       tensor([3., 3., 3.], dtype=torch.float64)
        print('{} 不共享内存 {}'.format('-'*15, '-'*15))
        x = np.zeros(3)
        y = torch.tensor(x)
        print(x, '\t', y)  # [0. 0. 0.]       tensor([0., 0., 0.], dtype=torch.float64)
        x += 1
        print(x, '\t', y)  # [1. 1. 1.]       tensor([0., 0., 0.], dtype=torch.float64)
        y += 2
        print(x, '\t', y)  # [1. 1. 1.]       tensor([2., 2., 2.], dtype=torch.float64)


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
