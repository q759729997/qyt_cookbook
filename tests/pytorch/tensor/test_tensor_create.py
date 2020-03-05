"""
    main_module - 创建Tensor测试，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest

import torch


class TestTensorCreate(unittest.TestCase):
    """创建Tensor测试.

    Main methods:
        test_dytpe - Tensor数据类型测试.
        test_create - Tensor创建测试.
    """
    @unittest.skip('debug')
    def test_dytpe(self):
        """Tensor数据类型测试.
        """
        print('{} test_dytpe {}'.format('-'*15, '-'*15))
        x = torch.ones(1, 2)
        print('x:{}, dtype:{}'.format(x, x.dtype))
        # 输出：x:tensor([[1., 1.]]), dtype:torch.float32

    # @unittest.skip('debug')
    def test_create(self):
        """Tensor创建测试.
        """
        print('{} test_create {}'.format('-'*15, '-'*15))
        x = torch.tensor([1, 2, 3])
        print('x:{}, dtype:{}'.format(x, x.dtype)) 
        # 输出 x:tensor([1, 2, 3]), dtype:torch.int64
        x = x.new_empty(1, 5)
        print('x:{}, dtype:{}'.format(x, x.dtype))  
        # 输出 x:tensor([[0, 0, 0, 0, 0]]), dtype:torch.int64
        x = x.new_empty(1, 5, dtype=torch.double)
        print('x:{}, dtype:{}'.format(x, x.dtype))
        # x:tensor([[0., 0., 0., 0., 0.]], dtype=torch.float64), dtype:torch.float64


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
