"""
    main_module - Tensor数学计算测试，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest

import torch


class TestTensorCalculate(unittest.TestCase):
    """Tensor数学计算测试.

    Main methods:
        test_add - Tensor相加测试.
        test_broadcasting - 广播机制测试.
        test_keepdim - 按维度求和.
        test_max - max函数.
    """
    @unittest.skip('debug')
    def test_add(self):
        """Tensor相加测试.
        """
        print('{} test_add {}'.format('-'*15, '-'*15))
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([0.1, 0.2, 0.3])
        print('{} 形式一 {}'.format('-'*15, '-'*15))
        # 形式一：直接相加
        print(x + y)  # tensor([1.1000, 2.2000, 3.3000])
        print('{} 形式二 {}'.format('-'*15, '-'*15))
        # 形式二：使用torch.add函数相加
        print(torch.add(x, y))  # tensor([1.1000, 2.2000, 3.3000])
        print('{} 形式三 {}'.format('-'*15, '-'*15))
        # 形式三：使用torch.add函数相加，将结果输出至指定的tensor
        result = torch.empty_like(y)
        print(torch.add(x, y, out=result))  # tensor([1.1000, 2.2000, 3.3000])
        print(result)  # tensor([1.1000, 2.2000, 3.3000])
        print('{} 形式四 {}'.format('-'*15, '-'*15))
        # 形式四：inplace版本，y的值会变化
        print(y.add_(x))  # tensor([1.1000, 2.2000, 3.3000])
        print(y)  # tensor([1.1000, 2.2000, 3.3000])

    @unittest.skip('debug')
    def test_broadcasting(self):
        """广播机制测试.
        """
        print('{} test_broadcasting {}'.format('-'*15, '-'*15))
        x = torch.eye(2, 2)
        print(x)  # tensor([[1., 0.], [0., 1.]])
        print(x + torch.ones(1))  # tensor([[2., 1.], [1., 2.]])
        print(x + torch.ones(1, 2))  # tensor([[2., 1.], [1., 2.]])
        print(x + torch.ones(2, 1))  # tensor([[2., 1.], [1., 2.]])
        print(x + torch.ones(2, 2))  # tensor([[2., 1.], [1., 2.]])

    @unittest.skip('debug')
    def test_keepdim(self):
        """按维度求和.
        """
        print('{} test_keepdim {}'.format('-'*15, '-'*15))
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        print(x.shape)  # torch.Size([2, 3])
        print(x.sum(dim=0, keepdim=True))  # tensor([[5, 7, 9]])
        print(x.sum(dim=0, keepdim=True).shape)  # torch.Size([1, 3])
        print(x.sum(dim=1, keepdim=True))  # tensor([[ 6], [15]])
        print(x.sum(dim=1, keepdim=True).shape)  # torch.Size([2, 1])

    # @unittest.skip('debug')
    def test_max(self):
        """max函数.
        """
        print('{} test_max {}'.format('-'*15, '-'*15))
        x = torch.tensor([-1, 0, 1])
        print(x)
        print(torch.max(x))  # tensor(1)
        print(torch.max(x, torch.tensor(0)))  # tensor([0, 0, 1])


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
