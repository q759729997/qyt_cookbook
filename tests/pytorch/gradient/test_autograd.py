"""
    main_module - 梯度自动计算测试，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest

import torch


class TestAutoGrad(unittest.TestCase):
    """梯度自动计算测试.

    Main methods:
        test_requires_grad - requires_grad属性测试.
        test_grad_fn - grad_fn属性测试.
        test_backward - backward函数测试.
    """
    @unittest.skip('debug')
    def test_requires_grad(self):
        """requires_grad属性测试.
        """
        print('{} test_requires_grad {}'.format('-'*15, '-'*15))
        x = torch.ones(2)  # # 缺失情况下默认 requires_grad = False
        print(x, x.requires_grad)  # tensor([1., 1.]) False
        x.requires_grad_(True)  # 通过.requires_grad_()用in-place的方式改变requires_grad属性
        print(x, x.requires_grad)  # tensor([1., 1.], requires_grad=True) True
        x = torch.ones(2, requires_grad=True)
        print(x, x.requires_grad)  # tensor([1., 1.], requires_grad=True) True

    @unittest.skip('debug')
    def test_grad_fn(self):
        """grad_fn属性测试.
        """
        print('{} test_grad_fn {}'.format('-'*15, '-'*15))
        x = torch.ones(2)
        print(x.grad_fn)  # None
        y = x * 2
        print(y)  # tensor([2., 2.])
        print(x.grad_fn, y.grad_fn)  # None None
        print('{} 设置requires_grad=True {}'.format('-'*15, '-'*15))
        x = torch.ones(2, requires_grad=True)
        print(x.grad_fn)  # None 直接创建的Tensor没有grad_fn，被称为叶子节点。
        y = x * 2
        print(x.is_leaf, y.is_leaf)  # True False
        print(y)  # tensor([2., 2.], grad_fn=<MulBackward0>)
        print(x.grad_fn, y.grad_fn)  # None <MulBackward0 object at 0x00000282A67BCF28>

    # @unittest.skip('debug')
    def test_backward(self):
        """backward函数测试.
        """
        print('{} test_backward {}'.format('-'*15, '-'*15))
        x = torch.ones(2, requires_grad=True)
        y = x * x
        print(y, y.grad_fn)  # tensor([1., 1.], grad_fn=<MulBackward0>) <MulBackward0 object at 0x000001521DC5BF60>
        y = y.sum()
        print(y, y.grad_fn)  # tensor(2., grad_fn=<SumBackward0>) <SumBackward0 object at 0x000001521DC5BF60>
        print(x.grad)  # None
        print(y.backward())  # None 等价于 y.backward(torch.tensor(1.))
        print(x.grad)  # tensor([2., 2.])


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
