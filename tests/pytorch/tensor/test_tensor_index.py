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
        test_gather - 根据index，在dim维度上选取数据.
        test_argmax - 获取最大元素的索引.
    """
    @unittest.skip('debug')
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

    @unittest.skip('debug')
    def test_gather(self):
        """根据index，在dim维度上选取数据.
        """
        print('{} test_gather {}'.format('-'*15, '-'*15))
        y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
        print(y_hat.shape)  # torch.Size([2, 3])
        index = torch.LongTensor([0, 2]).view(-1, 1)
        print(index.shape)  # torch.Size([2, 1])
        print(index)
        """
        tensor([[0],
        [2]])
        """
        print(y_hat.gather(dim=1, index=index).shape)  # torch.Size([2, 1])
        print(y_hat.gather(dim=1, index=index))
        """
        tensor([[0.1000],
        [0.5000]])
        """

    # @unittest.skip('debug')
    def test_argmax(self):
        """获取最大元素的索引.
        """
        print('{} test_argmax {}'.format('-'*15, '-'*15))
        y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
        print(y_hat.shape)  # torch.Size([2, 3])
        print(y_hat.argmax(dim=1).shape)  # torch.Size([2])
        print(y_hat.argmax(dim=1))  # tensor([2, 2])


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
