"""
    main_module - FashionMNIST数据读取与处理测试，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest

import torchvision


class TestFashionMNIST(unittest.TestCase):
    """FashionMNIST数据读取与处理测试.

    Main methods:
        test_FashionMNIST - FashionMNIST数据集加载.
    """
    # @unittest.skip('debug')
    def test_FashionMNIST(self):
        """FashionMNIST数据集加载.
        """
        print('{} test_FashionMNIST {}'.format('-'*15, '-'*15))
        mnist_train = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST', train=True, download=True, transform=torchvision.transforms.ToTensor())
        mnist_test = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST', train=False, download=True, transform=torchvision.transforms.ToTensor())
        print(type(mnist_train))  # <class 'torchvision.datasets.mnist.FashionMNIST'>
        print(len(mnist_train), len(mnist_test))  # 60000 10000


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
