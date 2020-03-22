"""
    main_module - 数据转换，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest

import torchvision


class TestTransforms(unittest.TestCase):
    """数据转换.

    Main methods:
        test_resize - 图片大小改变.
    """
    # @unittest.skip('debug')
    def test_resize(self):
        """图片大小改变.
        """
        print('{} test_resize {}'.format('-'*15, '-'*15))
        mnist_train = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST', train=True, download=True, transform=torchvision.transforms.ToTensor())
        mnist_test = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST', train=False, download=True, transform=torchvision.transforms.ToTensor())
        print(type(mnist_train))  # <class 'torchvision.datasets.mnist.FashionMNIST'>
        print(len(mnist_train), len(mnist_test))  # 60000 10000
        # 查看第一个样本
        feature, label = mnist_train[0]
        print(feature.shape, label)  # Channel x Height x Width
        # 输出：torch.Size([1, 28, 28]) 9
        # 形状改变
        trans = []
        # 原始大小torch.Size([1, 28, 28])
        trans.append(torchvision.transforms.Resize(size=224))
        trans.append(torchvision.transforms.ToTensor())
        transform = torchvision.transforms.Compose(trans)
        mnist_train = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST', train=True, download=True, transform=transform)
        feature, label = mnist_train[0]
        print(feature.shape, label)  # Channel x Height x Width
        # 输出：torch.Size([1, 224, 224]) 9


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
