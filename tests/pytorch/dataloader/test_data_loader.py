"""
    main_module - 数据读取与处理测试，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest

import torch
import torch.utils.data as Data


class TestDataLoader(unittest.TestCase):
    """数据读取与处理测试.

    Main methods:
        test_dataLoader - 随机读取小批量数据.
    """
    # @unittest.skip('debug')
    def test_dataLoader(self):
        """随机读取小批量数据.
        """
        print('{} test_dataLoader {}'.format('-'*15, '-'*15))
        features = torch.zeros(20, 2)  # 第一个维度表示样本数目，切分batch时以此为依据
        print('features:', features.shape)  # features: torch.Size([20, 2])
        labels = torch.ones(features.shape[0])  # labels: torch.Size([20])
        print('labels:', labels.shape)  # data_iter len: 3
        batch_size = 3
        # 将训练数据的特征和标签组合
        dataset = Data.TensorDataset(features, labels)  # tensors that have the same size of the first dimension.
        # 随机读取小批量
        data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
        print('data_iter len:', len(data_iter))  # data_iter len: 7
        for X, y in data_iter:
            print(X, y)
            break
        """输出
        tensor([[0., 0.],
        [0., 0.],
        [0., 0.]]) tensor([1., 1., 1.])
        """


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
