"""
    main_module - 读取和存储，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest

import torch
from torch import nn


class TestSaveLoad(unittest.TestCase):
    """读取和存储.

    Main methods:
        test_tensor_save_load - 读取和存储tensor.
        test_state_dict - state_dict函数测试.
        test_state_dict_save_load - state_dict保存与加载测试.
        test_whole_model_save_load - 整个模型保存与加载测试.
    """
    @unittest.skip('debug')
    def test_tensor_save_load(self):
        """读取和存储tensor.
        """
        print('{} test_tensor_save_load {}'.format('-'*15, '-'*15))
        x = torch.ones(1)
        torch.save(x, './data/save/x1.pt')
        x1 = torch.load('./data/save/x1.pt')
        print(x1)  # tensor([1.])
        # 存储列表
        torch.save([x, torch.ones(2)], './data/save/x2.pt')
        x2 = torch.load('./data/save/x2.pt')
        print(x2)  # [tensor([1.]), tensor([1., 1.])]
        # 存储字典
        torch.save({'x': x, 'y': torch.ones(3)}, './data/save/x3.pt')
        x3 = torch.load('./data/save/x3.pt')
        print(x3)  # {'x': tensor([1.]), 'y': tensor([1., 1., 1.])}

    @unittest.skip('debug')
    def test_state_dict(self):
        """state_dict函数测试.
        """
        print('{} test_state_dict {}'.format('-'*15, '-'*15))
        net = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
        print(net.state_dict())
        """输出
        OrderedDict([('0.weight', tensor([[-0.6398, -0.0105],
        [ 0.2083, -0.5284],
        [-0.1384,  0.0481]])), ('0.bias', tensor([ 0.0495,  0.1969, -0.5676])), ('1.weight', tensor([[-0.5332, -0.3395,  0.2963]])), ('1.bias', tensor([0.3041]))])
        """
        optimizer = torch.optim.RMSprop(net.parameters())
        print(optimizer.state_dict())
        """输出
        {'state': {}, 'param_groups': [{'lr': 0.01, 'momentum': 0, 'alpha': 0.99, 'eps': 1e-08, 'centered': False, 'weight_decay': 0, 'params': [2209963174936, 2209963175008, 2209963175224, 2209963175296]}]}
        """

    @unittest.skip('debug')
    def test_state_dict_save_load(self):
        """state_dict保存与加载测试.
        """
        print('{} test_state_dict_save_load {}'.format('-'*15, '-'*15))
        net = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
        torch.save(net.state_dict(), './data/save/state_dict.pt')
        net_state_dict = torch.load('./data/save/state_dict.pt')
        print(net_state_dict)
        """
        OrderedDict([('0.weight', tensor([[-0.1572, -0.5445],
        [-0.0474,  0.6642],
        [-0.3742,  0.4575]])), ('0.bias', tensor([ 0.3841, -0.3620,  0.0496])), ('1.weight', tensor([[-0.4403,  0.0146,  0.0514]])), ('1.bias', tensor([0.1762]))])
        """
        net2 = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
        print(net2.state_dict())
        """模型2随机初始化的参数，与模型1明显不同
        OrderedDict([('0.weight', tensor([[-0.4988,  0.6664],
        [ 0.4392,  0.1901],
        [ 0.7048,  0.6054]])), ('0.bias', tensor([-0.4389, -0.6592,  0.1810])), ('1.weight', tensor([[-0.3644, -0.1919, -0.2438]])), ('1.bias', tensor([0.2472]))])
        """
        net2.load_state_dict(net_state_dict)
        print(net2.state_dict())
        """使用模型1的参数初始化后，模型2的参数变成与模型1一致
        OrderedDict([('0.weight', tensor([[-0.1572, -0.5445],
        [-0.0474,  0.6642],
        [-0.3742,  0.4575]])), ('0.bias', tensor([ 0.3841, -0.3620,  0.0496])), ('1.weight', tensor([[-0.4403,  0.0146,  0.0514]])), ('1.bias', tensor([0.1762]))])
        """

    # @unittest.skip('debug')
    def test_whole_model_save_load(self):
        """整个模型保存与加载测试.
        """
        print('{} test_whole_model_save_load {}'.format('-'*15, '-'*15))
        net = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
        print(net.state_dict())
        """
        OrderedDict([('0.weight', tensor([[-0.1631, -0.0345],
        [ 0.3992, -0.1971],
        [-0.2313, -0.2398]])), ('0.bias', tensor([-0.1279,  0.0706,  0.7025])), ('1.weight', tensor([[-0.3476,  0.0543,  0.4400]])), ('1.bias', tensor([-0.3389]))])
        """
        torch.save(net, './data/save/whole_model.pt')
        net2 = torch.load('./data/save/whole_model.pt')
        print(net2)
        """
        Sequential(
        (0): Linear(in_features=2, out_features=3, bias=True)
        (1): Linear(in_features=3, out_features=1, bias=True)
        )
        """
        print(net2.state_dict())
        """与保存前参数一致
        OrderedDict([('0.weight', tensor([[-0.1631, -0.0345],
        [ 0.3992, -0.1971],
        [-0.2313, -0.2398]])), ('0.bias', tensor([-0.1279,  0.0706,  0.7025])), ('1.weight', tensor([[-0.3476,  0.0543,  0.4400]])), ('1.bias', tensor([-0.3389]))])
        """


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
