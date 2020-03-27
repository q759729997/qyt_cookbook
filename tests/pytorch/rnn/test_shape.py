"""
    main_module - 输出形状计算，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest

import torch
from torch import nn


class TestShape(unittest.TestCase):
    """输出形状计算.

    Main methods:
        test_Conv2d - RNN.
    """
    # @unittest.skip('debug')
    def test_RNN(self):
        """RNN.
        """
        print('{} test_RNN {}'.format('-'*15, '-'*15))
        vocab_size = 200
        rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=10)
        seq_len = 35
        batch_size = 8
        state = None
        X = torch.rand(seq_len, batch_size, vocab_size)
        print(X.shape)  # 输入形状为 (seq_len, batch, input_size)；torch.Size([35, 8, 200])
        Y, state_new = rnn_layer(X, state)
        print(Y.shape)  # 输出形状为(seq_len, batch, num_directions * hidden_size)；torch.Size([35, 8, 10])
        print(state_new.shape)  # 隐藏状态h的形状为(num_layers * num_directions, batch, hidden_size);torch.Size([1, 8, 10])
        print('{} num_layers=3,bidirectional {}'.format('-'*15, '-'*15))
        rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=10, num_layers=3, bidirectional=True)
        state = None
        X = torch.rand(seq_len, batch_size, vocab_size)
        print(X.shape)  # 输入形状为(seq_len, batch, input_size)；torch.Size([35, 8, 200])
        Y, state_new = rnn_layer(X, state)
        print(Y.shape)  # 输出形状为(seq_len, batch, num_directions * hidden_size)；torch.Size([35, 8, 20])
        print(state_new.shape)  # 隐藏状态h的形状为(num_layers * num_directions, batch, hidden_size)；torch.Size([6, 8, 10])



if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
