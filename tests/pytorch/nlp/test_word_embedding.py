"""
    main_module - 词嵌入，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest

import torch
from torch import nn


class TestWordEmbedding(unittest.TestCase):
    """词嵌入.

    Main methods:
        test_Embedding - Embedding.
    """
    # @unittest.skip('debug')
    def test_Embedding(self):
        """Embedding.
        """
        print('{} test_Embedding {}'.format('-'*15, '-'*15))
        vocabulary_size = 20
        embed = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=4)
        print(embed.weight.shape)  # torch.Size([20, 4])
        x = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        print(x.shape)  # torch.Size([2, 4])
        y = embed(x)
        print(y.shape)  # torch.Size([2, 4, 4])
        print(y)
        """
        tensor([[[ 2.0844,  0.4803, -0.8572,  0.5418],
         [ 0.9904,  1.0953,  0.4437, -0.2486],
         [-0.5855, -0.3845,  0.9660,  2.9621],
         [ 0.7133, -0.2607, -0.3606,  0.9859]],

        [[-0.5855, -0.3845,  0.9660,  2.9621],
         [ 0.8078,  0.9533,  0.1714, -0.1832],
         [ 0.9904,  1.0953,  0.4437, -0.2486],
         [ 0.7199, -0.0607,  0.1369, -1.6945]]], grad_fn=<EmbeddingBackward>)
        """


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
