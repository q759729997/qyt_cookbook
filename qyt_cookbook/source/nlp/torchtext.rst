==================
torchtext工具包
==================

- 官方github： https://github.com/pytorch/text
- 查看它目前提供的预训练词嵌入的名称，预训练模型的命名规范是“模型.（数据集.）数据集词数.词向量维度”

.. code-block:: python

    print(torchtext.vocab.pretrained_aliases.keys())
    # dict_keys(['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d'])

- 使用基于维基百科子集预训练的50维GloVe词向量。第一次创建预训练词向量实例时会自动下载相应的词向量到 ``cache`` 指定文件夹（默认为`.vector_cache`），因此需要联网。

.. code-block:: python

	cache_dir = './data/torchtext'
	# glove = vocab.pretrained_aliases["glove.6B.50d"](cache=cache_dir)
	glove = vocab.GloVe(name='6B', dim=50, cache=cache_dir) # 与上面等价
	print("一共包含%d个词。" % len(glove.stoi))  # 一共包含400000个词。

- 返回的实例主要有以下三个属性：

	- ``stoi``: 词到索引的字典：
	- ``itos``: 一个列表，索引到词的映射；
	- ``vectors``: 词向量。

- 我们可以通过词来获取它在词典中的索引，也可以通过索引获取词。

.. code-block:: python

	glove.stoi['beautiful'], glove.itos[3366] # (3366, 'beautiful')
