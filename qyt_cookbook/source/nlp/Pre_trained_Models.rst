==================
预训练模型
==================

- 预训练模型（Pre-trained Models，PTM）
- 使用预训练模型的理由：

    - 在大语料下预训练的模型可以学习到 universal language representations，来帮助下游任务
    - PTMs 提供了一个更好的初始化模型，可以提高目标任务的效果和加速收敛
    - PTMs 可以看做是一种正则，防止模型在小数据集上的过拟合

- 预训练模型分类

    - LM（Language Modeling）是 NLP 中最常见的无监督任务，通常特指自回归或单向语言建模，BiLM 虽然结合了两个方向的语言模型，但只是两个方向的简单拼接，并不是真正意义上的双向语言模型。MLM（Masked Language Modeling）可以克服传统单向语言模型的缺陷，结合双向的信息，但是 [MASK] 的引入使得预训练和 fine-tune 之间出现 gap，PLM（Permuted Language Modeling）则克服了这个问题，实现了双向语言模型和自回归模型的统一。
    - DAE（Denoising Autoencoder）接受部分损坏的输入，并以恢复原始输入为目标。与 MLM 不同，DAE 会给输入额外加一些噪声。
    - CTL（Contrastive Learning） 的原理是在对比中学习，其假设是一些 observed pairs of text 在语义上比随机采样的文本更为接近。CTL 比 LM 计算复杂度更低。

参考文章
######################

邱锡鹏预训练模型综述
***************************

- 参考文献： Qiu X , Sun T , Xu Y , et al. Pre-trained Models for Natural Language Processing: A Survey[J]. 2020.
- 论文解读： https://zhuanlan.zhihu.com/p/139015428

BERT
######################

- bert ernie bert_wwm bert_wwwm_ext等模型只是权重不一样，而模型本身主体一样，因此参数model_type=bert其余同理。
- BERT-large模型：24-layer, 1024-hidden, 16-heads, 330M parameters
- BERT-base模型：12-layer, 768-hidden, 12-heads, 110M parameters

bert_wwm
***************************

- Whole Word Masking (wwm)，暂翻译为全词Mask或整词Mask，是谷歌在2019年5月31日发布的一项BERT的升级版本，主要更改了原预训练阶段的训练样本生成策略。 简单来说，原有基于WordPiece的分词方式会把一个完整的词切分成若干个子词，在生成训练样本时，这些被分开的子词会随机被mask。 在全词Mask中，如果一个完整的词的部分WordPiece子词被mask，则同属该词的其他部分也会被mask，即全词Mask。
- github主页： https://github.com/ymcui/Chinese-BERT-wwm

ALBERT
***************************

- huggingface模型所属组： voidful
- github主页： https://github.com/brightmart/albert_zh
- pytorch版本下载： https://github.com/lonePatient/albert_pytorch/blob/master/README_zh.md
- 参考文献： Lan Z , Chen M , Goodman S , et al. ALBERT: A Lite BERT for Self-supervised Learning of Language Representations[J]. 2019.
- 对BERT进行了三个改造 Three main changes of ALBert from Bert：

    - 词嵌入向量参数的因式分解 Factorized embedding parameterization
    - 跨层参数共享 Cross-Layer Parameter Sharing
    - 段落连续性任务 Inter-sentence coherence loss.

- 不同版本： 

    - albert_tiny使用同样的大规模中文语料数据（文件大小16M、参数为4M），层数仅为4层、hidden size等向量维度大幅减少; 尝试使用如下学习率来获得更好效果：{2e-5, 6e-5, 1e-4} 
    - 

Huggingface-Transformers
############################################

- Transformers: State-of-the-art Natural Language Processing for Pytorch and TensorFlow 2.0. https://huggingface.co/transformers
- github主页： https://github.com/huggingface/transformers
- models搜索（我的用户名qiaoyongtian）： https://huggingface.co/models