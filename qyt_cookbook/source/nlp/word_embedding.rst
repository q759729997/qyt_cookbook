==================
词嵌入
==================

- 词嵌入（word embedding）：词向量是用来表示词的向量，也可被认为是词的特征向量或表征。把词映射为实数域向量的技术也叫词嵌入。

Word2vec
######################

- Word2Vec-知其然知其所以然: https://www.zybuluo.com/Dounm/note/591752
- word2vec工具的提出正是为了解决“由于任何两个不同词的one-hot向量的余弦相似度都为0，多个不同词之间的相似度难以通过one-hot向量准确地体现出来”这个问题。它将每个词表示成一个定长的向量，并使得这些向量能较好地表达不同词之间的相似和类比关系。word2vec工具包含了两个模型，即跳字模型（skip-gram）和连续词袋模型（continuous bag of words，CBOW）。

跳字模型
***************************

- 跳字模型（skip-gram）：假设基于某个词来生成它在文本序列周围的词。举个例子，假设文本序列是“the”“man”“loves”“his”“son”。以“loves”作为中心词，设背景窗口大小为2。如下图所示，跳字模型所关心的是，给定中心词“loves”，生成与它距离不超过2个词的背景词“the”“man”“his”“son”的条件概率，即

.. image:: ./word_embedding.assets/skip_gram_20200331231125.png
    :alt:
    :align: center

.. math::

    P(\textrm{``the"},\textrm{``man"},\textrm{``his"},\textrm{``son"}\mid\textrm{``loves"}).

- 假设给定中心词的情况下，背景词的生成是相互独立的，那么上式可以改写成

.. math::

    P(\textrm{``the"}\mid\textrm{``loves"})\cdot P(\textrm{``man"}\mid\textrm{``loves"})\cdot P(\textrm{``his"}\mid\textrm{``loves"})\cdot P(\textrm{``son"}\mid\textrm{``loves"}).

- 在跳字模型中，每个词被表示成两个 :math:`d` 维向量，用来计算条件概率。假设这个词在词典中索引为 :math:`i` ，当它为中心词时向量表示为 :math:`\boldsymbol{v}_i\in\mathbb{R}^d` ，而为背景词时向量表示为 :math:`\boldsymbol{u}_i\in\mathbb{R}^d` 。设中心词 :math:`w_c` 在词典中索引为 :math:`c` ，背景词 :math:`w_o` 在词典中索引为 :math:`o` ，给定中心词生成背景词的条件概率可以通过对向量内积做softmax运算而得到：

.. math::

    P(w_o \mid w_c) = \frac{\text{exp}(\boldsymbol{u}_o^\top \boldsymbol{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\boldsymbol{u}_i^\top \boldsymbol{v}_c)},

- 其中词典索引集 :math:`\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}` 。假设给定一个长度为 :math:`T` 的文本序列，设时间步 :math:`t` 的词为 :math:`w^{(t)}` 。假设给定中心词的情况下背景词的生成相互独立，当背景窗口大小为 :math:`m` 时，跳字模型的似然函数即给定任一中心词生成所有背景词的概率

.. math::

    \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),

- 这里小于1和大于 :math:`T` 的时间步可以忽略。


训练跳字模型
========================

- 跳字模型的参数是每个词所对应的中心词向量和背景词向量。训练中我们通过最大化似然函数来学习模型参数，即最大似然估计。这等价于最小化以下损失函数：

.. math::

    - \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \text{log}\, P(w^{(t+j)} \mid w^{(t)}).

- 如果使用随机梯度下降，那么在每一次迭代里我们随机采样一个较短的子序列来计算有关该子序列的损失，然后计算梯度来更新模型参数。梯度计算的关键是条件概率的对数有关中心词向量和背景词向量的梯度。根据定义，首先看到

.. math::

    \log P(w_o \mid w_c) = \boldsymbol{u}_o^\top \boldsymbol{v}_c - \log\left(\sum_{i \in \mathcal{V}} \text{exp}(\boldsymbol{u}_i^\top \boldsymbol{v}_c)\right)

- 通过微分，我们可以得到上式中 :math:`\boldsymbol{v}_c` 的梯度

.. math::

    \begin{aligned}
    \frac{\partial \text{log}\, P(w_o \mid w_c)}{\partial \boldsymbol{v}_c}
    &= \boldsymbol{u}_o - \frac{\sum_{j \in \mathcal{V}} \exp(\boldsymbol{u}_j^\top \boldsymbol{v}_c)\boldsymbol{u}_j}{\sum_{i \in \mathcal{V}} \exp(\boldsymbol{u}_i^\top \boldsymbol{v}_c)}\\
    &= \boldsymbol{u}_o - \sum_{j \in \mathcal{V}} \left(\frac{\text{exp}(\boldsymbol{u}_j^\top \boldsymbol{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\boldsymbol{u}_i^\top \boldsymbol{v}_c)}\right) \boldsymbol{u}_j\\
    &= \boldsymbol{u}_o - \sum_{j \in \mathcal{V}} P(w_j \mid w_c) \boldsymbol{u}_j.
    \end{aligned}

- 它的计算需要词典中所有词以 :math:`w_c` 为中心词的条件概率。有关其他词向量的梯度同理可得。
- 训练结束后，对于词典中的任一索引为 :math:`i` 的词，我们均得到该词作为中心词和背景词的两组词向量 :math:`\boldsymbol{v}_i` 和 :math:`\boldsymbol{u}_i` 。在自然语言处理应用中，一般使用跳字模型的中心词向量作为词的表征向量。

连续词袋模型
***************************

- 连续词袋模型（continuous bag of words，CBOW）与跳字模型类似。与跳字模型最大的不同在于，连续词袋模型假设基于某中心词在文本序列前后的背景词来生成该中心词。在同样的文本序列“the”“man”“loves”“his”“son”里，以“loves”作为中心词，且背景窗口大小为2时，连续词袋模型关心的是，给定背景词“the”“man”“his”“son”生成中心词“loves”的条件概率（如下图所示），也就是

.. image:: ./word_embedding.assets/cbow_20200331231846.png
    :alt:
    :align: center

.. math::

    P(\textrm{``loves"}\mid\textrm{``the"},\textrm{``man"},\textrm{``his"},\textrm{``son"}).

- 因为连续词袋模型的背景词有多个，我们将这些背景词向量取平均，然后使用和跳字模型一样的方法来计算条件概率。设 :math:`\boldsymbol{v_i}\in\mathbb{R}^d` 和 :math:`\boldsymbol{u_i}\in\mathbb{R}^d` 分别表示词典中索引为 :math:`i` 的词作为背景词和中心词的向量（注意符号的含义与跳字模型中的相反）。设中心词 :math:`w_c` 在词典中索引为 :math:`c` ，背景词 :math:`w_{o_1}, \ldots, w_{o_{2m}}` 在词典中索引为 :math:`o_1, \ldots, o_{2m}` ，那么给定背景词生成中心词的条件概率

.. math::

    P(w_c \mid w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\text{exp}\left(\frac{1}{2m}\boldsymbol{u}_c^\top (\boldsymbol{v}_{o_1} + \ldots + \boldsymbol{v}_{o_{2m}}) \right)}{ \sum_{i \in \mathcal{V}} \text{exp}\left(\frac{1}{2m}\boldsymbol{u}_i^\top (\boldsymbol{v}_{o_1} + \ldots + \boldsymbol{v}_{o_{2m}}) \right)}.

- 为了让符号更加简单，我们记 :math:`\mathcal{W}_o= \{w_{o_1}, \ldots, w_{o_{2m}}\}` ，且 :math:`\bar{\boldsymbol{v}}_o = \left(\boldsymbol{v}_{o_1} + \ldots + \boldsymbol{v}_{o_{2m}} \right)/(2m)` ，那么上式可以简写成

.. math::

    P(w_c \mid \mathcal{W}_o) = \frac{\exp\left(\boldsymbol{u}_c^\top \bar{\boldsymbol{v}}_o\right)}{\sum_{i \in \mathcal{V}} \exp\left(\boldsymbol{u}_i^\top \bar{\boldsymbol{v}}_o\right)}.

- 给定一个长度为 :math:`T` 的文本序列，设时间步 :math:`t` 的词为 :math:`w^{(t)}` ，背景窗口大小为 :math:`m` 。连续词袋模型的似然函数是由背景词生成任一中心词的概率

.. math::

    \prod_{t=1}^{T}  P(w^{(t)} \mid  w^{(t-m)}, \ldots,  w^{(t-1)},  w^{(t+1)}, \ldots,  w^{(t+m)}).

训练连续词袋模型
========================

- 训练连续词袋模型同训练跳字模型基本一致。连续词袋模型的最大似然估计等价于最小化损失函数

.. math::

    -\sum_{t=1}^T  \text{log}\, P(w^{(t)} \mid  w^{(t-m)}, \ldots,  w^{(t-1)},  w^{(t+1)}, \ldots,  w^{(t+m)}).

- 注意到

.. math::

    \log\,P(w_c \mid \mathcal{W}_o) = \boldsymbol{u}_c^\top \bar{\boldsymbol{v}}_o - \log\,\left(\sum_{i \in \mathcal{V}} \exp\left(\boldsymbol{u}_i^\top \bar{\boldsymbol{v}}_o\right)\right).

- 通过微分，我们可以计算出上式中条件概率的对数有关任一背景词向量 :math:`\boldsymbol{v}_{o_i}` （ :math:`i = 1, \ldots, 2m` ）的梯度

.. math::

    \frac{\partial \log\, P(w_c \mid \mathcal{W}_o)}{\partial \boldsymbol{v}_{o_i}} = \frac{1}{2m} \left(\boldsymbol{u}_c - \sum_{j \in \mathcal{V}} \frac{\exp(\boldsymbol{u}_j^\top \bar{\boldsymbol{v}}_o)\boldsymbol{u}_j}{ \sum_{i \in \mathcal{V}} \text{exp}(\boldsymbol{u}_i^\top \bar{\boldsymbol{v}}_o)} \right) = \frac{1}{2m}\left(\boldsymbol{u}_c - \sum_{j \in \mathcal{V}} P(w_j \mid \mathcal{W}_o) \boldsymbol{u}_j \right).

- 有关其他词向量的梯度同理可得。同跳字模型不一样的一点在于，我们一般使用连续词袋模型的背景词向量作为词的表征向量。
