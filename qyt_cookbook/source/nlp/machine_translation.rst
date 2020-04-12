==================
机器翻译
==================

- 机器翻译是指将一段文本从一种语言自动翻译到另一种语言。
- 可以将编码器—解码器和注意力机制应用于机器翻译中。

评价翻译结果
######################

BLEU
***************************

- 评价机器翻译结果通常使用BLEU（Bilingual Evaluation Understudy）。对于模型预测序列中任意的子序列，BLEU考察这个子序列是否出现在标签序列中。
- 具体来说，设词数为 :math:`n` 的子序列的精度为 :math:`p_n` 。它是预测序列与标签序列匹配词数为 :math:`n` 的子序列的数量与预测序列中词数为 :math:`n` 的子序列的数量之比。举个例子，假设标签序列为 :math:`A` 、 :math:`B` 、 :math:`C` 、 :math:`D` 、 :math:`E` 、 :math:`F` ，预测序列为 :math:`A` 、 :math:`B` 、 :math:`B` 、 :math:`C` 、 :math:`D` ，那么 :math:`p_1 = 4/5,\ p_2 = 3/4,\ p_3 = 1/3,\ p_4 = 0` 。设 :math:`len_{\text{label}}` 和 :math:`len_{\text{pred}}` 分别为标签序列和预测序列的词数，那么，BLEU的定义为

.. math::

    \exp\left(\min\left(0, 1 - \frac{len_{\text{label}}}{len_{\text{pred}}}\right)\right) \prod_{n=1}^k p_n^{1/2^n},

- 其中 :math:`k` 是我们希望匹配的子序列的最大词数。可以看到当预测序列和标签序列完全一致时，BLEU为1。
- 因为匹配较长子序列比匹配较短子序列更难，BLEU对匹配较长子序列的精度赋予了更大权重。例如，当 :math:`p_n` 固定在0.5时，随着 :math:`n` 的增大， :math:`0.5^{1/2} \approx 0.7, 0.5^{1/4} \approx 0.84, 0.5^{1/8} \approx 0.92, 0.5^{1/16} \approx 0.96` 。另外，模型预测较短序列往往会得到较高 :math:`p_n` 值。因此，上式中连乘项前面的系数是为了惩罚较短的输出而设的。举个例子，当 :math:`k=2` 时，假设标签序列为 :math:`A` 、 :math:`B` 、 :math:`C` 、 :math:`D` 、 :math:`E` 、 :math:`F` ，而预测序列为 :math:`A` 、 :math:`B` 。虽然 :math:`p_1 = p_2 = 1` ，但惩罚系数 :math:`\exp(1-6/2) \approx 0.14` ，因此BLEU也接近0.14。

- 参考文献：Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002, July). BLEU: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting on association for computational linguistics (pp. 311-318). Association for Computational Linguistics.