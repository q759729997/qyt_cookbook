==================
工具模块
==================

iterable数据操作
######################

排列组合
***************************

- 排列组合参考： https://www.cnblogs.com/xiao-apple36/p/10861830.html
- 笛卡尔积

.. code-block:: python

    itertools.product('ABCD', repeat=2)
    # AA AB AC AD BA BB BC BD CA CB CC CD DA DB DC DD

- 排列

.. code-block:: python

    itertools.permutations('ABCD', 2)
    # AB AC AD BA BC BD CA CB CD DA DB DC DC DD

- 组合

.. code-block:: python

    itertools.combinations('ABCD', 2)
    # AB AC AD BC BD CD

    itertools.combinations_with_replacement('ABCD', 2)
    # AA AB AC AD BB BC BD CC CD DD