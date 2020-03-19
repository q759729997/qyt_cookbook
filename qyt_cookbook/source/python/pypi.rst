==================
pypi相关
==================

常用pypi源
######################

- 豆瓣： ``http://pypi.douban.com/simple/``
- 阿里云： ``http://mirrors.aliyun.com/pypi/simple/``
- 清华： ``https://pypi.tuna.tsinghua.edu.cn/simple``

pip安装报错解决
######################

- 安装 ``gensim`` 时报错，改用清华源后成功安装：

.. code-block:: shell

    pip install gensim -i https://pypi.tuna.tsinghua.edu.cn/simple
