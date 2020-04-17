==================
pypi相关
==================

常用pypi源
######################

- 豆瓣： ``http://pypi.douban.com/simple/``
- 阿里云： ``http://mirrors.aliyun.com/pypi/simple/``
- 清华： ``https://pypi.tuna.tsinghua.edu.cn/simple``

Python库收藏
######################

进度条显示
***************************

- tqdm： \ https://github.com/tqdm/tqdm

日期时间
***************************

- 时间操作： \ https://github.com/crsmithdev/arrow

文本处理
***************************

- 字符串模糊匹配： https://github.com/seatgeek/fuzzywuzzy
- 倒排索引： https://github.com/willf/inverted_index
- SQLLITE倒排： https://github.com/lsq960124/Inverted-index-BM25

自然语言处理
***************************

- Ngram库： \ https://pythonhosted.org/ngram/ngram.html#ngram-a-set-class-that-supports-lookup-by-n-gram-string-similarity

语料获取
***************************

- web搜索封装： \ https://github.com/fossasia/query-server

UI界面
***************************

- 开源UI框架： https://github.com/kivy/kivy

数据库操作
***************************

- ORM和MySQL数据库的操作： https://www.liaoxuefeng.com/wiki/1016959663602400/1017803857459008

pip安装报错解决
######################

- 安装 ``gensim`` 时报错，改用清华源后成功安装：

.. code-block:: shell

    pip install gensim -i https://pypi.tuna.tsinghua.edu.cn/simple
