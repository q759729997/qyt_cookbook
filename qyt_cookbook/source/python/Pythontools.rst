==================
工具模块
==================

文件操作
######################

- 文件夹复制

.. code-block:: python

    import shutil  # 要求newdir必须不存在，否则不能使用
    from distutils.dir_util import copy_tree
    shutil.copytree("olddir","newdir")


- 文件复制

.. code-block:: python

    import shutil
    shutil.copyfile("oldfile","newfile")  # oldfile和newfile都只能是文件
    shutil.copy("oldfile","newfile")  # oldfile只能是文件夹，newfile可以是文件，也可以是目标目录   

编码与解码
######################

- 按文件流读取，对每一行进行解码

.. code-block:: python

    import codecs
    import chardet

    with codecs.open('./README.md', mode='rb') as fr:
        for line in fr:
            encoding = chardet.detect(line)['encoding']
            line = line.decode(encoding, errors='ignore')
            print(encoding, line)

- Unicode转汉字

.. code-block:: python

    import html

    text = '<?xml version="1.0" encoding="UTF-8"?><Message>&#26597;&#35810;&#26816;&#20462;&#21333;</Message>'
    print(html.unescape(text))
    # <?xml version="1.0" encoding="UTF-8"?><Message>查询检修单</Message>

事件与日期
######################

- 时间转时间戳

.. code-block:: python

    import time
    timestr = '2019-02-28 23:59:59'
    print(int(time.mktime(time.strptime(timestr, '%Y-%m-%d %H:%M:%S'))))
    # 1551369599

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
