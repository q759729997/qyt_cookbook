.. _header-n100:

linux
=====

-  ``shell``\ 多行输入，使用\ ``\``\ 换行连接：

.. code:: shell

   python -m rasa shell nlu \
       --model ./example/demo/models

.. _header-n106:

文件操作
--------

-  ``gz``\ 压缩与解压：

.. code:: shell

   tar zvcf a.tar.gz a/  # 压缩一个文件夹
   tar zvcf a.tar.gz a/ b/ c/  # 压缩多个文件夹
   tar -tzvf test.tar.gz  # 列出压缩文件内容
   tar -xzvf test.tar.gz  # 解压

-  查看文件信息：

.. code:: shell

   file -i 文件路径

-  文件合并：

.. code:: shell

   cat 文件1 文件2 > 新文件

-  文件转码 http://linux.51yip.com/search/iconv\ ：

.. code:: shell

   iconv -l  # 查看所有编码
   iconv abc.sh -o utf8  # 将文件转换成utf8格式
   iconv test.txt -f GBK -t UTF-8 -o test2.txt # GBK转utf8
   iconv test.txt -f UTF-8 -t GBK -o test2.txt # utf8转GBK

-  ``zip``\ 压缩与解压：

.. code:: shell

   zip -r a.zip a/  # 压缩一个文件夹
   unzip **.zip  # 解压

-  查看当前目录下各个文件与文件夹所占空间：

.. code:: shell

   du -h --max-depth=1 ./

-  软链接

.. code:: shell

   ln -s 源文件或目录 目标文件或目录
   ln -s /data/test.csv /data/share  # 例子

-  简体与繁体转换 https://segmentfault.com/a/1190000010122544\ ：

.. code:: shell

   opencc --version
   # 繁体转简体
   echo '歐幾里得 西元前三世紀的希臘數學家' | opencc -c t2s
   # 输出结果：欧几里得 西元前三世纪的希腊数学家
   # 简体转繁体
   echo '欧几里得 西元前三世纪的希腊数学家' | opencc -c s2t
   # 输出结果：歐幾里得 西元前三世紀的希臘數學家
   # 可以通过以下方式直接对文件进行繁简转换
   opencc -i 文件名 -o 新文件名 -c t2s.json

.. _header-n185:

系统信息
--------

-  查看内核版本：

.. code:: shell

   cat /proc/version

.. _header-n124:

Ubuntu
------

.. _header-n125:

设置国内源
~~~~~~~~~~

-  备份\ ``/etc/apt/sources.list``

.. code:: shell

   cp /etc/apt/sources.list /etc/apt/sources.list.bak

-  在\ ``/etc/apt/sources.list``\ 文件前面添加如下条目

.. code:: shell

   #添加阿里源
   deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
   deb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
   deb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
   deb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
   deb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
   deb-src http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
   deb-src http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
   deb-src http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
   deb-src http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
   deb-src http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse

-  执行如下命令更新源

.. code:: shell

   apt-get update  # 更新源
   apt-get upgrade

.. _header-n138:

中文语言环境问题
----------------

-  安装中文语言包

.. code:: shell

   apt-get install language-pack-zh-hans language-pack-zh-hans-base language-pack-gnome-zh-hans language-pack-gnome-zh-hans-base
   apt-get install `check-language-support -l zh-hans`
   locale-gen zh_CN.UTF-8

-  终端输入中文问题

.. code:: shell

   # 打开/etc/environment
   # 在下面添加如下两行
   LANG="zh_CN.UTF-8"
   LANGUAGE="zh_CN:zh:en_US:en"
   
   # 打开 /var/lib/locales/supported.d/local
   # 添加zh_CN.GB2312字符集，如下：
   en_US.UTF-8 UTF-8
   zh_CN.UTF-8 UTF-8
   zh_CN.GBK GBK
   zh_CN GB2312
   # 保存后，执行命令：
   locale-gen
   
   # 打开/etc/default/locale
   # 修改为：
   LANG="zh_CN.UTF-8"
   LANGUAGE="zh_CN:zh:en_US:en"
   
   vim ~/.bashrc # (不要加 sudo)
   # 複製下述這三行 貼在最後面
   export LANG=LANG="zh_CN.utf-8"
   export LANGUAGE="zh_CN:zh:en_US:en"
   export LC_ALL="zh_CN.utf-8"
   
   source ~/.bashrc
   
   ls -al ~/ # 查看是否有效
