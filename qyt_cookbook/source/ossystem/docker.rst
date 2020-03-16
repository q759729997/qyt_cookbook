.. _header-n0:

docker
======

.. _header-n7:

常用命令
--------

-  官方参考手册：\ https://docs.docker.com/engine/reference/commandline/run/

-  中文手册（docker-从入门到实践）：\ https://yeasy.gitbooks.io/docker_practice/

-  菜鸟教程：\ https://www.runoob.com/docker/docker-tutorial.html

.. _header-n77:

镜像操作
~~~~~~~~

-  官方镜像名称查询：\ https://hub.docker.com/

-  常用命令：

.. code:: shell

   docker save -o 镜像名.tar 镜像名  # 导出镜像
   docker load -i 镜像名.tar  # 导入镜像
   docker import  镜像名.tar 新镜像名  # 导入镜像并设置新的名称
   docker tag 镜像名 新镜像名  # 镜像修改名称tag

.. _header-n96:

容器操作
~~~~~~~~

-  首次启动，根据镜像创建容器：

.. code:: shell

   docker run -dit --name 容器名 镜像名  # 后台运行模式
   docker run -it --name 容器名 镜像名 /bin/bash  # shell界面运行模式
   # 参数-指定运行时的容器名：--name 容器名；若不指定，则会自动生成
   # 参数-端口映射：-p 宿主机端口:容器端口
   # 参数-文件挂载：-v /宿主机路径:/容器机路径 --privileged

-  常用命令：

.. code:: shell

   docker exec -it 容器名 /bin/sh  # 进入容器，且停留在容器内bash界面
   docker ps # 查看当前运行中的容器信息
   docker ps -a # 查看所有容器信息，包括已经退出或者异常的
   docker stop 容器名  # 关闭容器
   docker start 容器名  # 启动容器
   docker commit 容器名  镜像名 # 容器提交为镜像
   docker rm -f 容器名  # 容器删除

-  容器与宿主机文件拷贝：

.. code:: shell

   docker cp 宿主机路径 容器名:容器内路径

-  容器内命令：

.. code:: shell

   Ctrl+P+Q  # 退出容器，不关闭容器
   exit  # 退出容器，关闭容器

.. _header-n65:

docker安装
----------

-  官方参考手册：\ https://docs.docker.com/install/linux/docker-ce/centos/

-  docker依赖系统环境：\ ``Linux Kernel 3.10+``;低于此值时安装非常有可能失败。

-  安装后测试：

.. code:: shell

   docker -v  # 查看版本号
   docker images  # 查看镜像
   docker ps  # 查看运行中的容器
   docker run hello-world  # 运行hello-world镜像

.. _header-n37:

CentOS系统下安装
~~~~~~~~~~~~~~~~

-  离线安装包下载：\ https://download.docker.com/linux/centos/7/x86_64/stable/Packages/

-  离线安装时：使用yum依次安装\ ``container.io`` ``docker-ce-cli``
   ``docker-ce``\ ；上述包安装时，有可能会出现缺少\ ``container-selinux``\ 这个包的情况，若出现，则进行安装。

-  安装后docker服务操作命令：

.. code:: shell

   systemctl start docker  # 启动docker服务
   systemctl restart docker  # 重启docker服务
   systemctl stop docker  # 关闭docker服务

.. _header-n129:

问题记录
^^^^^^^^

-  docker启动\ ``Job for docker.service failed``\ ：\ https://blog.csdn.net/lixiaoyaoboy/article/details/82667510

-  ``Error response from daemon: container bdb30d57482f985713c87d9e240b9a2eb1815bc89e44c607d93c315d85e59de0: driv76186ec: devicemapper: Error running DeleteDevice dm_task_run failed``:`[https://moneyslow.com/docker%E5%AE%B9%E5%99%A8%E5%88%A0%E9%99%A4%E7%8A%B6%E6%80%81%E4%B8%BAremoval-in-progress.html](https://moneyslow.com/docker容器删除状态为removal-in-progress.html) <[https://moneyslow.com/docker%E5%AE%B9%E5%99%A8%E5%88%A0%E9%99%A4%E7%8A%B6%E6%80%81%E4%B8%BAremoval-in-progress.html](https://moneyslow.com/docker容器删除状态为removal-in-progress.html)>`__

.. _header-n127:

docker用户组设置
~~~~~~~~~~~~~~~~

-  非root用户加入docker用户组省去sudo：\ https://blog.csdn.net/u013948858/article/details/78429954

.. code:: shell

   cat /etc/group | grep docker  # 查看用户组
   usermod -aG docker 用户名  # 将相应的用户添加到这个分组
   cat /etc/group  # 检查一下创建是否有效
   systemctl restart docker  # 重启docker服务

.. _header-n124:

dockerfile
----------

-  dockerfile模板：

.. code:: shell

   # docker_test镜像
   FROM docker.io/python:3.6-buster
   # 创建项目根目录
   RUN mkdir /docker_test
   
   WORKDIR /docker_test
   # 拷贝文件
   COPY LICENSE ./
   COPY *.md ./
   COPY requirement* ./
   COPY *.py ./
   COPY 文件夹 ./文件夹
   
   # 依赖安装
   RUN pip install tensorflow-cpu==2.1.0 -i https://pypi.douban.com/simple && \
       pip install -r requirements.txt -i https://pypi.douban.com/simple && \
       # 清理pip安装缓存
       rm -rf /root/.cache/*
   
   CMD ["python", "--version"]
   # 构建镜像：docker build -f dockerfile -t docker_test .

-  相关命令：

.. code:: shell

   # 构建镜像：docker build -f docker_test.dockerfile -t docker_test .
   # 删除镜像：docker rmi -f docker_test
   # 查看镜像安装历史（可以分析每一步占用空间情况）：docker history docker_test
   # 导出镜像：docker save -o docker_test.tar docker_test
   # 压缩镜像：tar -czvf docker_test.tar.gz docker_test.tar
   # 解压镜像：tar -xzvf docker_test.tar.gz
   # 导入镜像：docker load -i docker_test.tar
