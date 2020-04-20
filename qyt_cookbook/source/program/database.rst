==================
数据库
==================

MySQL
######################

- SQL语法菜鸟教程： https://www.runoob.com/sql/sql-tutorial.html
- python使用mysql： https://www.liaoxuefeng.com/wiki/1016959663602400/1017802264972000
- 数据库连接工具``Navicat_Premium``，链接： https://pan.baidu.com/s/1jH0d8ZMAdADNGmX6Hr6RIw ; 提取码：twe3

安装
***************************

- 安装教程： https://www.runoob.com/mysql/mysql-install.html
- root账户设置：

.. code:: shell

    mysqladmin -u root password "kd123456";  # 创建root密码
    mysql -u root -pkd123456; # 连接mysql

- 导入数据：

.. code:: shell

    create database duolun;      # 创建数据库
    use duolun;                  # 使用已创建的数据库 
    set names utf8mb4;           # 设置编码；utf8mb4_general_ci utf8mb4_0900_ai_ci
    source /root/mysql/duolun_20200420.sql;  # 导入备份数据库

- 修改root访问权限，使其允许外网访问：

.. code:: shell

    SHOW DATABASES;  USE XX; SHOW TABLES;
    use mysql;
    select  User,authentication_string,Host from user; # 查询用户表命令
    GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' IDENTIFIED BY 'kd123456'; # 设置权限
    flush privileges;  # 刷新