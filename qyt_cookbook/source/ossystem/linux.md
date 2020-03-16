# linux

- `shell`多行输入，使用`\`换行连接：

~~~shell
python -m rasa shell nlu \
    --model ./example/demo/models
~~~

## 文件操作

- `gz`压缩与解压：

~~~shell
tar zvcf a.tar.gz a/  # 压缩一个文件夹
tar zvcf a.tar.gz a/ b/ c/  # 压缩多个文件夹
tar -tzvf test.tar.gz  # 列出压缩文件内容
tar -xzvf test.tar.gz  # 解压
~~~

- `zip`压缩与解压：

~~~shell
zip -r a.zip a/  # 压缩一个文件夹
unzip **.zip  # 解压
~~~

- 查看当前目录下各个文件与文件夹所占空间：

~~~shell
du -h --max-depth=1 ./
~~~

## 系统信息

- 查看内核版本：

~~~shell
cat /proc/version
~~~

