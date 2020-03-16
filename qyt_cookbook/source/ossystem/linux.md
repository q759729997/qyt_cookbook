# linux

- `shell`多行输入，使用`\`换行连接：

~~~shell
python -m rasa shell nlu \
    --model ./example/demo/models
~~~

## 文件操作

- 查看当前目录下各个文件与文件夹所占空间：

~~~shell
du -h --max-depth=1 ./
~~~

## 系统信息

- 查看内核版本：

~~~shell
cat /proc/version
~~~

