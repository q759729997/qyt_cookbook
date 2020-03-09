# module详解

## module引用顺序

- 首先判断这个module是不是built-in即内建模块，如果是则引入内建模块，如果不是则在一个称为sys.path的list中寻找。`sys.path`在python脚本执行时动态生成，包括以下3个部分：
  - 脚本执行的位置，即当前路径。
  - 环境变量中的`PYTHONPATH`, 即`.bash_profile`。
  - 安装python时的依赖位置
- python程序中使用`import XXX`时，python解析器会在当前目录、已安装和第三方模块中搜索 xxx，如果都搜索不到就会报错。
- 使用`sys.path.append()`方法可以临时添加搜索路径，方便更简洁的import其他包和模块。这种方法导入的路径会在python程序退出后失效。

~~~python
# 加入上层目录和绝对路径
import sys
sys.path.append('..') #表示导入当前文件的上层目录到搜索路径中
sys.path.append('/home/model') # 绝对路径
# 加入当前目录
import os,sys
sys.path.append(os.getcwd())  # os.getcwd()用于获取当前工作目录
# 定义搜索优先顺序
import sys
sys.path.insert(1, "./model")  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级
~~~

- 测试本项目时，可以使用`sys.path.insert`设置当前目录优先级，以避免与已安装包冲突。

~~~python
import sys

sys.path.insert(0, './')  # 定义搜索路径的优先顺序，序号从0开始，表示最大优先级

import qytPython  # noqa
print('qytPython module path :{}'.format(qytPython.__file__))  # 输出测试模块文件位置
~~~

