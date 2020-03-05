# qyt_cookbook

- 我的API手册
- 参考资料：<https://www.xncoding.com/2017/01/22/fullstack/readthedoc.html>
- `Sphinx`语法：<https://pythonhosted.org/an_example_pypi_project/sphinx.html#tables>
- `Sphinx`使用手册：<https://zh-sphinx-doc.readthedocs.io/en/latest/tutorial.html>
- `toctree`目录编写参考：<http://www.pythondoc.com/sphinx/markup/toctree.html>

## Sphinx

- 安装

~~~shell
pip install sphinx sphinx-autobuild sphinx_rtd_theme
~~~

- 项目初始化

~~~shell
> Separate source and build directories (y/n) [n]:y
> Project name: qyt_cookbook
> Author name(s): qiaoyongtian
> Project version []: 0.0.1
> Project release [1.0]: 0.0.1
> Project language [en]: zh_CN
~~~

- `windows`下编译，在对应目录下执行，进入`build/html`目录后用浏览器打开`index.html`：

~~~shell
make.bat html
~~~

## Read the Docs

- 官网：<https://readthedocs.org/>
- 使用github登录
- 导入项目
