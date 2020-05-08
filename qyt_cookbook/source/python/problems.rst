==================
常见问题汇总
==================

运行环境
######################

pycharm问题
***************************

- Q：运行脚本时，相对路径问题。
- A：在 ``Run/Debug Configurations`` 设置 ``working directory`` ,参考页面： https://www.cnblogs.com/geoffreyone/p/10767801.html

matplotlib
######################

- Q:``plt.imshow(img_obj)`` 运行后，不显示图片.
- A:代码中增加： ``pylab.show()``