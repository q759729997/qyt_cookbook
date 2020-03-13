.. _header-n23:

TensorFlow 环境配置
===================

.. _header-n25:

参考资料
--------

-  官方网站： https://tensorflow.google.cn/

.. _header-n30:

import出错
----------

-  查看是否有多个TensorFlow版本，包括tensorflow与tensorflow-gpu；若有多个则对不必要的进行清理

-  需要查看cuda与TensorFlow版本是否一致，如果不需要cuda版本的，则更换安装tensorflow-cpu

-  Windows下报错，有可能是C++依赖不完整，搜索\ ``VC_redist.x64.exe``\ ，下载安装后重启电脑。
