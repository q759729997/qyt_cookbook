.. _header-n0:

单元测试
========

.. _header-n2:

pytest
------

-  跳过某个测试用例：

.. code:: python

   @pytest.mark.skip(reason="跳过原因")

-  运行时显示详细信息：

.. code:: python

   if __name__ == '__main__':
       pytest.main(['-s', './test/import_time.py'])
