==================
绘图
==================

图片数据展示
######################

- 参考文档： https://matplotlib.org/tutorials/introductory/images.html#sphx-glr-tutorials-introductory-images-py
- 直接使用图片文件进行展示

.. code-block:: python

    import pylab
    from PIL import Image
    from matplotlib import pyplot as plt

    img_obj = Image.open(img_file_name)
    plt.imshow(img_obj)
    pylab.show()  # 直接利用plt.imshow()发现居然不能显示图片

- 使用numpy、tensor格式的数据进行展示时 ``plt.imshow(img[:, :, 0])`` ，channel放到最后一维。


