.. _header-n0:

计算机视觉
==========

.. _header-n2:

torchvision
-----------

-  torchvision包主要用来构建计算机视觉模型。torchvision主要由以下几部分构成：

   1. ``torchvision.datasets``: 一些加载数据的函数及常用的数据集接口；

   2. ``torchvision.models``:
      包含常用的模型结构（含预训练模型），例如AlexNet、VGG、ResNet等；

   3. ``torchvision.transforms``: 常用的图片变换，例如裁剪、旋转等；

   4. ``torchvision.utils``: 其他的一些有用的方法。

.. _header-n27:

获取数据集
~~~~~~~~~~

-  通过torchvision的\ ``torchvision.datasets``\ 来下载数据集。第一次调用时会自动从网上获取数据。我们通过参数\ ``train``\ 来指定获取训练数据集或测试数据集（testing
   data set）。测试数据集也叫测试集（testing
   set），只用来评价模型的表现，并不用来训练模型。

-  指定参数\ ``transform = transforms.ToTensor()``\ 使所有数据转换为\ ``Tensor``\ ，如果不进行转换则返回的是PIL图片。\ ``transforms.ToTensor()``\ 将尺寸为
   (H x W x C) 且数据位于[0,
   255]的PIL图片或者数据类型为\ ``np.uint8``\ 的NumPy数组转换为尺寸为(C
   x H x W)且数据类型为\ ``torch.float32``\ 且位于[0.0,
   1.0]的\ ``Tensor``\ 。

-  (C x H x W)，第一维是通道数，通道数为1代表是灰度图像。

-  注意：
   由于像素值为0到255的整数，所以刚好是uint8所能表示的范围，包括\ ``transforms.ToTensor()``\ 在内的一些关于图片的函数就默认输入的是uint8型，若不是，可能不会报错但可能得不到想要的结果。所以，\ **如果用像素值(0-255整数)表示图片数据，那么一律将其类型设置成uint8，避免不必要的bug。**
