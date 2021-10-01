# Computer-Vision-OpenCV
此项目主要是本人在研究生期间在计算机视觉方面的研究所学习的内容和程序，其中有收集前人的代码，也有自己改进的代码。会随时更新，仅供大家参考学习并讨论之用。另外本人是计算机视觉和深度学习方面的在读研究生，有需要一起沟通的话，可以联系微信：sincos246835

## 01 [图像颜色空间转换](https://github.com/YouthJourney/Computer-Vision-OpenCV/blob/master/color_space_conversion.py)

将图像由rgb颜色空间转换到其他空间，查看有什么变化。  

![This is an image](https://github.com/YouthJourney/Computer-Vision-OpenCV/blob/master/images/%E9%A2%9C%E8%89%B2%E7%A9%BA%E9%97%B4%E8%BD%AC%E6%8D%A2.png)

使用`numpy`包创建图像和复制图像。  

![This is an image](https://github.com/YouthJourney/Computer-Vision-OpenCV/blob/master/images/%E5%9B%BE%E5%83%8F%E5%88%9B%E5%BB%BA%E5%92%8C%E5%A4%8D%E5%88%B6.png)

## 02 [图像像素级操作](https://github.com/YouthJourney/Computer-Vision-OpenCV/blob/master/image_pixel_operation.py)

遍历像素点，修改像素值。  

![This is an image](https://github.com/YouthJourney/Computer-Vision-OpenCV/blob/master/images/result.jpg)

图像的加减乘除运算。  

![This is an image](https://github.com/YouthJourney/Computer-Vision-OpenCV/blob/master/images/%E5%9B%BE%E5%83%8F%E5%8A%A0%E5%87%8F.png)

## 03 [图像的亮度和对比度调整](https://github.com/YouthJourney/Computer-Vision-OpenCV/blob/master/trackBar_operation.py)

通过添加`cv2.createTrackerbar()`函数实现动态调整图像亮度和对比度。  

![This is an image](https://github.com/YouthJourney/Computer-Vision-OpenCV/blob/master/images/%E8%B0%83%E6%95%B4%E5%9B%BE%E5%83%8F%E4%BA%AE%E5%BA%A6%E5%92%8C%E5%AF%B9%E6%AF%94%E5%BA%A6.png)

## 04 [图像颜色变化和位级操作](https://github.com/YouthJourney/Computer-Vision-OpenCV/blob/master/color_operation.py)

- 给图像添加颜色

  ![This is an image](https://github.com/YouthJourney/Computer-Vision-OpenCV/blob/master/images/%E9%A2%9C%E8%89%B2%E6%B7%BB%E5%8A%A0.png)
  
- 图像位级运算（与、或、非、异或）

  ![This is an image](https://github.com/YouthJourney/Computer-Vision-OpenCV/blob/master/images/%E4%B8%8E%E6%88%96%E9%9D%9E.png)

- 图像通道运算（分离、合并、交换）

  ![This is an image](https://github.com/YouthJourney/Computer-Vision-OpenCV/blob/master/images/channel.png)

## 05 [一个窗口显示多幅图像](https://github.com/YouthJourney/Computer-Vision-OpenCV/blob/master/color_operation.py)

```python
# -*-coding:utf-8-*-
"""
File Name: color_operation.py
Program IDE: PyCharm
Create File By Author: Hong
"""
import cv2 as cv
import numpy as np


def image_display(image_path: str):
    """
    多个图像在一个窗口内显示
    :param image_path: 传入图像路径
    :return:
    """
    img = cv.imread(image_path, cv.IMREAD_COLOR)

    # 颜色取反
    invert = cv.bitwise_not(img)
    # 高斯模糊
    gaussianBlur = cv.GaussianBlur(img, (0, 0), 10)
    # 镜像
    flip = cv.flip(img, 1)  # 0表示绕x轴翻转；1表示绕y轴翻转；-1表示绕两个轴翻转

    # 方法1：创建一个画布，将所有图像复制到画布中，最后显示画布
    h, w, _ = img.shape

    img_list = [img, invert, gaussianBlur, flip]

    four_view = np.zeros((h * 2 + 10, w * 2 + 10, 3), np.uint8)
    four_view[:, :] = 255  # 给所有通道的像素值赋值255
    for i in range(len(img_list)):
        row = i // 2
        col = i % 2
        print(row, col)
        # 将小图像复制到大画布上。
        np.copyto(four_view[(h + 10) * row:h * (row + 1) + 10 * row, (w + 10) * col:w * (col + 1) + 10 * col],
                  img_list[i])
    cv.imshow('result', four_view)

    # 方法2：使用numpy的水平堆叠和竖直堆叠完成所有图像的堆叠，最后一起显示
    vs1 = np.hstack((img, invert))  # 水平堆叠
    vs2 = np.hstack((gaussianBlur, flip))  # 水平堆叠
    result = np.vstack((vs1, vs2))  # 竖直堆叠

    # 初略解决imshow()中文乱码显示的问题。不能完全解决，有些中文无法显示。这是python-opencv的弊端
    def zh_ch(string):
        return string.encode('gbk').decode(errors='ignore')

    cv.imshow(zh_ch('图片'), result)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    path = 'images/daiyutong.png'
    image_display(path)
```  

两种方法实现一个窗口显示多张图像。  

![This is an image](https://github.com/YouthJourney/Computer-Vision-OpenCV/blob/master/images/method2.jpg)
