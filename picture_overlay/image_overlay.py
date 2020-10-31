# -*- encoding: utf-8 -*-
"""
@Date ： 2020/9/30 16:41
@Author ： LGD
@File ：image_overlay.py
@IDE ：PyCharm
"""
import cv2 as cv

img = cv.imread('image.png')

# 图片转换为RGB模式
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 图片转换为灰度图
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 打印图片，发现图像是一堆像素值以类似矩阵的格式存储。
print(img)
print(type(img))
cv.imshow('img', img)

img[50:100, 50:100] = [255, 0, 0]
# print(img[50:100, 50:100])
cv.imshow('img1', img)

img1 = cv.imread('image1.png')
resize_img1 = cv.resize(img1, dsize=(50, 50))  # dsize表示需要修改的尺寸
img[50:100, 50:100] = resize_img1
cv.imshow('img2', img)
cv.waitKey(0)
