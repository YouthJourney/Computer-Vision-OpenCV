# -*- encoding: utf-8 -*-
"""
@Date ： 2020/9/28 16:14
@Author ： LGD
@File ：bird_segmentation.py
@IDE ：PyCharm
"""
import cv2 as cv
import numpy as np

# OpenCV读取图片
img = cv.imread("images/bird.png")
cv.imshow('img', img)

# 图片滤波处理
blur = cv.blur(img, (5, 5))
blur0 = cv.medianBlur(blur, 5)
blur1 = cv.GaussianBlur(blur0, (5, 5), 0)
blur2 = cv.bilateralFilter(blur1, 9, 75, 75)

# cv.imshow('img', blur2)

# 转换颜色空间
hsv = cv.cvtColor(blur2, cv.COLOR_BGR2HSV)

# 阈值分割， 颜色分类器慢慢找出来的，有点麻烦。
low_blue = np.array([55, 0, 0])
high_blue = np.array([118, 225, 225])
mask = cv.inRange(hsv, low_blue, high_blue)

# cv.imshow('mask', mask)

res = cv.bitwise_and(img, img, mask=mask)
cv.imshow("res", res)
cv.waitKey(0)
