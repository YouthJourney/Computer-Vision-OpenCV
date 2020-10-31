# -*- coding: UTF-8 -*-
"""
Author: LGD
FileName: test_ship
DateTime: 2020/10/27 10:49 
SoftWare: PyCharm
"""
import cv2 as cv
import numpy as np

camera = cv.VideoCapture("images/ship1.mp4")
# cv.imshow("video", camera)
while 1:
    # 读图片灰度化
    ret, frame = camera.read()  # 读取帧
    print(ret)
    # img_path = "images/bird.png"
    # img = cv.imread(img_path)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow("gray", gray)

    # 高斯滤波去噪
    blurred = cv.GaussianBlur(gray, (9, 9), 0)
    # cv.imshow("blurred", blurred)

    # 提取图像梯度
    gradX = cv.Sobel(gray, ddepth=cv.CV_32F, dx=1, dy=0)
    gradY = cv.Sobel(gray, ddepth=cv.CV_32F, dx=0, dy=1)

    gradient = cv.subtract(gradX, gradY)
    gradient = cv.convertScaleAbs(gradient)

    cv.namedWindow("gradient", 0)
    cv.resizeWindow("gradient", 800, 450)
    cv.imshow("gradient", gradient)
    if cv.waitKey(1) & 0xFF == ord('q'):  # 按Q退出
        break

# 释放资源并关闭窗口
camera.release()
cv.destroyAllWindows()
