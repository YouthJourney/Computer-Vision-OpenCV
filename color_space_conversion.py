# -*-coding:utf-8-*-
"""
File Name: color_space_conversion.py
Program IDE: PyCharm
Create File By Author: Hong
"""
import cv2 as cv
import numpy as np


def color_space_conversion(image_path: str):
    img = cv.imread(image_path)  # BGR读取图像，0~255，苹果手机上颜色是用sRGB，DCI-P3
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转为灰度图
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # 转为HSV颜色空间，H 0~180, S,V 0~255
    bgr = cv.cvtColor(img, cv.COLOR_HSV2BGR)

    cv.imshow('img', img)
    cv.imshow('gray', gray)
    cv.imshow('hsv', hsv)
    cv.imshow('bgr', bgr)
    cv.waitKey(0)
    cv.destroyAllWindows()


def numpy_img(image_path: str):
    """
    在opencv-python中所有的图像都是numpy数组，
    可以使用numpy操作图像，如生成图像、修改图像、复制图像
    :param image_path: 传入图像文件路径
    :return: 没有返回值
    """
    # cv2.IMREAD_COLOR: 默认参数，读入彩色图像，忽略alpha通道，数字1；
    # cv2.IMREAD_GRAYSCALE：读入灰度图片，数字0；
    # cv2.IMREAD_UNCHANGED：读入完整图片，包括alpha通道，数字-1
    img_color = cv.imread(image_path, cv.IMREAD_COLOR)
    img_gray = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    img_unchanged = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    print(img_color.shape)  # H, W, C  (888，500，3）
    print(img_gray.shape)  # H, W  (888, 500)
    print(img_unchanged.shape)  # H, W, C (888, 500, 4)

    h, w, c = img_color.shape
    roi = img_color[190:400, 150:380, :]  # 截取img_color图像的一部分, 第一维度指定高的范围，第二维度指定宽的范围，第三维度是通道
    # 方法1 创建一个和img_color形状相同的空白图片
    # blank = np.zeros_like(img_color)
    # 方法2 使用h w c和数据类型，创建一个空白图像
    # blank = np.zeros((h, w, c), dtype=np.uint8)
    # blank[190:400, 150:380, :] = img_color[190:400, 150:380, :]
    # print(blank)
    # 方法1 图像copy, 改变blank，不会改变img_color
    blank = np.copy(img_color)
    # 方法2 图像赋值, 共用同一个数据，改变blank，就会改变img_color
    # blank = img_color
    cv.imshow('roi', roi)
    cv.imshow('blank', blank)

    cv.imshow('img_color', img_color)
    # cv.imshow('img_gray', img_gray)
    # cv.imshow('img_unchanged', img_unchanged)
    cv.waitKey(0)
    cv.destroyAllWindows()


def image_matting(image_path: str):
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    cv.imshow('input', img)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cv.imshow('hsv', hsv)
    # 根据像素的范围进行过滤，把符合像素范围的保留，不符合的赋值0或者255
    mask = cv.inRange(hsv, (35, 43, 46), (77, 255, 255))
    cv.imshow('cc', mask)
    mask = cv.bitwise_not(mask)

    # 只在mask区域做与运算
    result = cv.bitwise_and(img, img, mask=mask)

    cv.imshow('mask', mask)
    cv.imshow('result', result)

    cv.waitKey(0)
    cv.destroyAllWindows()
    

if __name__ == '__main__':
    path = 'images/2.png'
    color_space_conversion(path)
    # numpy_img(path)
    # image_matting(path)
