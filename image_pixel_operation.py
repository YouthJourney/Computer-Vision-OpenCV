# -*-coding:utf-8-*-
"""
File Name: image_pixel_operation.py
Program IDE: PyCharm
Create File By Author: Hong
"""
import cv2 as cv
import numpy as np


def image_pixel(image_path: str):
    """
    获取图像的高和宽，遍历图像的所有像素值。
    :param image_path: 传入图像文件路径
    :return: 没有返回值
    """
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    cv.imshow('input', img)

    h, w, c = img.shape
    # 遍历像素点，修改图像b,g,r值
    for row in range(h):
        for col in range(w):
            b, g, r = img[row, col]
            # img[row, col] = (255 - b, 255 - g, 255 - r)  # 所有通道颜色取反
            # img[row, col] = (255 - b, g, r)  # b通道颜色取反
            # img[row, col] = (255 - b, g, 255 - r)  # b和r通道颜色取反
            img[row, col] = (0, g, r)

    cv.imshow('result', img)
    # cv.imwrite('images/result.jpg', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def math_pixel(image_path: str):
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    cv.imshow('input', img)
    h, w, c = img.shape

    blank = np.zeros_like(img)
    blank[:, :] = (50, 50, 50)  # 所有像素值设为50

    # 改变图像亮度
    mask = cv.add(img, blank)  # 图像加操作。图像形状一样就可以相加，像素值类型不一样不影响, 人为的增加了亮度
    mask = cv.subtract(img, blank)  # 图像减操作。人为的降低了亮度

    # 改变图像对比度
    result = cv.divide(img, blank)  # 图像除操作
    # result = cv.multiply(img, blank)  # 图像乘操作

    # cv.imshow('blank', blank)
    cv.imshow('mask', mask)
    cv.imshow('contrast', result)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    path = 'images/2.png'
    # image_pixel(path)
    math_pixel(path)
