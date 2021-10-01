# -*-coding:utf-8-*-
"""
File Name: trackBar_operation.py
Program IDE: PyCharm
Create File By Author: Hong
"""
import cv2 as cv
import numpy as np


def nothing(x):
    pass


def adjust_lightness(image_path: str, value=0, count=100):
    """
    使用滚动滑块调整图像的亮度
    :param image_path: 传入图像文件的路径
    :param value: 滑块初始位置
    :param count: 滑块可以移动的最大值
    :return: 没有返回值
    """
    img = cv.imread(image_path, cv.IMREAD_COLOR)

    cv.namedWindow('input', cv.WINDOW_KEEPRATIO)
    cv.createTrackbar('lightness', 'input', value, count, nothing)
    cv.imshow('input', img)

    blank = np.zeros_like(img)

    while True:
        pos = cv.getTrackbarPos('lightness', 'input')
        blank[:, :] = [pos, pos, pos]
        result = cv.add(img, blank)

        print('lightness: ', pos)
        cv.imshow('result', result)

        c = cv.waitKey(1)
        if c == 25:
            break


def trackbar_to_adjust(image_path: str):
    """
    通过OpenCV的tackBar滑块动态的修改图像的亮度和对比度
    :param image_path: 传入图片文件路径
    :return: 没有返回值
    """
    img = cv.imread(image_path)

    cv.namedWindow('input', cv.WINDOW_KEEPRATIO)
    cv.createTrackbar('lightness', 'input', 0, 100, nothing)  # 第3个参数代表滑块初始位置；第4个参数代表滑块可以滑动的最大值
    cv.createTrackbar('contrast', 'input', 100, 200, nothing)
    cv.imshow('input', img)
    blank = np.zeros_like(img)  # 创建一个与img形状相同的图像blank

    while True:
        light = cv.getTrackbarPos('lightness', 'input')  # 控制亮度的滚动条
        contrast = cv.getTrackbarPos('contrast', 'input') / 100  # 控制对比度的滚动条
        print('light: {}, contrast: {}.'.format(light, contrast))
        result = cv.addWeighted(img, contrast, blank, 0.5, light)

        cv.imshow('result', result)
        c = cv.waitKey(1)
        if c == 27:
            break

    cv.destroyAllWindows()


def keyboard_response(image_path: str):
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    cv.namedWindow('input', cv.WINDOW_AUTOSIZE)
    cv.imshow('input', img)

    while True:
        c = cv.waitKey(1)

        if c == 49:  # 1
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            cv.imshow('result', gray)

        if c == 50:  # 2
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            cv.imshow('result', hsv)

        if c == 51:  # 3
            invert = cv.bitwise_not(img)
            cv.imshow('result', invert)
        if c == 27:  # esc
            break
    cv.destroyAllWindows()


if __name__ == '__main__':
    path = 'images/daiyutong.png'
    trackbar_to_adjust(path)
    # adjust_lightness(path, value=50)
    # keyboard_response(path)
