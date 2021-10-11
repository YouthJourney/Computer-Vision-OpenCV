# -*-coding:utf-8-*-
"""
File Name: image_operation.py
Program IDE: PyCharm
Date: 16:24
Create File By Author: Hong
"""
import cv2 as cv
import numpy as np


def resize_image(image_path: str):
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    cv.imshow('input', img)
    h, w, c = img.shape
    # 图片放缩, 指定尺寸
    # dst = cv.resize(img, (w * 2, h * 2), interpolation=cv.INTER_CUBIC)
    # dst = cv.resize(img, (w // 2, h // 2), interpolation=cv.INTER_CUBIC)
    # 缩放比例
    # dst = cv.resize(img, (0, 0), fx=0.75, fy=0.75, interpolation=cv.INTER_CUBIC)
    dst = cv.resize(img, (0, 0), fx=2.0, fy=2.0, interpolation=cv.INTER_CUBIC)

    cv.imshow('resize', dst)

    cv.waitKey(0)
    cv.destroyAllWindows()


def flip_image(image_path: str):
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    cv.imshow('input', img)
    # 上下翻转
    dst1 = cv.flip(img, 0)
    res1 = np.vstack((img, dst1))
    # 左右翻转
    dst2 = cv.flip(img, 1)
    res2 = np.vstack((img, dst2))
    # 对角线翻转
    dst3 = cv.flip(img, -1)
    res3 = np.vstack((img, dst3))

    result = np.hstack((res1, res2, res3))
    cv.imshow('flip', result)
    cv.imwrite('images/result_flip.jpg', result)

    cv.waitKey(0)
    cv.destroyAllWindows()


def rotate_image(image_path: str):
    """
    旋转图像，介绍两种旋转方式。
    1、特定角度旋转函数，但是只支持90、180、270这样特殊的角度旋转。
    2、任意角度旋转函数，需要旋转矩阵M，有两种获取旋转矩阵M的方式：手动配置（可以实现没有裁剪后的旋转图像）和内置函数获取
    :param image_path: 传入的图像文件
    :return: 没有返回值
    """
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    cv.imshow('input', img)

    h, w, c = img.shape

    # ###以下旋转方式获取的都是裁剪后的旋转图像#######
    # ##########手动设置旋转矩阵M#################
    # 定义空矩阵
    M = np.zeros((2, 3), dtype=np.float32)

    # 设定旋转角度
    alpha = np.cos(np.pi / 4.0)
    beta = np.sin(np.pi / 4.0)
    print('alpha: ', alpha)
    # 初始化旋转矩阵
    M[0, 0] = alpha
    M[1, 1] = alpha
    M[0, 1] = beta
    M[1, 0] = -beta

    # 图片中心点坐标
    cx = w / 2
    cy = h / 2

    # 变化的宽高
    tx = (1 - alpha) * cx - beta * cy
    ty = beta * cx + (1 - alpha) * cy

    M[0, 2] = tx
    M[1, 2] = ty

    # 获取旋转矩阵M
    M = cv.getRotationMatrix2D((w/2, h/2), 45, 1)
    # 执行旋转, 任意角度旋转
    result = cv.warpAffine(img, M, (w, h))

    # #######内置旋转函数，仅支持90，180，270#################
    dst1 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    dst2 = cv.rotate(img, cv.ROTATE_180)
    dst3 = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)

    # 将4张图像在一个窗口显示，注意：四张图像形状一致，否则会报错
    res = np.hstack((img, dst1, dst2, dst3))
    cv.imwrite('images/rotate4.jpg', res)
    cv.imshow('res', res)

    result = np.hstack((img, result))
    cv.imwrite('images/rotate2.jpg', result)
    cv.imshow('rotate center', result)

    # # # #######获取没有裁剪的旋转图像#########
    # # 定义空矩阵
    # M = np.zeros((2, 3), dtype=np.float32)
    # # 设定旋转角度
    # alpha = np.cos(np.pi / 4.0)
    # beta = np.sin(np.pi / 4.0)
    # print('alpha: ', alpha)
    # # 初始化旋转矩阵
    # M[0, 0] = alpha
    # M[1, 1] = alpha
    # M[0, 1] = beta
    # M[1, 0] = -beta
    # # 图片中心点坐标
    # cx = w / 2
    # cy = h / 2
    #
    # # 变化的宽高
    # tx = (1 - alpha) * cx - beta * cy
    # ty = beta * cx + (1 - alpha) * cy
    # M[0, 2] = tx
    # M[1, 2] = ty
    #
    # # 旋转后的图像高、宽
    # rotated_w = int(h * np.abs(beta) + w * np.abs(alpha))
    # rotated_h = int(h * np.abs(alpha) + w * np.abs(beta))
    #
    # # 移动后的中心位置
    # M[0, 2] += rotated_w / 2 - cx
    # M[1, 2] += rotated_h / 2 - cy
    #
    # result = cv.warpAffine(img, M, (rotated_w, rotated_h))
    # cv.imshow('result', result)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    path = 'images/daiyutong.png'
    # resize_image(path)
    # flip_image(path)
    rotate_image(path)
