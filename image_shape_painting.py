# -*-coding:utf-8-*-
"""
File Name: image_shape_painting.py
Program IDE: PyCharm
Date: 21:12
Create File By Author: Hong
"""
import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def pixel_operation(image_path: str):
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    cv.imshow('input', img)

    # 注意：python中的print函数默认换行，可以用end=''或者接任意字符
    # 像素均值、方差
    means, dev = cv.meanStdDev(img)
    print('means: {}, \n dev: {}'.format(means, dev))
    # 像素最大值和最小值
    min_pixel = np.min(img[:, :, 0])
    max_pixel = np.max(img[:, :, -1])
    print('min: {}, max: {}'.format(min_pixel, max_pixel))

    # 若是一个空白图像
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    # 像素均值、方差
    # blank[:, :] = (255, 0, 255)
    means, dev = cv.meanStdDev(blank)
    print('means: {}, \n dev: {}'.format(means, dev))

    cv.waitKey(0)
    cv.destroyAllWindows()


# 弥补putText()显示中文乱码的问题
def image_add_text(img, text, left, top, text_color, text_size):
    if isinstance(img, np.ndarray):  # 判断是否是opencv图片类型，是就转换Image类型
        image = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    # 创建一个可以在给定图像上绘制的对象
    draw = ImageDraw.Draw(image)
    # 字体的格式
    font_style = ImageFont.truetype("font/simsun.ttc", text_size, encoding='utf-8')
    # 绘制文本
    draw.text((left, top), text, text_color, font=font_style)
    # 转换回opencv格式并返回
    return cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)


def drawing_demo():
    img1 = np.zeros((512, 512, 3), dtype=np.uint8)

    temp = np.copy(img1)
    # 绘制矩形
    cv.rectangle(img1, (50, 50), (400, 400), (0, 0, 255), 4, 8, 0)
    # 绘制圆形
    cv.circle(img1, (200, 200), 100, (255, 0, 0), -1, 8, 0)
    # 绘制直线
    cv.line(img1, (50, 50), (400, 400), (0, 255, 0), 2, 8, 0)
    # 写文字
    # cv.putText(img1, '你好，世界', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, 8)
    img1 = image_add_text(img1, '你好，世界', 50, 50, (255, 0, 0), 20)

    # # 擦除
    # # 方法1
    # img1[:, :, :] = 0
    # # 方法2
    # img1 = temp

    cv.imshow('input', img1)

    cv.waitKey(0)
    cv.destroyAllWindows()


def random_color():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv.imshow('img', img)

    while True:
        # 设置随机位置
        xx = np.random.randint(0, 300, 2, dtype=np.int32)
        yy = np.random.randint(0, 300, 2, dtype=np.int32)
        # 设置随机颜色，可以用于目标检测画不同目标边界框的随机颜色
        bgr = np.random.randint(0, 255, 3, dtype=np.int32)
        print(bgr[0], bgr[1], bgr[2])
        # 画直线，将每个bgr分量int转换以下，不然被认为不是数字，出错。
        cv.line(img, (xx[0], yy[0]), (xx[1], yy[1]), (int(bgr[0]), int(bgr[1]), int(bgr[2])), 1, 8, 0)

        cv.imshow('line', img)
        c = cv.waitKey(1)
        if c == 27:
            break

    cv.destroyAllWindows()


def polygon_drawing():
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)

    # 定义多边形的顶点
    pts = np.array([[100, 100], [350, 100], [450, 280], [320, 450], [80, 400]], dtype=np.int32)
    # 多边形绘制
    # cv.polylines(canvas, [pts], True, (0, 0, 255), 2, 8, 0)
    # 多边形填充
    # cv.fillPoly(canvas, [pts], (255, 0, 255), 8, 0)
    # 既可以填充也可以绘制形状, thickness为时绘制形状，-1时填充形状
    # 可以添加多个轮廓，用,号隔开，比如[pts1, pts2, ...]
    cv.drawContours(canvas, [pts], -1, (255, 0, 0), thickness=1)

    cv.imshow('polyline', canvas)
    cv.imwrite('images/fill_polygon.jpg', canvas)
    cv.waitKey(0)

    cv.destroyAllWindows()


if __name__ == '__main__':
    path = 'images/2.png'
    # pixel_operation(path)
    # drawing_demo()
    random_color()
    # polygon_drawing()
