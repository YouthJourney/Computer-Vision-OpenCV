# -*-coding:utf-8-*-
"""
File Name: mouse_response_and_type_conversion.py
Program IDE: PyCharm
Date: 10:04
Create File By Author: Hong
"""
import cv2 as cv
import numpy as np

# 在图像上画矩形框
x1 = -1
y1 = -1
x2 = -1
y2 = -1

# canvas = np.zeros((300, 300, 3), dtype=np.uint8)
canvas = cv.imread('images/2.png', cv.IMREAD_COLOR)
img = np.copy(canvas)


# 回调，系统调用回调函数解决你的问题
# 鼠标响应回调函数，参数固定；对应鼠标事件、横坐标、纵坐标、flags和其他参数
def mouse_drawing(event, x, y, flags, param):
    # print(x, y)
    global x1, y1, x2, y2
    if event == cv.EVENT_LBUTTONDOWN:
        x1 = x
        y1 = y
    if event == cv.EVENT_MOUSEMOVE:
        if x1 < 0 or y1 < 0:
            return
        x2 = x
        y2 = y
        dx = x2 - x1
        dy = y2 - y1
        if dx > 0 and dy > 0:
            # 擦除重叠
            # canvas[:, :] = 0
            canvas[:, :, :] = img[:, :, :]
            cv.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2, 8, 0)
    if event == cv.EVENT_LBUTTONUP:
        x2 = x
        y2 = y
        dx = x2 - x1
        dy = y2 - y1
        if dx > 0 and dy > 0:
            # canvas[:, :] = 0
            canvas[:, :, :] = img[:, :, :]
            cv.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2, 8, 0)

        x1 = -1
        y1 = -1
        x2 = -1
        y2 = -1


def mouse_response():
    cv.namedWindow('Mouse Response', cv.WINDOW_AUTOSIZE)
    # 再某个窗口上设置鼠标响应函数
    cv.setMouseCallback('Mouse Response', mouse_drawing)

    while True:
        cv.imshow('Mouse Response', canvas)
        c = cv.waitKey(1)
        if c == 27:
            break

    cv.destroyAllWindows()


# 图像像素类型转换和归一化
def pixel_normalization(image_path: str):
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    # 可以显示整数和浮点数像素图片
    # 图像归一化
    # 方法1
    # print(img/255.0)
    cv.imshow('input', img / 255.0)

    # 方法2
    result = np.zeros_like(np.float32(img))
    cv.normalize(img, result, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)
    print(result)
    cv.imshow('result', result)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    path = 'images/2.png'
    mouse_response()
    # pixel_normalization(path)
