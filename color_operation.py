# -*-coding:utf-8-*-
"""
File Name: color_operation.py
Program IDE: PyCharm
Create File By Author: Hong
"""
import cv2 as cv
import numpy as np

color_map = [
    cv.COLORMAP_AUTUMN,
    cv.COLORMAP_BONE,
    cv.COLORMAP_JET,
    cv.COLORMAP_WINTER,
    cv.COLORMAP_PARULA,
    cv.COLORMAP_OCEAN,
    cv.COLORMAP_SUMMER,
    cv.COLORMAP_SPRING,
    cv.COLORMAP_COOL,
    cv.COLORMAP_PINK,
    cv.COLORMAP_HOT,
    cv.COLORMAP_PARULA,
    cv.COLORMAP_MAGMA,
    cv.COLORMAP_INFERNO,
    cv.COLORMAP_PLASMA,
    cv.COLORMAP_TWILIGHT,
    cv.COLORMAP_TWILIGHT_SHIFTED
]


def color_operation(image_path: str):
    img = cv.imread(image_path, cv.IMREAD_COLOR)  # 以彩色模式读图像
    cv.namedWindow('input', cv.WINDOW_AUTOSIZE)  # 根据图像大小自动调节窗口大小
    cv.imshow('input', img)

    index = 0

    while True:
        dst = cv.applyColorMap(img, color_map[index % len(color_map)])  # 在原图上应用不同的颜色模式

        cv.imshow('{}'.format(color_map[index % len(color_map)]), dst)
        index += 1

        c = cv.waitKey(1000)
        if c == 27:
            break

    cv.destroyAllWindows()


def bitwise_operation(image_path1: str, image_path2: str):
    img1 = cv.imread(image_path1, cv.IMREAD_COLOR)
    img2 = cv.imread(image_path2, cv.IMREAD_COLOR)
    img2 = cv.resize(img2, (300, 300))

    # img1 = np.zeros((400, 400, 3), dtype=np.uint8)  # 创建一个空白图像
    # img1[:, :] = (255, 0, 255)  # 给所有像素的b和r通道赋值
    # img2 = np.zeros((400, 400, 3), dtype=np.uint8)
    # img2[:, :] = (0, 255, 0)  # 给所有像素的g通道赋值

    dst1 = cv.bitwise_and(img1, img2)  # 图像的与操作
    dst2 = cv.bitwise_or(img1, img2)  # 图像的或操作
    dst3 = cv.bitwise_xor(img1, img2)  # 图像的异或操作
    dst4 = cv.bitwise_not(img1)  # 图像的非操作

    cv.imshow('img1', img1)
    cv.imshow('img2', img2)

    cv.imshow('bitwise_and', dst1)
    cv.imshow('bitwise_or', dst2)
    cv.imshow('bitwise_xor', dst3)
    cv.imshow('bitwise_not', dst4)

    cv.waitKey(0)
    cv.destroyAllWindows()


def channel_operation(image_path: str):
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    cv.namedWindow('input', cv.WINDOW_AUTOSIZE)
    cv.imshow('input', img)  # 彩色图像，3个通道，每个通道都是H×W。

    # 通道分离
    mv = cv.split(img)

    print('mv[0]', mv[0])  # 图像的b通道
    print('mv[1]', mv[1])  # 图像的g通道
    print('mv[2]', mv[2])  # 图像的r通道

    mv[0][:, :] = 255  # 给b通道上的所有像素值全部赋值为255
    # 通道合并
    result = cv.merge(mv)

    # 通道交换
    dst = np.zeros(img.shape, dtype=np.uint8)
    cv.mixChannels([img], [dst], fromTo=[2, 0, 1, 1, 0, 2])
    out = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 与上面的通道交换bgr->rgb结果类似，

    cv.imshow('bbb', img[:, :, 0])  # 显示第1个通道
    cv.imshow('ggg', img[:, :, 1])  # 显示第2个通道
    cv.imshow('rrr', img[:, :, 2])  # 显示第3个通道
    cv.imshow('result', result)
    cv.imshow('dst', dst)
    cv.imshow('out', out)

    cv.waitKey(0)
    cv.destroyAllWindows()


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
    cv.imshow('method 1', four_view)

    # 方法2：使用numpy的水平堆叠和竖直堆叠完成所有图像的堆叠，最后一起显示
    vs1 = np.hstack((img, invert))  # 水平堆叠
    vs2 = np.hstack((gaussianBlur, flip))  # 水平堆叠
    result = np.vstack((vs1, vs2))  # 竖直堆叠

    # 初略解决imshow()中文乱码显示的问题。不能完全解决，有些中文无法显示。这是python-opencv的弊端
    def zh_ch(string):
        return string.encode('gbk').decode(errors='ignore')

    cv.imshow(zh_ch('method 2'), result)
    cv.imwrite('images/method2.jpg', result)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    path1 = 'images/daiyutong.png'
    path2 = 'images/2.png'
    # color_operation(path)
    # bitwise_operation(path1, path2)
    # channel_operation(path1)
    image_display(path1)
