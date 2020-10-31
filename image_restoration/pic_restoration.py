# -*- coding: UTF-8 -*-    
# Author: LGD
# FileName: pic_restoration
# DateTime: 2020/10/18 17:11
# SoftWare: PyCharm
import numpy as np
import cv2 as cv

# OpenCV Utility Class for Mouse Handling
from pip._vendor.certifi.__main__ import args


class Sketcher:
    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.show()
        cv.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv.imshow(self.windowname, self.dests[0])
        cv.imshow(self.windowname + ": mask", self.dests[1])

    # onMouse function for Mouse Handling
    # 使用鼠标把掩膜画出来也就是第二幅图的白色线条
    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv.EVENT_LBUTTONUP:
            self.prev_pt = None

        if self.prev_pt and flags & cv.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv.line(dst, self.prev_pt, pt, color, 5)
            self.dirty = True
            self.prev_pt = pt
            self.show()


def main():
    print("Usage: python inpaint <image_path>")
    print("Keys: ")
    print("t - inpaint using FMM")
    print("n - inpaint using NS technique")
    print("r - reset the inpainting mask")
    print("ESC - exit")

    # Read image in color mode
    # 读取输入图片
    img = cv.imread("Lincoln.jpg", cv.IMREAD_COLOR)

    # If image is not read properly, return error
    if img is None:
        print('Failed to load image file: {}'.format(args["image"]))
        return

    # Create a copy of original image
    img_mask = img.copy()
    # Create a black copy of original image
    # Acts as a mask
    inpaintMask = np.zeros(img.shape[:2], np.uint8)
    # print(inpaintMask)
    # Create sketch using OpenCV Utility Class: Sketcher
    sketch = Sketcher('image', [img_mask, inpaintMask], lambda: ((255, 255, 255), 255))

    while True:
        ch = cv.waitKey()
        if ch == 27:
            break
        if ch == ord('t'):
            # Use Algorithm proposed by Alexendra Telea: Fast Marching Method (2004)
            res = cv.inpaint(src=img_mask, inpaintMask=inpaintMask, inpaintRadius=3, flags=cv.INPAINT_TELEA)
            cv.imshow('Inpaint Output using FMM', res)
        if ch == ord('n'):
            # Use Algorithm proposed by Bertalmio, Marcelo, Andrea L. Bertozzi, and Guillermo Sapiro:
            # Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting (2001)
            res = cv.inpaint(src=img_mask, inpaintMask=inpaintMask, inpaintRadius=3, flags=cv.INPAINT_NS)
            cv.imshow('Inpaint Output using NS Technique', res)
        if ch == ord('r'):
            img_mask[:] = img
            inpaintMask[:] = 0
            sketch.show()

    print('Completed')


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
