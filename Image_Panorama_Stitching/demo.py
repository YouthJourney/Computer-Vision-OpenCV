# -*- encoding: utf-8 -*-
"""
@Date ： 2020/10/5 11:05
@Author ： LGD
@File ：demo.py
@IDE ：PyCharm
"""
import cv2


def sift_kp(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT.create()
    kp, des = sift.detectAndCompute(gray_image, None)
    kp_image = cv2.drawKeypoints(gray_image, kp, None)
    return kp_image, kp, des

