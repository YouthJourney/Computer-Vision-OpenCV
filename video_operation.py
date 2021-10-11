# -*-coding:utf-8-*-
"""
File Name: video_operation.py
Program IDE: PyCharm
Date: 21:10
Create File By Author: Hong
"""
import cv2 as cv
import numpy as np


def read_video(video_path: str):
    cap = cv.VideoCapture(video_path)

    # 获取视频帧的宽和高
    w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    # 获取视频总帧数和fps
    count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv.CAP_PROP_FPS)

    print('w: {}, h: {}, count: {}, fps: {}'.format(w, h, count, fps))

    # 视频保存
    fourcc = cv.VideoWriter_fourcc('P', 'I', 'M', '1')
    # fourcc = cv.VideoWriter_fourcc(*'XVID')
    # 视频编码格式
    fourcc = cv.VideoWriter_fourcc(*'mp4v')

    out = cv.VideoWriter('images/video_save.mp4', fourcc, fps, (int(w), int(h)), True)

    while cap.isOpened():
        ret, frame = cap.read()
        # 调用本地摄像头时，需要左右翻转一下，若是视频文件则不需要翻转
        # frame = cv.flip(frame, 1)
        if not ret:
            break
        cv.imshow('frame', frame)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # 写入视频
        out.write(hsv)
        cv.imshow('hsv', hsv)

        c = cv.waitKey(1)
        if c == 27 or 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    path = 'images/guangzhou.mp4'
    read_video(path)
