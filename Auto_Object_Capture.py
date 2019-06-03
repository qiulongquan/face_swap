# 这个是检测摄像头输入的物体的移动程序
# https://docs.opencv.org/master/d7/d00/tutorial_meanshift.html

# 利用直方图，反向投影图，检测摄像头中的移动目标
# 【opencv学习笔记】027之直方图反向投影 - calcBackProject函数详解
# https://blog.csdn.net/shuiyixin/article/details/80331839

# Computer Vision Face Tracking for Use in a Perceptual User Interface

# 问题
# 可以跟踪捕捉了，但是系统不知道什么才是需要捕捉的，需要让系统知道要捕捉什么

import numpy as np
import cv2 as cv
import argparse

# hardcode定义   camera_device = 0当前摄像头
camera_device = 0
#-- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

# take first frame of the video
ret, frame = cap.read()
# setup initial location of window
# 这个是设定视频文件中蓝色的追踪窗口的大小
x, y, w, h = 300, 200, 100, 50  # simply hardcoded the values
track_window = (x, y, w, h)
# set up the ROI(Region of Interest) for tracking
roi = frame[y:y+h, x:x+w]
# hsv_roi 这个是跟踪对象窗口
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
# 介绍inRange方法的使用
# http://pynote.hatenablog.com/entry/opencv-inrange
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
# opencv学习(三十七)之图像直方图计算calcHist()
# https://blog.csdn.net/keith_bb/article/details/56680997
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])

# 归一化处理normalize()函数
# normalize(grayHist,grayHist,0,histImage.rows,NORM_MINMAX,-1,Mat());
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
# 根据测试，pt为10的时候跟踪窗口移动比较稳定，pt为1的时候移动跳转比较快
iteration = 100
pt = 10
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, iteration,pt)

while(True):
    ret, frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # apply meanshift to get the new location
        ret, track_window = cv.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x, y, w, h = track_window
        img2 = cv.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
        cv.imshow('img2', img2)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break
