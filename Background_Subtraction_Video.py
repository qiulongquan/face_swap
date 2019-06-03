# 这个是利用 直方图反向投影的方式将摄像头图像和投影后的结果同时显示出来。

from __future__ import print_function
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

# hardcode定义   camera_device = 0当前摄像头
camera_device = 0
#-- 2. Read the video stream
capture = cv.VideoCapture(camera_device)
if not capture.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask = backSub.apply(frame)
    # thickness 组成矩形的线条的粗细程度。取负值时（如CV_FILLED）函数绘制填充了色彩的矩形。
    cv.rectangle(frame, (10, 2), (100, 20), (255, 255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)

    # 使用下面的设定暂停程序 ，视频可以按空格键 一帧一帧的移动
    # Opencv中的cvWaitkey函数的用法
    # 返回值为int型，函数的参数为int型，当delay小于等于0的时候，
    # 如果没有键盘触发，则一直等待，
    # 此时的返回值为 - 1，否则返回值为键盘按下的码字；
    # 当delay大于0时，如果没有键盘的的触发，则等待delay的时间，
    # 此时的返回值是 - 1，否则返回值为键盘按下的码字。

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break