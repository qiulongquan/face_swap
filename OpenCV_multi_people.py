# 这个程序是调用haarcascades类库，实时识别视频中的人物还有眼睛的程序。
# 支持多人同时识别，识别的是眼镜和脸部。

from __future__ import print_function
import cv2 as cv
import argparse


def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    # 可以支持图像中有多个人，分别画出每个人的脸部和眼部
    for (x,y,w,h) in faces:
        # " / "就表示浮点数除法，返回浮点结果;" // "表示整数除法。
        # 比如 50/2  结果25.0
        # 50 // 2 结果25
        center = (x + w//2, y + h//2)

        # 椭圆函数如下，主要介绍下官方提供的cv：：ellipse函数。
        # cv：：ellipse（Mat img，Point（x, y）, Size(a, b), angle, 0, 360, Scalar（，，）, thickness, lineType);
        # Point(x, y)
        # 是椭圆的中心点
        # Size(a, b)
        # 是椭圆的长轴和短轴
        # angle是椭圆的倾斜角度
        # 0, 360
        # 是椭圆的有效显示角度，0 - 360
        # 则整个椭圆都会显示。0 - 180
        # 则会显示一半椭圆。
        # Scalar(b, g, r)
        # 是椭圆的颜色
        # thickness是线的宽度
        # lineType是线性.
        #
        # 【OpenCV3】几何图形（直线、矩形、圆、椭圆、多边形等）绘制
        # https://blog.csdn.net/guduruyu/article/details/68490206

        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        print('x=%s,y=%s,w=%s,h=%s'%(x,y,w,h))
        faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            # 这个是圆心坐标
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            # radius 这个是半径
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)
    cv.imshow('Capture - Face detection', frame)

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera device number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
camera_device = args.camera
#-- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    # 如果按了ESC键，退出程序。
    # // 矢印キー
    # const int CV_WAITKEY_CURSORKEY_TOP    = 2490368;
    # const int CV_WAITKEY_CURSORKEY_BOTTOM = 2621440;
    # const int CV_WAITKEY_CURSORKEY_RIGHT  = 2555904;
    # const int CV_WAITKEY_CURSORKEY_LEFT   = 2424832;
    # // エンターキーとか
    # const int CV_WAITKEY_ENTER            = 13;
    # const int CV_WAITKEY_ESC              = 27;
    # const int CV_WAITKEY_SPACE            = 32;
    # const int CV_WAITKEY_TAB              = 9;
    if cv.waitKey(10) == 27:
        break



