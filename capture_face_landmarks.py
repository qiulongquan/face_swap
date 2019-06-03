# 这个是视频实时人脸识别程序，可以识别视频中的多个人脸同时显示出嘴部的动作，可以截图保存。
# https://qiita.com/ufoo68/items/b1379b40ae6e63ed3c79
# 按空格键 截图
# 按ESC键 退出

#! /usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import dlib
import numpy as np
import imutils
from imutils import face_utils


def face_shape_detector_dlib(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # frontal_face_detectorクラスは矩形, スコア, サブ検出器の結果を返す
    # 下の行が主に顔検出を行う処理です。
    dets, scores, idx = detector.run(img_rgb, 0)
    if len(dets) > 0:
        for i, rect in enumerate(dets):
            shape = predictor(img_rgb, rect)
            shape = face_utils.shape_to_np(shape)
            clone = img.copy()
            cv2.putText(clone, "mouth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # landmarkを画像に書き込む
            for (x, y) in shape[48:68]:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
            # shapeで指定した個所の切り取り画像(ROI)を取得
            (x, y, w, h) = cv2.boundingRect(np.array([shape[48:68]])) #口の部位のみ切り出し
            roi = img[y:y + h, x:x + w]
            roi = cv2.resize(roi,(100,100))
        return clone, roi
    else:
        return img, None


def main():
    # 这个地方导入的是脸部识别的dat文件，如果导入其他的识别dat可以识别其他的东西。
    # 程序本身不需要过多修改。
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    global predictor
    predictor = dlib.shape_predictor(predictor_path)
    global detector
    # 下の行が主に顔検出を行う処理です。
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=500)
        frame, roi = face_shape_detector_dlib(frame)
        cv2.imshow('img', frame)
        if roi is not None :
            cv2.imshow('roi', roi)
        else:
            cv2.destroyWindow('roi')
        c = cv2.waitKey(1)
        if c == 27:#ESCを押してウィンドウを閉じる
            break
        if c == 32:#spaceで保存
            count += 1
            cv2.imwrite('./filename%03.f'%(count)+'.jpg', roi) #001~連番で保存
            print('save done')
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()



