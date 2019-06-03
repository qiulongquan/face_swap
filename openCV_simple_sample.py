# 图片识别简单版本，指定图片，指定cascade类文件。

# -*- coding:utf-8 -*-
import numpy
import cv2
cascade_path = './models/haarcascade_eye.xml'
image_path = '1.jpg'
color = (255, 0, 0) #青
#ファイル読み込み
image = cv2.imread(image_path)
#グレースケール変換
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#カスケード分類器の特徴量を取得する
cascade = cv2.CascadeClassifier(cascade_path)
#物体認識（顔認識）の実行
facerect = cascade.detectMultiScale(
                image_gray, scaleFactor=1.3, minNeighbors=2, minSize=(50, 50))
if len(facerect) > 0:
    #検出した顔を囲む矩形の作成
    for rect in facerect:
        cv2.rectangle(image, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
    #認識結果の保存
    cv2.imwrite("detected.jpg", image)
