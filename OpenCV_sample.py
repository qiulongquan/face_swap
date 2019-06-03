# 通过指定的cascade类文件，检测出指定目标faces，eyes。
# faces获取的值转化为 x,y,w,h 坐标，然后画出rectangle。
# 然后再进一步画出eyes的rectangle。
# 最后保存img
# 可以通过本程序的例子，把多个detectMultiScale检测放到一张图片里面进行。
# 同理也可以把多个detectMultiScale检测放到一个视频里面进行。

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('./models/haarcascade_mcs_nose.xml')

img = cv2.imread('1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3,minSize=(50, 50))
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=3,minSize=(80, 80))
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

# 如果只是显示，可以调用imshow方法进行显示。
cv2.imshow('img',img)
# 認識結果の保存  如果需要保存可以调用imwrite方法，写入输出地址output_path
# cv2.imwrite(output_path, image)
cv2.waitKey(0)
cv2.destroyAllWindows()