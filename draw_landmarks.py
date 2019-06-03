# 这个是显示人脸的识别图框，同时标注出68个识别点的程序，程序中指定一张图片，然后进行分析。

import cv2
import dlib

# 加载面部检测器
detector = dlib.get_frontal_face_detector()
# 加载训练模型并获取面部特征提取器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 以 RGB 模式读入图像
im = cv2.imread('6.jpg', cv2.IMREAD_COLOR)

# 使用检测器检测人脸
rects = detector(im, 1)
for index, face in enumerate(rects):
        print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(), face.bottom()))

        # 在图片中标注人脸，并显示
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(im, (left, top), (right, bottom), (0, 255, 0), 3)
        # cv2.namedWindow('1.jpg', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('1.jpg', im)


print(rects)
print(len(rects))
# 使用特征提取器获取面部特征点
for n in range(len(rects)):
    print(n)
    print(rects[n])
    l = [(p.x, p.y) for p in predictor(im, rects[n]).parts()]
    # 遍历面部特征点并绘制出来
    for cnt, p in enumerate(l):
        # cricle_size代表的是黄色圆圈的尺寸
        cricle_size = 1
        cv2.circle(im, p, cricle_size, (0, 255, 255), 2)

        # str_size代表的是文字的字体尺寸
        str_size = 0.25
        cv2.putText(im, str(cnt), (p[0]+5, p[1]-5), 0, str_size, color=(0, 0, 255))
# 保存图像
cv2.imwrite('landmarks.jpg', im)
cv2.waitKey(0)
