# 这个是通过命令行指定输入图片文件名，然后指定识别的部位，包括（眼睛，鼻子等）
# 最后输出到outputs文件夹中。
# 训练好的xml文件保存在models文件里面。
# 执行命令
# python openCVWithParameter01.py --image_file "1.jpg" --cascade "nose" --min 50
# 文档原文
# https://qiita.com/FukuharaYohei/items/116932920c99a5b73b32


import cv2
import argparse

# 基本的なモデルパラメータ
FLAGS = None

# 学習済モデルの種類
cascade = ["default","alt","alt2","upperbody","profile","nose"]

# 直接実行されている場合に通る(importされて実行時は通らない)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cascade",
        type=str,
        default="default",
        choices=cascade,
        help="cascade file."
  )
    parser.add_argument(
        "--image_file",
        type=str,
        default="cut_source0.jpg",
        help="image file."
  )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.3,
        help="scaleFactor value of detectMultiScale."
  )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=2,
        help="minNeighbors value of detectMultiScale."
  )
    parser.add_argument(
        "--min",
        type=int,
        default=50,
        help="minSize value of detectMultiScale."
  )

# パラメータ取得と実行
FLAGS, unparsed = parser.parse_known_args()

# 分類器ディレクトリ(以下から取得)
# https://github.com/opencv/opencv/blob/master/data/haarcascades/
# https://github.com/opencv/opencv_contrib/blob/master/modules/face/data/cascades/

if   FLAGS.cascade == cascade[0]:#"default":
    cascade_path = "./models/haarcascade_frontalface_default.xml"
elif FLAGS.cascade == cascade[1]:#"alt":
    cascade_path = "./models/haarcascade_frontalface_alt.xml"
elif FLAGS.cascade == cascade[2]:#"alt2":
    cascade_path = "./models/haarcascade_frontalface_alt2.xml"
elif FLAGS.cascade == cascade[3]:#"tree":
    cascade_path = "./models/haarcascade_mcs_upperbody.xml"
elif FLAGS.cascade == cascade[4]:#"profile":
    cascade_path = "./models/haarcascade_profileface.xml"
elif FLAGS.cascade == cascade[5]:#"nose":
    cascade_path = "./models/haarcascade_mcs_nose.xml"

# 使用ファイルと入出力ディレクトリ
image_path  =  FLAGS.image_file
output_path = "./outputs/" + FLAGS.image_file

# ディレクトリ確認用(うまく行かなかった時用)
#import os
#print(os.path.exists(image_path))

#ファイル読み込み
image = cv2.imread(image_path)

#グレースケール変換
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#カスケード分類器の特徴量を取得する
cascade = cv2.CascadeClassifier(cascade_path)

#物体認識（顔認識）の実行
#image - CV_8U 型の行列．ここに格納されている画像中から物体が検出されます
#objects - 矩形を要素とするベクトル．それぞれの矩形は，検出した物体を含みます
#scaleFactor - 各画像スケールにおける縮小量を表します
#minNeighbors - 物体候補となる矩形は，最低でもこの数だけの近傍矩形を含む必要があります
#flags - このパラメータは，新しいカスケードでは利用されません．古いカスケードに対しては，cvHaarDetectObjects 関数の場合と同じ意味を持ちます
#minSize - 物体が取り得る最小サイズ．これよりも小さい物体は無視されます
facerect = cascade.detectMultiScale(image_gray, scaleFactor=FLAGS.scale, minNeighbors=FLAGS.neighbors, minSize=(FLAGS.min, FLAGS.min))

#print(facerect)

color = (255, 255, 255) #白

# 検出した場合
if len(facerect) > 0:

    #検出した顔を囲む矩形の作成
    for rect in facerect:
        cv2.rectangle(image, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)

    #認識結果の保存
    cv2.imwrite(output_path, image)