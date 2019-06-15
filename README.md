```

安装 dlib
dlib 是一个基于 C++ 编写的扩展库，包含有许多常用的机器学习算法以及图像处理函数。
并且还支持大量的数值计算，如矩阵、大整数随机运算等。
但是在编译安装 dlib 之前我们还需要先给系统装上各种依赖环境，步骤如下。

参考：dlib 官网
http://dlib.net


安装 OpenCV
OpenCV 是一款功能强大的跨平台计算机视觉开源库，可以用于解决人机交互、物体检测、人脸识别等领域的问题。
库本身是采用 C++ 编写的，但是同时也对 Python, Java, C# 等语言提供接口支持。

https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html

安装方法
pip3 install opencv-python


安装 docopt
docopt 是 Python 的一个第三方参数解析库，可以根据使用者提供的文档描述自动生成解析器。因此使用者可以用它来定义交互参数与解析参数。

所以 docopt(__doc__, version=__version__) 函数能自动根据三个双引号中的文档内容（存储于 __doc__ 中）生成命令行解析器，并将解析结果以字典对象返回。
所以使用 docopt 进行命令行解析，我们唯一需要做好的事情就是编写好帮助文档，然后其它一切解析的问题都可以甩手给 docopt 自动解决。
安装 docopt
$ sudo pip3 install docopt
参考：http://docopt.org/


-----------------------------------------------------------------------

dlib 很准确地检测出来了下巴、嘴巴、鼻子、眼睛和眉毛这些面部特征点。并且每个部位都被分配固定索引的点进行标注，
范围如下表，一共是 68 个标记点。

部位	    索引
下巴	    0~16
左眉毛	    17~21
右眉毛	    22~26
鼻子	    27~35
左眼睛	    36~41
右眼睛	    42~47
嘴巴	    48~67


参考文档
https://www.w3cschool.cn/opencv/opencv-ful52dkf.html
https://qiita.com/ufoo68/items/b1379b40ae6e63ed3c79
https://blog.csdn.net/hongbin_xu/article/details/78347484


程序中使用到的 shape_predictor_68_face_landmarks.dat 是 dlib 官方提供的模型数据，
有了这个模型之后我们就不需要自己再耗费时间去训练模型，直接拿来使用即可。
下载方式：
wget https://labfile.oss.aliyuncs.com/courses/686/shape_predictor_68_face_landmarks.dat


tutorials
https://docs.opencv.org/master/d9/df8/tutorial_root.html


Ever wondered how your digital camera detects peoples and faces? Look here to find out!
https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html


Look here in order to find use on your video stream algorithms like: motion extraction, 
feature tracking and foreground extractions.
https://docs.opencv.org/master/da/dd0/tutorial_table_of_content_video.html


This section contains tutorials about how to read/save your video files.
https://docs.opencv.org/master/df/d2c/tutorial_table_of_content_videoio.html


各种cascade类文件的效果展示。包括 嘴巴 鼻子 眼睛 身体的识别.
https://symfoware.blog.fc2.com/blog-entry-1556.html


# 从下面的地址获获取opencv的训练完成后的类文件，包括鼻子 脸部 嘴巴等识别文件。
# https://github.com/opencv/opencv/blob/master/data/haarcascades/
# https://github.com/opencv/opencv_contrib/blob/master/modules/face/data/cascades/


----------------------------------------------------------
机器深度学习的时候需要大量的图片，手工制作非常麻烦。
通过这个教程可以很快的用一张图片自动生成大量的图片，包括方向改变，颜色改变，尺寸改变等。
水増し　掺水分 作假
画像の水増し方法をTensorFlowのコードから学ぶ
https://qiita.com/Hironsan/items/e20d0c01c95cb2e08b94


----------------------------------------------------------
openCVで効率的に大量画像を顔検出するためのtips
https://qiita.com/FukuharaYohei/items/1e15d79c87f1de513eac
重要参数设置数值
scaleFactor=1.2, minNeighbors=2, minSize=[50,50]が誤検知率が低くて優秀でした。


----------------------------------------------------------
优化分析 提高准确率降低分析时间降低系统负荷

物体検出（detectMultiScale）をパラメータを変えて試してみる（scaleFactor編）
http://workpiles.com/2015/04/opencv-detectmultiscale-scalefactor/

物体検出（detectMultiScale）をパラメータを変えて試してみる（minNeighbors編）
http://workpiles.com/2015/04/opencv-detectmultiscale-minneighbors/


----------------------------------------------------------
需要完成的部分
分析 支持多人同时识别的代码 OpenCV_multi_people.py   ok
实现能够支持手势gesture的实现方法
实现读取视频文件，然后分析识别视频中的人物的脸部和眼睛
实现追踪特征点，根据指定的物体，追踪物体移动，并在图像中标注出来   ok
如何识别特定的物体，或者是指定的物体  特定人脸识别   ok
实现换脸的特效，检测2张脸，然后互换眼睛，鼻子，嘴巴，眉毛等特征
readme文档中没有实现的代码   ok
了解使用GCP google cloud plateform 来实现快速训练


Tensorflow object detection API 搭建属于自己的物体识别模型（1）——环境搭建与测试
https://blog.csdn.net/dy_guox/article/details/79081499
在实际使用前需要进行安装  installation instructions 安装下面的官方操作方法安装
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
（1）如果有下面的错误
ModuleNotFoundError: No module named 'tensorflow.python.saved_model.model_utils'
参考这个url解决问题
pip uninstall tensorflow_estimator
pip install tensorflow_estimator
https://stackoverflow.com/questions/55082483/why-i-cannot-import-tensorflow-contrib-i-get-an-error-of-no-module-named-tensor

（2）如果出现 ModuleNotFoundError: No module named 'object_detection'这样的模块没有找到的错误
参考这个url解决问题
选择File—>settings—>project:pythonWork—>project structure  调整root路径为models/research 这样就可以找到object_detection模块了
https://blog.csdn.net/qq_20367813/article/details/79608108

（3）如果出现Protobuf没有安装，参照下面的方法安装，选择Mac版本下载
Protobuf 安装与配置
在 https://github.com/google/protobuf/releases 

创建软连接到/usr/local/bin
ln -s /Users/qiulongquan/protoc-3/bin/protoc /usr/local/bin

把include文件夹拷贝到/usr/local里面去
cp -r include /usr/local

在models\research\目录下打开命令行窗口，输入：
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.


（更新视频教程）Tensorflow object detection API 搭建属于自己的物体识别模型（2）——训练并使用自己的模型
https://blog.csdn.net/dy_guox/article/details/79111949
这个是对应的视频  重点参考看
https://www.bilibili.com/video/av21539370/?p=1


tensorflow进行人脸识别已经可以使用，准确率还是很高的，可以侧脸识别。
如果有一部分脸部被遮挡，匹配率会下降，这说明程序按照预想进行了视频分析。

----------------------------关于参数和图片数量  重点-----------------------------------

1.图片要150张以上，可以确保比较高的准确率。图片越多训练时间越长。同一个目标各种不同方向，光照，姿势的图片都应该加入。
2.训练的时候 batch_size（一次批处理的大小）选择5-10都可以，越大越容易收敛归一，但是内存容量要更多。
3.训练step选择25000步基本就可以了，主要看lose损失函数是不是已经下降并且平稳了。
4.150张图片25000步大概20多个小时可以完成。大概2.8秒一步。


------------------------------------------------------------------------------------

训练完成先冻结模型，然后生成TF lift文件，轻量化文件可以在手机端使用。pb文件不能在手机端正常调用。
【TF lite】从tensorflow模型训练到lite模型移植
https://blog.csdn.net/lukaslong/article/details/86649453

Tensorflow Lite之编译生成tflite文件
https://blog.csdn.net/qq_16564093/article/details/78996563

这个是查看pb文件内容的网站
https://lutzroeder.github.io/netron/


Tensorflow object detection API 搭建属于自己的物体识别模型——常见问题汇总
https://blog.csdn.net/dy_guox/article/details/80139981


# 下载模型的名字
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
 
# Path to frozen detection graph. This is the actual model that is used for the object detection. 
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
 
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
 
NUM_CLASSES = 90

官方的各种模型
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
github上有对应官方的各种模型（地址摸我），这些都是基于不用的数据集事先训练好的模型，下载好以后就可以直接调用。
下载的文件以 '.tar.gz'结尾。'PATH_TO_CKPT'为‘.pb’文件的目录，'.pb'文件是训练好的模型（frozen detection graph），即用来预测时使用的模型。
‘PATH_TO_LABELS’为标签文件，记录了哪些标签需要识别，'NUM_CLASSES'为类别的数目，根据实际需要修改。



动态手势识别实战
https://blog.csdn.net/wonderseen/article/details/78341932


手把手教你如何安装Tensorflow（Windows和Linux两种版本）   ok
不安装gpu部分，只使用cpu来运算
https://blog.csdn.net/Cs_hnu_scw/article/details/79695347


windows下安装配置cudn和cudnn    ok
配置完成了，但是整体测试还是有问题，暂时取消掉，不安装gpu部分，只使用cpu来运算
https://www.jianshu.com/p/9bdeb033e765
----------------------------------------------------------

分析 opencv的实现原理    ok
http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html#face-detection
https://qiita.com/FukuharaYohei/items/ec6dce7cc5ea21a51a82
http://www.ime.info.hiroshima-cu.ac.jp/%7Ehiura/lec/ime/04.pdf
https://memo.sugyan.com/entry/20151203/1449137219
https://blog.mudatobunka.org/entry/2016/10/03/014520
下面这个地址是网页版本face-detector 输入一个图像的url会得到这个图片里面人物的脸部和眼睛的标注。
https://face-detector.herokuapp.com



下面这篇文章简单的介绍了 
2001年のViola & Johns超高速＆超正確な検出アルゴリズム的实现方法。
简单来说 是通过 脸部主要是眼睛和鼻子部分的光亮度差别来检测是不是人脸，这样的方法可以很快的检测一张图片中是不是有人脸
一般通过10次左右可以把候选对象降低到千分之一， 20次可以降低到百万分之一。
サルでもわかる顔検出の原理
https://cru.hatenadiary.org/entry/20100613/1276436975#f-401898cd



【OpenCV笔记 06】OpenCV中绘制基本几何图形【矩形rectangle()、椭圆ellipse() 、圆circle() 】
https://blog.csdn.net/sinat_34707539/article/details/51912610

一个成熟的人脸识别系统通常由
人脸检测、
人脸最优照片选取、
人脸对齐、
特征提取、
特征比对几个模块组成。

拒识和误识二者不可兼得，
所以评价人脸识别算法时常用的指标是误识率小于某个值时（例如0.1%）的拒识率。
如何理解误识率（FAR）拒识率（FRR），TPR,FPR以及ROC曲线
https://blog.csdn.net/colourful_sky/article/details/72830640

人脸识别技术相关知识原理
http://baijiahao.baidu.com/s?id=1598498964557818156&wfr=spider&for=pc   一般，不是很清晰
深入浅出人脸识别原理
回顾下流程：
1、使用HOG找出图片中所有人脸的位置。
2、计算出人脸的68个特征点并适当的调整人脸位置，对齐人脸。
3、把上一步得到的面部图像放入神经网络，得到128个特征测量值，并保存它们。
4、与我们以前保存过的测量值一并计算欧氏距离，得到欧氏距离值，比较数值大小，即可得到是否同一个人。
https://blog.csdn.net/LEON1741/article/details/81358974    普通理论 没有实现方法


1.从识别的准确率考虑，通过实验发现要保证人脸识别的准确率，人脸照片中双眼瞳距之间要大于80个像素，
这就意味着在选择摄像头时要充分考虑焦距和分辨率两个指标。
2.人脸大小在满足比对要求时控制在150个像素是比较合适的。
3.同时对图片质量的压缩也是非常必要的。经我们测试，75%的jpeg压缩率对人脸识别的性能影响可以忽略，却可以节约几倍的带宽资源。
除了常用的jpeg、png图像编码，苏宁还使用WebP图像压缩格式，可以使带宽资源的占用进一步降低20%~30%。


视频图像的一个非常重要的特性是它的时间连续性，以及由此产生的人脸信息的不确定性。在人脸跟踪和识别中利用时间信息是视频人脸识别算法和基于静态图像的人脸识别算法的最大区别。
目前这类算法大致可分为两类：
1、 跟踪 - 然后 - 识别，这类方法首先检测出人脸，然后跟踪人脸特征随时间的变化。当捕捉到一帧符合一定标准（大小，姿势）的图像时，用基于静态图像的人脸识别算法进行识别。
这类方法中跟踪和识别是单独进行的，时间信息只在跟踪阶段用到。识别还是采用基于静态图像的方法，没用到时间信息。
2、 跟踪 - 且 - 识别，这类方法中，人脸跟踪和识别是同时进行的，时间信息在跟踪阶段和识别阶段都用到。

参考
瓦希德·卡奇米（Vahid Kazemi）和约瑟菲娜·沙利文（Josephine Sullivan）在 2014 年发明的方法。
这一算法的基本思路是找到68个人脸上普遍存在的点（称为特征点， landmark）。

保罗·比奥拉（Paul Viola）和迈克尔·琼斯（Michael Jones）在2000年发明了一种能够快速在廉价相机上运行的人脸检测方法，人脸检测在相机上的应用才成为主流。
然而现在我们有更可靠的解决方案HOG（Histogram of Oriented Gradients）方向梯度直方图，一种能够检测物体轮廓的算法。
首先把图片灰度化，因为颜色信息对于人脸检测而言没什么用。

Anaconda install archive  Anaconda的安装URL
https://repo.continuum.io/archive/
```