# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:55:43 2018
@author: Xiang Guo
"""
# Imports
import time

start = time.time()
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
print(tf.__version__)
if tf.__version__ < '1.12.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.12.* or later!')

os.chdir('/Users/qiulongquan/models/research/object_detection')

# Env setup
# This is needed to display the images.
# %matplotlib inline

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Object detection imports
from utils import label_map_util

from utils import visualization_utils as vis_util

# Model preparation
# What model to download.
MODEL_NAME = 'qiulongquan'  # [30,21]  best
# MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'            #[42,24]
# MODEL_NAME = 'faster_rcnn_inception_v2_coco_2017_11_08'         #[58,28]
# MODEL_NAME = 'faster_rcnn_resnet50_coco_2017_11_08'     #[89,30]
# MODEL_NAME = 'faster_rcnn_resnet50_lowproposals_coco_2017_11_08'   #[64, ]
# MODEL_NAME = 'rfcn_resnet101_coco_2017_11_08'    #[106,32]

# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'qiulongquan.pbtxt')

NUM_CLASSES = 1

'''
#Download Model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
'''

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    # Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Detection
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 10)]

# Size, in inches, of the output images.
IMAGE_SIZE = (30, 20)

output_path = '/Users/t-lqiu/models/research/object_detection/output_folder'

vidcap = cv2.VideoCapture(0)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        count = 0
        while (True):
            ret, image_np = vidcap.read()

            if ret == True:

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    # 这个是阀值，超过0.5以后才会显示框框，小于0.5的不显示。
                    min_score_thresh=0.5,
                    # 当True的时候不显示分数。
                    skip_scores=False,
                    # 当True的时候不显示标签。
                    skip_labels=False,
                    # 当True的时候不显示追踪id号。
                    skip_track_ids=False,
                    use_normalized_coordinates=True,
                    # 显示框框的粗细尺寸。
                    line_thickness=4)

                # 这个是设定摄像头显示画面的大小
                cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))

                c = cv2.waitKey(1)
                if c == 27:  # ESCを押してウィンドウを閉じる
                    cv2.destroyAllWindows()
                    break
                if c == 32:  # spaceで保存
                    count += 1
                    cv2.imwrite(output_path+'/filename%03.f' % (count) + '.jpg', image_np)  # 001~連番で保存
                    print('save done')

            # Break the loop
            else:
                break

end = time.time()
print("Execution Time: ", end - start)
