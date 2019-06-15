# 原文：https: // blog.csdn.net / lukaslong / article / details / 86649453
# 输入指定的一段视频，可以检测pb训练集结果

import tensorflow as tf
import cv2
import os
import time
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

videofile = '/Users/qiulongquan/qlq.mp4'
cap = cv2.VideoCapture(videofile)
MODEL_NUM_CLASSES = 3
MODEL_LABEL_MAP = '/Users/qiulongquan/face_swap/video_face_swap/data/qiulongquan.pbtxt'
MODEL_PB = '/Users/qiulongquan/face_swap/video_face_swap/qiulongquan/frozen_inference_graph.pb'

# read graph model
with tf.gfile.GFile(MODEL_PB, 'rb') as fd:
    _graph = tf.GraphDef()
    _graph.ParseFromString(fd.read())
    tf.import_graph_def(_graph, name='')

# get the default graph model
detection_graph = tf.get_default_graph()

# read labelmap
label_map = label_map_util.load_labelmap(MODEL_LABEL_MAP)
categories = label_map_util.convert_label_map_to_categories(label_map, MODEL_NUM_CLASSES)
category_index = label_map_util.create_category_index(categories)

with tf.Session(graph=detection_graph) as sess:
    while (cap.isOpened()):
        ret, frame = cap.read()
        frame_np_expanded = np.expand_dims(frame, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        t1 = time.time()
        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], \
                                                            feed_dict={image_tensor: frame_np_expanded})

        vis_util.visualize_boxes_and_labels_on_image_array(frame, np.squeeze(boxes),
                                                           np.squeeze(classes).astype(np.int32), np.squeeze(scores),
                                                           category_index,
                                                           use_normalized_coordinates=True, line_thickness=3)
        t2 = time.time()
        print('FPS:',1 / (t2 - t1))
        cv2.imshow('MobilenetTF', frame)
        if cv2.waitKey(1) & 0xff == 27:
            break
    cap.release()



