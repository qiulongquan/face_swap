# 原文：https: // blog.csdn.net / qq_16564093 / article / details / 78996563

import tensorflow as tf
import pathlib2 as pathlib

# converter = tf.contrib.lite.TocoConverter.from_frozen_graph('model.pb',["input_image"],["result"], input_shapes={"input_image":[1,626,361,3]})   #Python 2.7.6版本,但测试量化后模型大小不会变小
converter = tf.lite.TFLiteConverter.from_frozen_graph('frozen_inference_graph.pb', ["input_image"], ["result"], input_shapes={"input_image": [1, 626, 361, 3]})  # python3.4.3--nightly版本,测试量化后模型大小会变小

converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8

converter.quantized_input_stats = {"input_image": (127, 2.)}

converter.default_ranges_stats = (0, 6)

tflite_quantized_model = converter.convert()

open("quantized_model.tflite", "wb").write(tflite_quantized_model)



