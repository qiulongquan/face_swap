# tensorflow has been installed complete and then state test program.
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# ---------------------
# Sample Output
#
# [name: "/cpu:0"
# device_type: "CPU"
# memory_limit: 268435456
# locality
# {}
# incarnation: 4402277519343584096,
# name: "/gpu:0"
# device_type: "GPU"
# memory_limit: 6772842168
# locality
# {bus_id: 1}
# incarnation: 7471795903849088328
# physical_device_desc: "device: 0, name: GeForce GTX 1070, pci bus id: 0000:05:00.0"]
# ---------------------
