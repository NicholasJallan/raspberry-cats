import tensorflow as tf
from tensorflow.python.client import device_lib
from keras import backend as K

# print(device_lib.list_local_devices())
#
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print('Checking')
K.tensorflow_backend._get_available_gpus()
