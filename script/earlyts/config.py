# coding=utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf

# print(tf.sysconfig.get_build_info())

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.enable_eager_execution()

# tf.config.gpu.set_per_process_memory_growth()


data_root_dir = os.path.join(os.path.dirname(__file__), "../data")

