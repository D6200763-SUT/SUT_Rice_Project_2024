import tensorflow as tf
import time
import tensorflow as tf

print("TF:", tf.__version__)
print("CUDA build:", tf.test.is_built_with_cuda())
print("GPU:", tf.config.list_physical_devices("GPU"))