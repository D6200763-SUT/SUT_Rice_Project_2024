import sys
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("TensorFlow:", tf.__version__)

print("\nGPU devices:")
print(tf.config.list_physical_devices("GPU"))

print("\nKeras test:")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.summary()

print("\nRESULT: ENV OK")