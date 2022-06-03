# General imports
import sys
import os
# Remove this if TensorFlow debugging information is important
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

from keras.preprocessing import image

MODEL_PATH = "model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Defines predict function
def model_predict(img_path, model):
    img = image.image_utils.load_img(img_path, target_size=(150,150))
    x = image.image_utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x =  np.vstack([x])

    preds = model.predict(x)
    if preds[0] > 0.5:
        return "Dog"
    else:
        return "Cat"


print("Enter image location:")
image_path = input()

# Checks if image is a cat or dog
result = model_predict(image_path, model)
print(image_path, "is likely a", result)