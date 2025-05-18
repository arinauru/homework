'''
# Нейронные сети
# - сверточные (конволюционные) нейронные сети (CNN) - компьютерное зрение, классификация изображений
# - рекуррентные (RNN) - распознования рукописного текста, обработка естественного языка
# - генеративные состязательные сети (GAN) - создание художественных и музыкальных произведений
# - многослойный перептрон - простейший тип НС

# Данная сеть способна сложить 2+2 или другие небольшие значения
w0 = 0.9907079
w1 = 1.0264927
w2 = 0.01417504
w3 = -0.8950311
w4 = 0.88046944
w5 = 0.7524377
w6 = 0.794296
w7 = 1.1687347
w8 = 0.2406084

b0 = -0.00070612
b1 = -0.06846002
b2 = -0.00055442
b3 = -0.00000929


def relu(x):
    return max(0, x)


def predict(x1, x2):
    h1 = (x1 * w0) + (x2 * w1) + b0
    h2 = (x1 * w2) + (x2 * w3) + b1
    h3 = (x1 * w4) + (x2 * w5) + b2

    y = (relu(h1) * w6) + (relu(h2) + w7) + (relu(h3) + w8) + b3
    return y


print(predict(2, 2))
print(predict(1.5, 1.5))

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

img_path = 'cat.png'
img = image.load_img(img_path, target_size=(224, 224))

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np

img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_processed = preprocess_input(img_batch)

model = ResNet50()

prediction = model.predict(img_processed)

from tensorflow.keras.applications.resnet50 import decode_predictions

print(decode_predictions(prediction, top=5)[0])

# plt.imshow(img)
# plt.show()
'''

import os

from keras import Model
from keras.src.applications.mobilenet import MobileNet

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.preprocessing import image

TRAIN_DATA_DIR = "train_data/"
VALIDATION_DATA_DIR = "val_data/"
TRAIN_SAMPLES = 500
VALIDATIONS_SAMPLES = 500
NUM_CLASSES = 2
IMG_WIDTH, IMG_HIGHT = 224, 224
BATCH_SIZE = 64

from tensorflow.keras.applications.mobilenet import preprocess_input

train_datagen = image.ImageDataGenerator(
    preprocessing_function=preprocess_input, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    zoom_range=0.2
)

val_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(TRAIN_DATA_DIR, target_size=(IMG_WIDTH, IMG_HIGHT),
                                                    batch_size=BATCH_SIZE, shuffle=True, seed=12345,
                                                    class_mode="categorical")

val_generator = train_datagen.flow_from_directory(VALIDATION_DATA_DIR, target_size=(IMG_WIDTH, IMG_HIGHT),
                                                  batch_size=BATCH_SIZE, shuffle=False,
                                                  class_mode="categorical")

from tensorflow.keras.layers import (Input, Flatten, Dense, Dropout, GlobalAveragePooling2D)


def model_maker():
    base_model = MobileNet(include_top=False, input_shape=(IMG_WIDTH, IMG_HIGHT, 3))
    for layer in base_model.layers[:]:
        layer.trainable = False

        input = Input(shape=(IMG_WIDTH, IMG_HIGHT, 3))
        custom_model = base_model(input)
        custom_model = GlobalAveragePooling2D()(custom_model)
        custom_model = Dense(64, activation="relu")(custom_model)
        custom_model = Dropout(0.5)(custom_model)
        prediction = Dense(NUM_CLASSES, activation="softmax")(custom_model)
        return Model(inputs=input, output=prediction)


from tensorflow.keras.optimizers import Adam

model = model_maker()
model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["acc"])

import math

num_steps = math.ceil(float(TRAIN_SAMPLES) / BATCH_SIZE)

model.fit(train_generator, steps_per_epoch=num_steps, epochs=10, validation_data=val_generator,
          validation_steps=num_steps)

print(val_generator.class_indices)

model.save("model.h5")

import os
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("model.h5")
img_path = "cat.png"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

from tensorflow.keras.applications.mobilenet import preprocess_input

img_processed = preprocess_input(img_batch)

prediction = model.predict(img_processed)
print(prediction)

# Дз

import plotly.express as px

df = px.data.iris()

fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width', color='species')

fig.show()


