#%%

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers


#%%

labels = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
img_size = 512


def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        index = 0
        for img in os.listdir(path):
            if index == 100:
                break
            try:
                img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]  #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
                data.append([resized_arr, class_num])

            except Exception as e:
                print(e)
            index += 1
    return np.array(data)

#%%

def get_test_data(path):
    data = []
    for img in os.listdir(path):
        try:
            img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]  #convert BGR to RGB format
            resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
            data.append(resized_arr)

        except Exception as e:
            print(e)
    return np.array(data)


#%%

train_data = get_data(r'D:\Licenta\alaska2-steganalysis')
print(train_data)

#%%

test_data = get_test_data(r'D:\Licenta\alaska2-steganalysis\Test')
print(test_data)

#%%

x_train = []
y_train = []

for feature, label in train_data:
    x_train.append(feature)
    y_train.append(label)

# Normalize the data
x_train = np.array(x_train) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)


#%%

# Define the model

model = models.Sequential()
model.add(layers.Conv2D(32, 3, padding="same", activation="relu", input_shape=(512, 512, 3)))
model.add(layers.MaxPool2D())

model.add(layers.Conv2D(32, 3, padding="same", activation="relu"))
model.add(layers.MaxPool2D())

model.add(layers.Conv2D(64, 3, padding="same", activation="relu"))
model.add(layers.MaxPool2D())
model.add(layers.Dropout(0.4))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(4, activation="softmax"))

print(model.summary())

#%%

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#%%

history = model.fit(x_train, y_train, epochs=50, verbose=1)

