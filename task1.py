# RUN getData.py BEFORE RUNNING THIS

import numpy as np
import tensorflow as tf
import os
import cv2
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten

gpus = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)

photos = np.load('palm-images.npy')
labels = np.load('palm-labels.npy')

print(labels)
# print(photos.shape, labels.shape)

model = tf.keras.models.Sequential([
  Conv2D(64, (3, 3), activation = 'relu', input_shape = (160, 120, 3)),
  MaxPool2D(2, 2),
  Conv2D(128, (3, 3), activation = 'relu'),
  MaxPool2D(2, 2),
  Conv2D(256, (3, 3), activation = 'relu'),
  MaxPool2D(2, 2),
  Flatten(),
  Dense(128, activation = 'relu'),
  Dense(1, activation = 'sigmoid', dtype = 'float32')
])
opt = tf.keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = 'acc')


# fit model
history = model.fit(photos, labels, batch_size=8,
                    epochs=5, validation_split=0.3)

model.save("fina_task1Model")

# model = tf.keras.models.load_model("task1model")


testingData = []
ansArray = []

for img in os.listdir("C:/Varun/Codenges/ML/exun2021/prelims/task1/test"):
    imgArray = cv2.imread(os.path.join("C:/Varun/Codenges/ML/exun2021/prelims/task1/test",img), cv2.IMREAD_COLOR)
    newArray = cv2.resize(imgArray, (160, 120))
    testingData.append([newArray])
    the = model.predict(newArray.reshape(-1, 160, 120, 3))
    ansArray.append([str(img).split(".")[0], int(the[0][0])])


ans = pd.DataFrame(ansArray, columns=['id', 'Aspectofhand'])
ans.to_csv('PLEASEWORK.csv')
print("done")
