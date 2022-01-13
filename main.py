import tensorflow as tf
from tensorflow.keras import layers, models, utils
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
import pandas as pd

gpus = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)

CATEGORIES = ["Amazon", "Apple", "Disney", "Facebook", "Google", "IBM", "Intel", "Netflix"]
TRAININGDIR = 'C:/Varun/Codenges/ML/exun2021/prelims/task2/train'
TESTINGDIR = 'C:/Varun/Codenges/ML/exun2021/prelims/task2/test'
IMGSIZE = 100
trainingData = []
testingData = []
imgTrain = []
labelTrain = []
imgTest = []
labelTest = []

# for category in CATEGORIES:
#     path = os.path.join(TRAININGDIR, category)
#     classIndex = CATEGORIES.index(category)
#     for img in os.listdir(path):
#         imgArray = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
#         newArary = cv2.resize(imgArray, (IMGSIZE, IMGSIZE))
#         trainingData.append([newArary, classIndex])
# random.shuffle(trainingData)
# for img,label in trainingData:
#     imgTrain.append(img)
#     labelTrain.append(label)

# imgTrain = np.array(imgTrain).reshape(-1, IMGSIZE, IMGSIZE, 3)
# imgTrain = utils.normalize(imgTrain)

# labelTrain = np.array(labelTrain)
# labelTrain = utils.to_categorical(labelTrain, 8)
# np.save("imgTrain", imgTrain)
# np.save("labelTrain", labelTrain)

# # imgTrain = np.load("imgTrain.npy")
# # labelTrain = np.load("labelTrain.npy")

# # print(imgTrain[0])












# model = models.Sequential([
#     layers.Conv2D(64, (3,3), activation='relu', input_shape=imgTrain.shape[1:]),
#     layers.MaxPooling2D(2, 2),
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPooling2D(2,2),
#     layers.Conv2D(128, (3,3), activation='relu'),
#     layers.MaxPooling2D(2,2),
#     layers.Conv2D(128, (3,3), activation='relu'),
#     layers.MaxPooling2D(2,2),
#     layers.Conv2D(256, (3,3), activation='relu'),
#     layers.MaxPooling2D(2,2),

#     layers.Flatten(),
#     layers.Dropout(0.5),
    
#     layers.Dense(256, activation='relu'),
#     layers.Dense(512, activation='relu'),
#     layers.Dense(8, activation='softmax')
# ])


# model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics = ["accuracy"])

# history = model.fit(imgTrain, labelTrain, epochs=50, validation_split=0.2)

# model.save("the"+str(round(history.history['accuracy'][49], 2)))

model = models.load_model('the0.98')
ansArray = []

print("\n\n\n\n")

for img in os.listdir(TESTINGDIR):
    imgArray = cv2.imread(os.path.join(TESTINGDIR,img), cv2.IMREAD_COLOR)
    newArray = cv2.resize(imgArray, (IMGSIZE, IMGSIZE))
    testingData.append([newArray])
    the = model.predict(newArray.reshape(-1, IMGSIZE, IMGSIZE, 3))
    print(the)
    print(img)
    ansArray.append([str(img).split(".")[0], np.argmax(the)])
    # plt.imshow(imgArray)
    # plt.title(CATEGORIES[np.argmax(the)])
    # plt.show()

ans = pd.DataFrame(ansArray, columns=['id', 'company'])
ans.to_csv('submissions3.csv')
print("done")