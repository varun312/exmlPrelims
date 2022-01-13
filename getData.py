from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import save
from numpy import asarray
import os

dorsalFiles = os.listdir('C:/Varun/Codenges/ML/exun2021/prelims/task1/train/dorsal')
palmarFiles = os.listdir('C:/Varun/Codenges/ML/exun2021/prelims/task1/train/palmar')
photos, labels = list(), list()

IMG_WIDTH, IMG_HEIGHT = 160,120

for file in dorsalFiles:
    photo = load_img("C:/Varun/Codenges/ML/exun2021/prelims/task1/train/dorsal/" + file, target_size=(160, 120))
    photo = img_to_array(photo)
    photos.append(photo)
    labels.append(0)
for file in palmarFiles:
    photo = load_img("C:/Varun/Codenges/ML/exun2021/prelims/task1/train/palmar/" + file, target_size=(160, 120))
    photo = img_to_array(photo)
    photos.append(photo)
    labels.append(1)

photos = asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)

save('palm-images.npy', photos)
save('palm-labels.npy', labels)