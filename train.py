import csv

import imageio
import numpy as np
from keras.layers import Flatten, Dense
from keras.layers import Lambda
from keras.models import Sequential

lines = []
images = []
measurements = []
correction = 0.2

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines.pop(0)


def add_img(img_path, measurement):
    image = imageio.imread(img_path)
    image_flipped = np.fliplr(image)
    images.append(image)
    images.append(image_flipped)

    measurements.append(measurement)
    measurements.append(-measurement)


for line in lines:
    measurement = float(line[3])
    path = './data/'
    add_img(path + line[0], measurement)
    add_img(path + line[1].strip(), measurement+correction)
    add_img(path + line[2].strip(), measurement-correction)

print(len(images))
print(len(measurements))

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)

model.save('model.h5')
