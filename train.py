import csv

import imageio
import numpy as np
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.layers import Lambda

lines = []
images = []
measurements = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines.pop(0)

for line in lines:
    path = './data/' + line[0]
    image = imageio.imread(path)
    images.append(image)
    measurements.append(float(line[3]))

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
