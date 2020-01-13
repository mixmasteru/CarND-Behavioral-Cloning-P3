import csv
from math import ceil

import imageio
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf
from keras.layers import Flatten, Dense, Conv2D, AveragePooling2D, Cropping2D
from keras.layers import Lambda
from keras.models import Sequential
from sklearn.model_selection import train_test_split

lines = []
images = []
measurements = []
correction = 0.2
data_path = './data/IMG/'
batch_size = 256
epochs = 10

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


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name_c = data_path + batch_sample[0].split('/')[-1]
                name_l = data_path + batch_sample[1].strip().split('/')[-1]
                name_r = data_path + batch_sample[2].strip().split('/')[-1]

                center_image = imageio.imread(name_c)
                center_angle = float(batch_sample[3])
                left_image = imageio.imread(name_l)
                left_angle = float(batch_sample[3]) + correction
                right_image = imageio.imread(name_r)
                right_angle = float(batch_sample[3]) - correction

                images.append(center_image)
                angles.append(center_angle)
                images.append(left_image)
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle)
                images.append(np.fliplr(left_image))
                angles.append(-left_angle)
                images.append(np.fliplr(right_image))
                angles.append(-right_angle)


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def simple_model():
    for line in lines:
        measurement = float(line[3])
        path = './data/'
        add_img(path + line[0], measurement)
        add_img(path + line[1].strip(), measurement + correction)
        add_img(path + line[2].strip(), measurement - correction)

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


def lenet5_model():
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)

    ch, row, col = 3, 80, 320  # Trimmed image format
    # Set our batch size
    batch_size = 32
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(90, 320, 3), output_shape=(90, 320, 3)))
    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(90, 320, 3)))
    model.add(AveragePooling2D())
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    # model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)
    model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples) / batch_size),
                        validation_data=validation_generator,
                        validation_steps=ceil(len(validation_samples) / batch_size),
                        epochs=5, verbose=1)
    model.save('model.h5')


train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = Sequential()
# model.add(Lambda(lambda x: tf.image.rgb_to_grayscale(x), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Conv2D(64, 3, 3, activation="relu"))
model.add(Conv2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=ceil(len(train_samples) / batch_size),
                                     validation_data=validation_generator,
                                     validation_steps=ceil(len(validation_samples) / batch_size),
                                     epochs=epochs,
                                     verbose=1)
model.save('model.h5')

print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('img/error_loss.png')
