# -*- coding: utf-8 -*-
"""DhruvAdi.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HrUfmDIK36Vg7E-sXhl6wrgW5vqwYRDb
"""

import os
base_dir = 'C:\\Users\\Dhruv\\Documents\\cam\\dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir  = os.path.join(base_dir, 'test')

train_adi = os.path.join(train_dir, '0')
test_adi  = os.path.join(test_dir, '0')

train_dhruv = os.path.join(train_dir, '1')
test_dhruv  = os.path.join(test_dir, '1')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# All images will be rescaled by 1./255
train_valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=float(5/75))
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_valid_datagen.flow_from_directory(
        train_dir,
        batch_size=16,
        target_size=(150,150),
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary',
        subset='training'
        ) # Total of 4139 images

valid_generator = train_valid_datagen.flow_from_directory(
        train_dir,
        batch_size=16,
        target_size=(150,150),
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary',
        subset='validation'
        ) # Total of 887 images

test_generator = test_datagen.flow_from_directory(
    test_dir,
    batch_size=16,
    target_size=(150,150),
    class_mode='binary'
    ) # Total of 887 images

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras import regularizers

# Creating a Sequential Model and adding the layers
def make_model():
  model = Sequential()

  #Block one
  model.add(Conv2D(4, (3, 3), input_shape=(150, 150, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  #Block two
  model.add(Conv2D(4, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  #Block 3
  model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
  model.add(Dense(16,  activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(1,activation='sigmoid'))
  return model

model = make_model()
from tensorflow.keras.optimizers import RMSprop
from keras import metrics



model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
model.summary()
history = model.fit_generator(
      train_generator,
      steps_per_epoch=5,  # 2000 images = batch_size * steps
      epochs=10,
      validation_data=valid_generator,
      validation_steps=1,  # 1000 images = batch_size * steps
      verbose=1)

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_loss'], label='val loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend( loc='upper right')
plt.show()


# plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['acc'], label='val accuracy')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()
print(history.history.keys())

model.evaluate(test_generator)

modelsave = os.path.join(base_dir, 'complete')