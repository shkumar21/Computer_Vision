# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 23:47:01 2021

@author: Arman
"""
imagenette_map = { 
    "n01440764" : "tench",
    "n02102040" : "springer",
    "n02979186" : "casette_player",
    "n03000684" : "chain_saw",
    "n03028079" : "church",
    "n03394916" : "French_horn",
    "n03417042" : "garbage_truck",
    "n03425413" : "gas_pump",
    "n03445777" : "golf_ball",
    "n03888257" : "parachute"
}

from keras.preprocessing.image import ImageDataGenerator

# create a new generator
imagegen = ImageDataGenerator()
# load train data
train = imagegen.flow_from_directory(r'D:/study/5th Semester/Computer Vision and Deep learning/New Material/Project and exercises/imagenette2-160/train/', class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))
# load val data
val = imagegen.flow_from_directory(r'D:/study/5th Semester/Computer Vision and Deep learning/New Material/Project and exercises/imagenette2-160/val/', class_mode="categorical", shuffle=False, batch_size=128, target_size=(224, 224))


from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout

# build a sequential model
model = Sequential()
model.add(InputLayer(input_shape=(224, 224, 3)))

# 1st conv block
model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

# 2nd conv block
model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())

# 3rd conv block
model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
model.add(BatchNormalization())

# ANN block
model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(0.25))

# output layer
model.add(Dense(units=10, activation='softmax'))
model.summary()

# compile model
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

# fit on data for 30 epochs
model.fit_generator(train, epochs=30, validation_data=val)

# Save the model to disk.
model.save_weights('img_cla')

# extract train and val features
features_train = model.predict(train)
features_val = model.predict(val)