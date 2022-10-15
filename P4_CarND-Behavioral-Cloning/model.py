# import the libraries
import math
import os
import csv
import cv2
import numpy as np
import sklearn
from random import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers import Conv2D

#load images
samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)

#split images between train samples in %80 and validation samples in (%20)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# define generator to laod and preprocess it on fly
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            correction = 0.2 # taken from class as a starting point 
            for batch_sample in batch_samples:
                name_center = 'data/IMG/'+batch_sample[0].split('/')[-1]
                img_center = cv2.imread(name_center)
                center_angle = float(batch_sample[3])
                images.append(img_center)
                angles.append(center_angle)
                center_AugmentedImage = cv2.flip(img_center,1) # create an augmented images usling flip techniques for the center
                center_AugmentedAngle = center_angle*-1.0
                images.append(center_AugmentedImage)
                angles.append(center_AugmentedAngle)                
                # create adjusted measurements for the side camera images
                # left images
                name_left = 'data/IMG/'+batch_sample[1].split('/')[-1]
                img_left = cv2.imread(name_left)
                left_angle = float(batch_sample[3]) + correction
                images.append(img_left)
                angles.append(left_angle)                
                left_AugmentedImage = cv2.flip(img_left,1) # create an augmented images usling flip techniques for the left
                left_AugmentedAngle = left_angle*-1.0
                images.append(left_AugmentedImage)
                angles.append(left_AugmentedAngle)                
                #right images
                name_right = 'data/IMG/'+batch_sample[2].split('/')[-1] 
                img_right = cv2.imread(name_right)
                right_angle = float(batch_sample[3]) - correction
                images.append(img_right)
                angles.append(right_angle)                  
                right_AugmentedImage = cv2.flip(img_right,1) # create an augmented images usling flip techniques for the right
                right_AugmentedAngle = right_angle*-1.0
                images.append(right_AugmentedImage)
                angles.append(right_AugmentedAngle) 
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# nvidia deep neural network model as proposed in the course
ch, row, col = 3, 160, 320  # Input image format 
model = Sequential()
# Normalize image and centralize images
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(row,col,ch)))
# Crop image to filter out the irrelevant parts of the images
model.add(Cropping2D(cropping=((70,25),(0,0))))
# Convolution 5x5 Layers using elu activation function
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="elu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="elu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="elu"))
# Convolution 3x3 Layers using 
model.add(Conv2D(64, (3, 3), activation="elu"))
model.add(Conv2D(64, (3, 3), activation="elu"))
# Pooling layer 2x2
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
# Dropout layer with keep_prob = 0.5
model.add(Dropout(0.5))
model.add(Flatten())
# Full-Connected Layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#print(model.summary())
#print(len(train_samples), len(validation_samples))


#adam optimizer and Mean squared error 
model.compile(loss='mse', optimizer='adam')
# history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,   nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)
history_object = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size),
            validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size),
            epochs=5, verbose=1)
model.save('model.h5')

# from keras.models import Model
# import matplotlib.pyplot as plt
# ### print the keys contained in the history object
# print(history_object.history.keys())
### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()



