# Import lib
import csv
import os
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from scipy import ndimage
import matplotlib.pyplot as plt
import random
import math

from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras import optimizers

import ntpath
# get the head of a path
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

# Randomly changes the brightness of the image.

def change_brightness(img, min=0.3, max=3.0):
	
	#I choosed the HSV color space because the Value channel describes the brightness or intensity of the color.
	
	# We covert from RGB to HSV using openCV
    HSV = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
	
	# the min-max arguments is a range wherefrom the random_brightness variable value is generated so we can use it by multiplying it to the V channel to have random brightness in each pixel (0.3 and 3 are randomly choosed)
    random_brightness = np.random.uniform(min,max)
	
    # After multiplying the V channel to random_brightness, we can exceed the max brightness value in certain pixels, to avoid this, we should define a mask where the surplus occurred and then set the values to the maximum at those locations using the "np.where" method
    mask = HSV[:,:,2] * random_brightness > 255
	# channel_V is the new channel with random brithness values 
    channel_V = np.where(mask, 255, HSV[:,:,2] * random_brightness)
	# set the channel to the HSV color space of the image then return it as an RGB 
    HSV[:,:,2] = channel_V
    return cv2.cvtColor(HSV,cv2.COLOR_HSV2RGB) 

# Read line infor from the CSV file
samples = []
csv_file = '../CarND-Behavioral-Cloning-P3/data/driving_log.csv'
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
           samples.append(row)
            
# Define the generator
def generator(samples, batch_size):
    num_samples = len(samples)
    correction = 0.22
    # Loop forever so the generator never terminates
    while 1: 
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            steering_angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    file_name = path_leaf(source_path)
                    current_path = '../CarND-Behavioral-Cloning-P3/data/IMG/'+ file_name
                    image = ndimage.imread(current_path)
                    if (i == 0):
                        angle = float(batch_sample[3])
                    if (i == 1):
                        angle = float(batch_sample[3]) + correction
                    if (i == 2):
                        angle = float(batch_sample[3]) - correction
                    images.append(image)
                    steering_angles.append(angle)

            augmented_images = []
            augmented_steering_angles = []
            
            for image, angle in zip(images,steering_angles):
                augmented_images.append(image)
                augmented_steering_angles.append(angle)
                
                augmented_images.append(cv2.flip(image,1))
                augmented_steering_angles.append(angle*-1.0)
                
                augmented_images.append(change_brightness(image))
                augmented_steering_angles.append(angle)
                
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_steering_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print(len(train_samples))
print(len(validation_samples))

# Define the generator function for the training and valiadtion set
batch_size = 64
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
ch, row, col = 160, 320, 3  # Trimmed image format

# the full model (Nvidia Architecture)

model = Sequential()
# Preprocessing  : normalize image pixels
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(ch, row, col),output_shape=(ch, row, col)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="elu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="elu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="elu"))
model.add(Convolution2D(64,3,3,activation="elu"))
model.add(Convolution2D(64,3,3,activation="elu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
#model.add(Dropout(0.2))
model.add(Dense(1))

# compile the model
adam = optimizers.Adam(lr=0.0001) # set adam optimizer with a precise learning rate
model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])

# train the model using the generator function
history_object = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
                                     validation_data=validation_generator, 
                                     nb_val_samples=math.ceil(len(validation_samples)/batch_size), 
                                     nb_epoch=3, 
                                     verbose=1)

model.save('model.h5')
print('model saved')
