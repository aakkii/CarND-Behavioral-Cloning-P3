
# coding: utf-8

# ## Import Packages

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
import numpy as np
import cv2
import sklearn
import sklearn.utils

#Loading the csv file lines only. Not loading the images. 
lines = [] 
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

d_name = lines[1][0].split('/')[0]
name = d_name +'/IMG/'+lines[1][0].split('/')[-1]
print(name)
image = cv2.imread(name)
print(image.shape)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

print(len(train_samples))
print(train_samples[-1])
print(len(validation_samples))
print(validation_samples[-1])

#Generator. Loading of actual images happens inside this. It also augments the images by flipping it and also loading center, left camera images.
def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction= 0.1
    print(num_samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:                
                center_name = batch_sample[0].split('/')[0] +'/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(center_name)
                center_angle = float(batch_sample[3])
                
                images.append(center_image)
                angles.append(center_angle)
		                
                left_name = batch_sample[1].split('/')[0] +'/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(left_name)
                left_angle = float(batch_sample[3])
                
                images.append(left_image)
                angles.append(left_angle+0.1)
                
                right_name = batch_sample[2].split('/')[0] +'/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(right_name)
                right_angle = float(batch_sample[3])
                
                images.append(right_image)
                angles.append(right_angle-0.2)
                
                aug_images, aug_angles = [], []
                
                for image, angle in zip(images, angles):
                    aug_images.append(image)
                    aug_angles.append(angle)
                    aug_images.append(cv2.flip(image,1))
                    aug_angles.append(angle*-1.0)

            
            X_train = np.array(aug_images)
            y_train = np.array(aug_angles)
            yield sklearn.utils.shuffle(X_train, y_train)
        
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#Model to train. 
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

#using adam optimizer. 
model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
#using generator
model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*6, validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, nb_epoch=5)
model.save('model.h5')

