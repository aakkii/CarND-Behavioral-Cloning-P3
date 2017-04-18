#**Behavioral Cloning** 

##
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  


####1. File Submission

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. Model

* At first I am normalizing the image data using Lambda Layer. (line 94)
* Second step is to cropp the images. This is done to reduce the images to only required part needed to determine steering angle. (line 95) 
* Third step is Convolution Neural Network with 5x5 filter (line 96 to 100)
* This includes RELU layers to introduce nonlinearity. (line 96 to 100)
* After this MaxPooling Layer is added (line 97, 99)

####2. Attempts to reduce overfitting in the model

The model was trained and validated by shuffling the data sets to ensure that the model was not overfitting (code line 28-81). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 106).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving using different camera angels and flipping. I also tried recovering from left and right but that didn't help much to improve the driving on track. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to stick to the basics. That means capture and use good driving data, model based on proven well-known architecture, different data augmentations techniques and optimizing parameters. 

I started with basic LENET model. Then slowly added Lambda Layer, Cropping of images, using left and right camera images and flipping of images step by step. Each time measuring the run on simulator and making sure it improves the run incrementally. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track mainly during turn, on the bridge and steep turn immediately after the bridge. To improve the driving behavior in these cases I used left and right camera images and then tuned the correction parameter. I also captured good driving runs dataset to train model better.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 93-103) consisted of a convolution neural network with the following layers and layer sizes.

* Lambda Layer
* Cropping
* CNN2D Layer using 5x5 filter
* RELU activation
* Maxpooling Layer
* CNN2D Layer using 5x5 filter
* RELU activation
* Maxpooling
* Flatting and then fully connected layers

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded one lap on track one using center lane driving.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover. I didn't use this dataset in my final run since I didn't find it much useful and in some cases it was making run worse. I found using left and right camera images much more useful. I used left camera and right camera images heavily so that model can learn to recover to center if car drifts off the left or to the right. This helped in navigating thourgh tricky turns specially the one that is right after the bridge. 

To augment the data sat, I also flipped images and angles thinking that this would help in generalization and will also provide more data to train the model. I also added flipping for center and left camera images. This almost increased my dataset by 6 times and provided much better generalization.

After the above process, I had ~40000 of data points. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as validation loss didn't improve much and started increasing. I used an adam optimizer so that manually training the learning rate wasn't necessary.
