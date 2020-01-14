# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* [x] Use the simulator to collect data of good driving behavior
* [x] Build, a convolution neural network in Keras that predicts steering angles from images
* [x] Train and validate the model with a training and validation set
* [x] Test that the model successfully drives around track one without leaving the road
* [x] Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/model.png "Model Visualization"
[image2]: ./img/error_loss.png "Error loss"
[image3]: ./img/normal.jpg "Normal Image"
[image4]: ./img/flipped.jpg "Flipped Image"
[image5]: ./img/cropped.jpg "Cropped Image"
[image6]: ./img/video.gif "Video"
## Rubric Points
---
### Files Submitted & Code Quality

#### 1. required files to run the simulator in autonomous mode

My project includes the following files:
* train.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 with the final run

#### 2. functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Training/model code

The train.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model architecture

I used a network architecture created by [nvidia](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)
 
 My model consists of:
 
 * 9 layers
 * a cropping (lambda) layer (line 129)
 * a normalization layer
 * 5 convolutional layers
 * 3 fully connected layers

 (train.py lines 127-138) 


#### 2. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 140).

#### 3. Appropriate training data

I used the provided data set. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I stared with a very simple model which had only a normalization and a flatten layer.  

I moved on to a lenet5 model with some convolution and pooling layer. 

After not getting the car to stay on the track, I switched to the network design from  [nvidia](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

I did not use Dropout layers because results have been good enough.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

![alt text][image6]

#### 2. Final Model Architecture

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Training Set & Training Process

I only used the provided dataset for training

To augment the data set, I:

- flipped images and angles thinking that this would 
- used the images from the left and right camera and a correction value for the angle
- cropped the images

![alt text][image3]
![alt text][image4]
![alt text][image5]


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

+ I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
+ adam optimizer so that manually training the learning rate wasn't necessary.
+ correction of 0.2 for the left/right angles 
+ My batch_size was 256 and run for 10 epochs

![alt text][image2]