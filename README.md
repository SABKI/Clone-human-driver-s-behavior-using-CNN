# Clone-human-driver-s-behavior-using-CNN

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/data_augm_ex.png "original image & augmented image with brightness"
[image2]: ./examples/cnn_model.png "Nvidia CNN architecture"
[image3]: ./examples/loss_plot.png "Model MSE loss"

#### Submission files 

My project includes the following files:

* model.py containing the complete pipeline 
* drive.py for driving the car in autonomous mode
* model.h5 containing the trained Nvidia's CNN model 
* writeup_report.md summarizing the results.


### 1. The data 

I did use the data samples provided by Udacity for training and validation. The test dataset is the autonomous mode in the simulator.

In order to generate multiple data, I did the following :
* 1. Use the training data provided by Udacity
* 2. Use the three cameras with a correction angle 
* 3. Data augmentation with flipping 
* 4. Data augmentation with random brightness 
* 5. Data augmentation with random shadows

##### 1.1 Angle orrection

A constant 0.22 is added to left camera image steering angles and substracted from right camera image steering angles. This forces aggressive right turns when drifting to the left of the lane, and vice-versa. The code below describes the implementation of this approach.
```
correction = 0.22
images = []
            steering_angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    file_name = path_leaf(source_path)
                    current_path = '../CarND-Behavioral-Cloning-P3/data_udac/IMG/'+ file_name
                    image = ndimage.imread(current_path)
                    if (i == 0):
                        angle = float(batch_sample[3])
                    if (i == 1):
                        angle = float(batch_sample[3]) + correction
                    if (i == 2):
                        angle = float(batch_sample[3]) - correction
                    images.append(image)
                    steering_angles.append(angle)
```

##### 1.2 Data Augmentation

The training recorded data contains noises and variations in throttle, speed at various instances of the track.
The Udacity's training set preserve a constant throttle, which reduce some part of the noises.

###### 1.2.1 Flipping the image

The recorded images are flipped to generate additional data, and also to balance the data to stay away from a bias for going left or right. The steering angles are multiplied by (-1) for all the flipped images. Images are also cropped to remove the top 70 pixels and the bottom 25 pixels. These areas should not be relevant in training our model.

The code below describes the implementation of this approach.
```
    augmented_images.append(cv2.flip(image,1))
    augmented_steering_angles.append(measurement*-1.0)
```
###### 1.2.2 Apply random brightness to the image

This approach is for goal to simulate day & night conditions. the function generates images with random brightness by first converting images to HSV then randomly scaling the V_channel, then converting back to the RGB_channel. The code + comments that explains how it works for this augmentation are presented by the function defined between lines 29 & 45 in "model.py".

![alt text][image1]

### 2. Model Architecture

In order to reduce the loss, I did tweak the model with dropout layers (based on this article: https://www.scitepress.org/Papers/2019/75759/75759.pdf ), they claim that their architecture could help with overfitting and helped my model to have the lowest loss rate and highest accuracy in the training dataset. After some experiments, I didn't notice a big difference in term of performances, in fact, I was unable to reach a smaller loss than Nvidia's model on the Udacity training set.

##### Final Model Architecture

I adopted Nvidia's model as my final architecture of the model after starting with basic CNN models,

![alt text][image2]


### 3. Training

I trained the model using the keras generator with batch size of 64 for 3 epochs. I used adam optimizer with a learning rate of 0.0001. The entire training took about 3 minutes.


```
history_object = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
                                     validation_data=validation_generator, 
                                     nb_val_samples=math.ceil(len(validation_samples)/batch_size), 
                                     nb_epoch=3, 
                                     verbose=1)
```

##### Visualize Model Training History in Keras

![alt text][image3]

### 4. Testing

The output video corresponding to my testing in the simulator is provided with the submission files.
