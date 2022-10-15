# **Traffic Sign Recognition** 


---
One of the essential part of bringing the autonoous vehicles on the road is to estimate the traffic signs, since without rules on the road it would be like crashing the robots in luna park.

There are many ways to recognize the traffic signs thanks to latest researches. 
The very first step of recognizing something is to know what to recognize. For this purpose it is needed to pump some example traffic signs into some kind of system to teach what the system should recognize, which is called training sets. After the system knows what to recognize, we need to validate the trained system how precise the system is, while it is recognizing the inputs. At this place the inputs are the real images taken from the system, e.g. camera on board. The system observes the environment and feeds the environment through trained network to say what is the meaning of the taken input. 

For this puprpose in this project the algotihm is implemented to recognize the traffic signs via LeNet neural network. 

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./WriteUp/allImages.png "all Images"
[image2]: ./WriteUp/distImages.png "Bar Chart"
[image3]: ./WriteUp/grayScale.png "gray scale"
[image4]: ./WriteUp/normalized.png "normalized"
[image5]: ./WriteUp/gtsrb.png "gtsrb images"

### 1. Data Set Summary & Exploration

The required datas are provided in picled form. The pickled data is a dictionary with 4 key/value pairs:

- 'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- 'labels' is a 1D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
- 'sizes' is a list containing tuples, (width, height) representing the original width and height the image.
- 'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

The code is given in python script.

### 2. Visualization of the dataset.

I have used matplotlib to visualize one image per unique label and a bar chart for the total number of train images with respect to its label. Here is an exploratory visualization of the data set. 

One example test image per unique label is given as follows:

![alt text][image1]

The distribution of the images is given on bar chart as follows:

![alt text][image2]

### 3. Preprocessing of data


As a first step, I decided to convert the images to grayscale because it will have only one channel to analyze instead of three channels, thus, it reduces the amount feature which results in better/faster computing performance. In training images there are also bad images which are really dark and hard to recognize them, so converting the image to gray scale will also help at this point to make easier recognition.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

As a last step, I normalized the image data by mapping the pixels' values between 0 and 255 to the values between -1 and 1, using the provided formula `(pixel - 128)/ 128`, which makes the recognition better, since we work in narrower value region.

Here is an example of a traffic sign image before and after grayscaling and normalization.

![alt text][image4]



### 4. Final LeNet architecture


My first model consisted of the following layers:

| Layer                 |     Description                                | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 normalized grayscale image            | 
| Convolution 5x5         | 1x1 stride, valid padding, outputs 28x28x6     |
| RELU                    |                                                |
| Max pooling              | 2x2 stride,  outputs 14x14x6                     |
| Convolution 5x5        | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                    |                                                |
| Max pooling              | 2x2 stride,  outputs 5x5x16                     |
| Flatten                | outputs 400                                   |
| Fully connected        | outputs 120                                    |
| RELU                    |                                                |
| Fully connected        | outputs 84                                    |
| RELU                    |                                                |
| Fully connected        | outputs 43 (the number of classes)            |
| Softmax                |                                                |

My final model consisted of the following layers:

| Layer                 |     Description                                | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 normalized grayscale image            | 
| Convolution 5x5         | 1x1 stride, valid padding, outputs 28x28x6     |
| RELU                    |                                                |
| Max pooling              | 2x2 stride,  outputs 14x14x6                     |
| Convolution 5x5        | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                    |                                                |
| Max pooling              | 2x2 stride,  outputs 5x5x16                     |
| Flatten                | outputs 400                                   |
| Fully connected        | outputs 120                                    |
| RELU                    |                                                |
| Dropout Prob            | 0.6                                            |
| Fully connected        | outputs 84                                    |
| RELU                    |                                                |
| Dropout Prob            | 0.6                                            |
| Fully connected        | outputs 43 (the number of classes)            |
| Softmax                |                                                |

### 5. Model Specifications
| Name                   |     Value                                        |
|:---------------------:|:---------------------------------------------:| 
| Batch Size            | 128                                            |
| number of EPOCS        | outputs 175                                    |
| learning rate            | 0.0089                                        |
| mu                    | 0                                                |
| sigma                 | 0.01                                           |
| Dropout Prob            | 0.6                                            |
| Optiimzer             | Adam                                          |

### 6. Final Solution

My final model results were:
* training set accuracy of %100
* validation set accuracy of %97.7 
* test set accuracy of %95.3

I chose an iterative aprroach. First of all I have taken basic LeNet model from course and check how it acts. Wit this model, I did not have enough accuracy (min %93). Then I have started to play with parameters untill I do not have any changes anymore. My all iterative parameter changes and their results given in Table as follows:

| Batch Size | number of EPOCS|learning rate|Dropout Prob.|    training set accuracy| validation set accuracy|test set accuracy|
|:----------:|:--------------:|:-----------:|:-----------:|:--------------------:|:----------------------:|:---------------:| 
|128          |20               |0:001        |-            |%99.3                 |%89.2                   |%88.3            |
|128          |50               |0:001        |-            |%100                  |%90.3                   |%88              | 
|128          |50               |0:0008       |-            |%99.9                 |%93                     |%91.3            |
|128          |100               |0:0008       |-            |%100                  |%93.8                   |%91.6            | 
|128          |100               |0:00089      |-            |%100                  |%93.4                   |%92.5            |
|128          |170               |0:00089      |-            |%100                  |%94.1                   |%92.9            |
|128          |175               |0:00089      |-            |%100                  |%94.0                   |%92.6            |
|128          |175               |0:00089      |0.5          |%100                  |%96.8                   |%95.1            |
|128          |175               |0:001        |0.5          |%100                  |%95.9                   |%94.7            |
|128          |175               |0:00089      |0.6          |%100                  |%97.2                   |%94.9            |
|128          |200               |0:00089      |0.6          |%100                  |%96.8                   |%94.4            |
|128          |175               |0:00089      |0.6          |%100                  |%97.7                   |%95.3            |

As it mentioned before, i have taken the LeNet model from the course as a starting point and then I have started to modify the parameters using trial and error method. If it gives better result in next iteration, continue with the parameters in next iteration, but if it gives worse result, then continue with actual parameters. I have kept this logic until I do not have very big changes anymore (>%1) in validation set accuracy. 

And another issue was overfitting that I was getting always big difference between test set accuracy (~%100) and validation test accuracy (~%90-%94). In order to prevent overfitting I have used dropout with keep probability (0.5 and 0.6) to decrease the noisy images in validation sets (e.g. very dark images, very bright images etc.). With this change, i got finally %97.7 validation set accuracy at the end. 

 

### 7. Test a Model on New Images

Here are five German traffic signs that I have randomly choosen on the web:

![alt text][image5]

`Update after review:` I have especially chosen the images which have 32X32 resolution since I wanted to use the same resolution as it is used in my LeNet model, which only accepts 32x32 images as input. I want to mention here that the images could be resized but they will definietly loose their qulaity, which can affect the result negatively. And also I wanted to have a bright image (keep right), which is the second from the beginning. I wanted to have also dark image to see how my model performs on it, thus, i have chosen the image 'no passing for vehicles over 3.5 metric tones'. Additional to it, i have tried to get jittered image which is 'Turn right ahead' to see how good may model performs on jittered image, which could result in difficulty to be recognized. The first sign 'Vehicles over 3.5 metric tones prohibited' is not from the street, it is taken from the paper virtually, which I wanted to see how the model perfroms on it, which can also confuse the model. I have also searched an image, which has two signs on one image, for this prupose I have chosen the last one that has half of the other sign, which can confuse the model during recognition.

Here are the results of the prediction:

| Class - Image            |     Class - Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 16 - Vehicles over 3.5 metric tones prohibited | 16 - Vehicles over 3.5 metric tones prohibited | 
| 38 - Keep Right       | 38 - Keep Right                                |
| 10 - no passing for vehicles over 3.5 metric tones| 10 - no passing for vehicles over 3.5 metric tones|
| 33 - Turn right ahead              | 33 - Turn right ahead |
| 0 - Speed limit(20 km/h) |  0 - Speed limit(20 km/h)                                |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. It serves better result than the validation sets. It is good :).

I have uset `tf.nn.top.k` build-in function to show  how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. The top 5 softmax probabilities for each image along with the sign type of each probability are given in the table below.

The top five soft max probabilities were

| Original Image class | 1st|2nd|3rd|4th|5th|
|:----------:|:--------------:|:-----------:|:-----------:|:--------------------:|:----------------------:|
|16          |16 (%100)               |7 (%0)        |9 (%0)           |40(%0)                 |10(%0)            |
|38          |38 (%100)              |0 (%0)       |1 (%0)           |2 (%0)                  |3 (%0)   | 
|10          |10 (%100)                  |9 (%0)      |16 (%0)          |12 (%0)              |26 (%0)|
|33         |33 (%100)              |35 (%0)    |13 (%0)         |26 (%0)                |4 (%0)              | 
|0          |0 (%100)                  |4 (%0)    |1 (%0)          |8 (%0)                |40 (%0)               |

Actually it gave very good result that it predicted all 5 images with ~%100 accuracy better than validation set accuracy.

### 8. Further Discussion

Because of the timing limit (due to my job, as I am doing this course as an hobby) was not able to put some effort to generate more test images sets using for example prespective transformation, which I have learned on previous section, which definitely helps to improve the validation set accuracy. I am planning to implement it later. 
