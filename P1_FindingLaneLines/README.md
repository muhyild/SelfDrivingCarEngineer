# **Project 1: Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

Finding lane lines in real time is one of the most difficult task in the research field of the autonomous driving, because of two main resons:
    * Hardware : Camera used in the vehicle, on board microprocessor 
    * Bad road conditions in the real world

However thanks to many researches in the field of Hardware development, the first reason is almost solved. And also thanks to the recent researches in the SW development field has also solved many problems, which is caused due to real world bad road conditions, for example, not clearly seeable lines, different line colours, bad road surface, etc. 

In this project I am going to implement a computer vision algorithms in the pipeline in order to be able to find the road lane line on the real data recorded with the front camera of a vehicle that is driven on a highway. I am going to explain the application of computer vision algorithms and the result gotten from these example recorded videos, which are also a series frame of images, and the example images in the next sections.

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report
* Apply the image pipelinne to the video format
* Apply the different algorithms in order to be able to find the lane line out
    * Adjust the parameter of the applied algorithms -> it takes lots of time :)
---
This project is implemented using the following tools and programming languages:
* Praogramming languages : Python with the extension to OpenCV 
    * Libraries : matplotlib, numpy, math, cv2 
* Tool: Anaconda, Jupyter Notebook
* OS: Windows

I do not want to include any images here, since they are already in my python notebook (or in my project workspace)

### Pipeline Steps

#### 1. Preprocessing the image 

The original images have three channels, which are red, green and blue. Working with these three channels are complex to identofy the lane, because we caompare the pixels with three channels to find exact line. Even if it would be possible to work original image, it is difficult to adapt the algoeritm based on different lane line colors, as all lane lines are not white or other colour in the real world. 

Therefor it is important to convert the original image to only one channel. As it is represented in the previous course, I have used the following method to convert my image to gray value (one channel image).

On the top of that we need to upload the image, and read to continue. The commands are as follows:

```
    ### upload the image and read the image ###
    image = matplotlib.image.imread('solidWhiteRight.jpg')
    
    ### convert from RGB to GRAY ###
    cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)   
```
#### 2. Smoothening the image 

As soon as we convert the image from RGB to GRAY, the gray image can be smoothed as follows:
```
    ### smooth the gray image ###
    cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)  
```
here is the kernel size will help to determine how much we want to blur the image, which will help us later, while detecting the edge. There will not be sharp changes on the whole image, but only between the lane line surfaces and the other surfaces. It will increase the efficiency during an edge detection.  

In my algorithm, I have used the kernel size as:
```
    ### kernel size ###
    kernel_size = 7
```   
I have found my optimal kernel_size by testing different values, but as it is shown in previous course, there are some written tools helping to decide the kernel_size based on an individual images. I would like to write also one of them, since I spent a lot of time to find an optimal kernel_size working with other important parameters in my algorithm.

#### 3. Detecting the edges in the image 

Now it is time to detect the edges (the sharp changes in image) in the image, since it is much easier to work only with lines than the whole image itself. It is very straightforward with build-in funciton in OpenCv library as follows:
```
    ### edge detection in the image ###
    cv2.Canny(img, low_threshold, high_threshold) 
```

The algorithm works as follows:
* Strong edge pixels above the high_threshold will be detected, and it will reject pixels below the low_threshold
* Pixels with values between the low_threshold and high_threshold will be included as long as they are connected to strong edges. 
* The output edges is a binary image with white pixels tracing out the detected edges and black everywhere else.

Here i chose the `low_threshold` as `100`, and `high_threshold` as `200`. How I chose them is again testing the different combinations, and choosing the one working the best with other parameters. Especially I was having some issue during the lane detection on the videos, since it was dropping `trace_back` error, which I do not still know why. But I will check it later (it will be my first for possible improvements, why it was driopping this error sometimes) 

#### 4. Selecting region of interest in the image 

Yes, I wrote couple of times here, but I will write once more to place this kind of kinding on my head. Working in the smalller area is always easier than the larger one. Thererfore it is now time to restrict the region of interest in the image. For this purpose I have learned two different ways to perfom, which are as follows:

```
    ### using the triangle method in the image ###
    region_select = np.copy(image)
    left_bottom = [100, ysize]
    right_bottom = [800, ysize]
    apex = [xsize/2, ysize/2]
    # fit the point on the line of the triangle = y= Ax + B
    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1) 
    
    ### using fillPolly method in the image ###
    ignore_mask_color = 255
    vertices = np.array([[(100,image.shape[0]),(380, 337), (610, 337), (890,image.shape[0])]], dtype=np.int32)
    mask = np.zeros_like(image) 
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
```
The first way is to say, it is manual by choosing the edge points of the triangle and form the region of the interest. I have not used this method, as there is alreay built-in function in OpenCv library, to mask  the region of the interest in a polygon form by only defining the vertices of the polygon. It took me a lot of time to find the best values. After many trials. I came up with values which are given at the top, which fit perfectly to my algorithm. As to remind, the masking is applied using `bitwise_and` to the grayscale image, if the input image is grayscale image.

#### 5. Finding the lines from the edge pixels

After having the edge pixels, it is now time to form the line using these edge pixels. For this purpose the `Hough Transform` is used, which was also presented in the course. 

The Hough Transform works as follows:
* it takes the edge pixels from (x,y) the coordinates of the image space, then it transforms them to Hough space, on which the points are represented as a line. This line represents the image space line which can pass through the points.

I have used to implement this transormation by using built-in function on OpenCV as follows:

```
    ### Hough Transformation ###
    lines = hough_lines(masked_image, rho=1, theta=np.pi/180, threshold=20, min_line_len=75, max_line_gap=170)
```
I have chosen the input parameter `rho, theta , threshold, min_line_len, max_line_gap` by testing different values. I found the most optimal values for these parameters as mentioned above.

Normally at the beginning the draw_lines function was also implemented in this houg_lines funciton in order to draw the lines on a blank image. However I have changed this function to get the lines only not to draw the lines, since I needed to have the lines points for my other two function which I have implemented to generate only one line for the left line and for the right line (averaging and extrapolation). The draw_lines function is only used to draw lines from the given pixel points. The draw_lines function returns the image. The modified functions are given as below:

```
    ### modified Hough Transformation helper funciton ###
    def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        return lines
     
    ### modified draw_lines helper funciton ###
    def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
        return line_img      
```

#### 6. Averaging and Extrapolating the lines to get a single line for left line and right line

After generating lines (pixel points of the lines) using Hough transformation, now it is time to get a common/single line for left line and also for the right line. In order to find a single line for both side, the important steps in my algorithm are as follows:

* classifying the lines to know which lines lie on the left and which lines lie on the right based on the slope:  positive slope is the right line with respect to coordinate system and negative slop is the left line wrp. to coordinate system
* extrapolating the average of the right and the left lines to match the height of the edges in the region of interests (in this case it is in vertices defined in the section 4)
    * it firstly averages the all left lines with their slopes, and also the all right lines with their slopes
    * Then the average values are used in the extrapolation to find one single line for the left and another for the right, using extrapolation formula as follows:
      $x = X1_{avg} + ((y-Y1_{avg})/(Y2_{avg}-Y1_{avg}))*(X2_{avg}-X1_{avg}) =  X1_{avg} + (y-Y1_{avg})/SLOPE_{avg}$
    

The written funcitons are as below: 

* the laneLineClassifier takes the lines from Hough Transofmation and classifies the lines to the right and left based on slope.Then it returns the line points and slope in a numpy array.
```
    ### lane classifier function###
    def laneLineClassifier(lines):
        right_line = [] 
        left_line = []
        for line in lines:
            for x1,y1,x2,y2 in line:
                # slope ((y2-y1)/(x2-x1)) -> x2 cannot be equal to x1 -> vertical line
                # positive slope is the right line with respect to coordinate system
                # negative slop is the left line wrp. to coordinate system
                if x2 != x1:
                    slope = ((y2-y1)/(x2-x1))
                    if slope < 0 :
                        left_line.append([x1,y1,x2,y2,slope]) 
                        #print(int(slope))
                    elif slope > 0.3 :
                        right_line.append([x1,y1,x2,y2,slope]) 
                    else:
                        continue # ignore parellel lines -> might be if the line finder was not good enough
        right_lines = np.array([[x] for x in right_line]) # convert the lis in ndarray
        left_lines =  np.array([[x] for x in left_line])
        classified_lines = [left_lines, right_lines]
        return classified_lines # ([x1_left, y1_left, x2_left, y2_left, slope_left],[x1_right, y1_right, x2_right, y2_right, slope_left])
```
* the extrapolateLines function takes the right lines, left lines, and also the height of the edges in the region of interests from vertices defined for the region of interests as input. Then it returns on single line for the left and antoher for the right.

```
    ### lane classifier function###
    def extrapolateLines(left_lines, right_lines, y_extrp):
        # find the average of of the left lines for having the starting point to extroplate to the targert point
        average_left_lines = np.mean(left_lines, axis=0) # left
        average_right_lines = np.mean(right_lines, axis=0) #right
   
        # extrapolation x = X1 + ((y-Y1)/(Y2-Y1))*(X2-X1) -> x = X1 + (y-Y1)/SLOPE
        # -> left lane line
        y1_endLeft = y_extrp[0][0] # end point to extrapolate from the vertices of the region restriction algorithm
        x1_endLeft = int(average_left_lines[0][0] + (y1_endLeft-average_left_lines[0][1])/average_left_lines[0][4]) # convert to int the next closed possible value
        y2_endLeft = y_extrp[0][1] # end point to extrapolate
        x2_endLeft = int(average_left_lines[0][2] + (y2_endLeft-average_left_lines[0][3])/average_left_lines[0][4])
        exPnts_Left = [x1_endLeft, y1_endLeft, x2_endLeft, y2_endLeft ]
        # -> right lane line
        y1_endRight = y_extrp[1][0] # end point to extrapolate
        x1_endRight = int(average_right_lines[0][0] + (y1_endRight-average_right_lines[0][1])/average_right_lines[0][4])
        y2_endRight = y_extrp[1][1] # end point to extrapolate
        x2_endRight = int(average_right_lines[0][2] + (y2_endLeft-average_right_lines[0][3])/average_right_lines[0][4])
        exPnts_Right = [x1_endRight, y1_endRight, x2_endRight, y2_endRight]

        return [[exPnts_Left], [exPnts_Right]]
```
#### 6. Draw the single lines

And finally it comes to the easiest part that shows the result on the image/video. Here two helper functions are used as follows:
```
    ext_line_image = draw_lines(original_image, ext_lines) # draw the single lines on a blank image
    final_image = weighted_img(ext_line_image, original_image) #combine blank image including the lines with original image
```
### 2. Identify potential shortcomings with your current pipeline

I want to know why I am having traceback error, with the last optional video. I may need help :). I have tried everything but I was not able to find the reason.


