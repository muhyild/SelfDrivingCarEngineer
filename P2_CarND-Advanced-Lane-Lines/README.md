## Writeup Project: Advanced Lane Finding Project

In the previos preject very fundamental methods were used to determine the geometry of the road. However it was not enough to consider the complex road conditions, for example, curved roads, different lane line colors and so on. With this poject the computer vison algorithms are implemented to find out a precise road geometry, which is one of the most and core topic , since the car can only drive autonomously by knowing the envireoment geometry and acting to it.

For this purpose I have implemented the computer vision algorithms in this project. The goals of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The used libraries are liste as follows:
```python
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
import cv2
import glob
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML
```


[//]: # (Image References)

[image1]: ./WriteUp/undistort_cameracalibration_WriteUp.PNG "Undistorted"
[image2]: ./WriteUp/undist_testRoad_WU.PNG  "Road Undistorted"
[image3]: ./WriteUp/thresholding_WU.PNG  "Thresholding"
[image4]: ./WriteUp/combinedThresholding_WU.PNG  "Combined Thresholding"
[image5]: ./WriteUp/perspectiveTransform_WU.PNG  " Perspective Transform"
[image6]: ./WriteUp/histogram_WU.PNG  " histogram "
[image7]: ./WriteUp/slidingwindow_WU.PNG  " Perspective Transform"
[image8]: ./WriteUp/form2.PNG 
[image9]: ./WriteUp/curvatureVehPos_WU.PNG 
[image10]: ./WriteUp/result_WU.PNG 


Now let's continue to go through each steps to see how I implemented my project.

### 1. Compute the camera calibration matrix and distortion coefficients

The most of the cameras we are using in recent years, have distortions because of the lenses, the camera producers use. The distorion occurs as we transform image from 3D in the real world into 2D image in the image plane. This distortion makes difficult to describe the road geometry,thus, to detect the object in the envireonment. There are two common distortions, which are radial distortion and tangential distortion. 

In order to get good result, the fist step is to undistort the image, which includes basically two prepareation steps. The first step is to set object points (3D points in the real world plane) and to set the imagepoints (2D points in the image plane). We use bunch of chessboard images to find image points by finding corners using built-in OpenCV function ` findChessboardCorners`, which provides the corners that are basically our 2D points in the image plane. And also, as I said before we need to prepare also the object points that are basically our 3D points (cornes on the real image through the rows and colums). For that none of calculation is required, but they should be prepared. 

As soon as the first step is done, now it is time for the second step which is to calibrate the camera. By using the image points and object points as input on the build-in OpenCV function `calibrateCamera` the required outputs are generated, which are camera matrix, distortion coefficients, rotation and translation vectors. 

I have implemented the following function to perform these two steps as follows:

```python 
def calibCamera():
    images = glob.glob('camera_cal/calibration*.jpg') # access all calibration images   
    # set the chessboard size
    nx = 9
    ny = 6
    # Store the 3D and 2D points
    objpoints = [] # 3D points in the real world plane
    imgpoints = [] # 2D points in the image plane
    # prepare the object points
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) 
    for image in images:
        img = mpimg.imread(image) # read the image        
     # Use cv2.COLOR_RGB2GRAY if you've read in an image using mpimg.imread(). 
     # Use cv2.COLOR_BGR2GRAY if you've read in an image using cv2.imread().
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # convert the image to gray color space
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None) # find the chessboard corners 
        if ret:
            imgpoints.append(corners) # fill the image corner points
            objpoints.append(objp) # fill the object points in real world plane
            # draw the corners
            img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
    #calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    return mtx, dist
```

This function turns out two outputs, which are camera matrix `mtx` and distortion coefficients `dist`. They will be used as input in the next section.  

### 2. Apply a distortion correction 

Now it is time to proceed to the next step, which is to undistort the images using the camera matrix `mtx` and the distortion coeefficients `dist`. 

I have written the following function to undistort the given image.

```python
def undistImage(img, mtx, dist):
    #undistort the given image
    undistImg = cv2.undistort(img, mtx, dist, None, mtx)
    return undistImg
```

This function takes the given image, camera matrix and distortion matrix as inputs, and returns the undistorted image. The example input and output images are in the Figure below.

![alt text][image1]

This funciton has been also applied to the one of the test road image, and the resulting undistorted image as follows:

![alt text][image2]


### 3. Create a thresholded binary image

In the preivous project we have used the built-in function `Canny` to detect the edges using prespecified the minimum and maximim threshold. However for the advanced line finding it is not enough. Therefore I have used different thresholding techniques in order to detect the edges. The techniques which I have used are as follows:

    - Gradient Thresholding using Sobel through x axis and y axis
    - Magnitude Gradient Thresholding
    - Direction of the gradient Thresholding
    - using HLS color spaces Thresholding 
    
I have tried all above mentioned methods alone to find out which one is better than other. At the end i have written an algorithm to use the combination of them. The results are in the following Figure.

```python
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold  
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    if orient == 'x':
        gradx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        abs_gradx = np.sqrt(np.power(gradx,2))
        scaled_grad= np.uint8(255*abs_gradx/np.max(abs_gradx))
        grad_binary = np.zeros_like(scaled_grad)
        grad_binary[(scaled_grad >= thresh[0]) & (scaled_grad <= thresh[1])] = 1
    elif orient == 'y':    
        grady = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_grady = np.sqrt(np.power(grady,2))
        scaled_grad= np.uint8(255*abs_grady/np.max(abs_grady))
        grad_binary = np.zeros_like(scaled_grad)
        grad_binary[(scaled_grad >= thresh[0]) & (scaled_grad <= thresh[1])] = 1   
    return grad_binary

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    gradx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    grady = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients
    abs_grad = np.sqrt(np.power(gradx,2)+np.power(grady,2))
    # Create a binary mask where direction thresholds are met
    scaled_sobel = np.uint8(255*abs_grad/np.max(abs_grad))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1  
    return mag_binary

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    gradx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    grady = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients
    abs_gradx = np.sqrt(np.power(gradx,2))
    abs_grady = np.sqrt(np.power(grady,2))
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dir_grad = np.arctan2(abs_grady,abs_gradx)
    # Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(dir_grad)
    dir_binary[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1
    return dir_binary

def hls_thresh(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h = hls[:,:,0]
    l = hls[:,:,1]
    s = hls[:,:,2]
    thresh_h = (10,60)
    thresh_l = (200,255)
    thresh_s = (90,255)
    h_binary = np.zeros_like(h)
    h_binary[(h >= thresh_h[0]) & (h <= thresh_h[1])] = 1
    l_binary = np.zeros_like(l)
    l_binary[(l >= thresh_l[0]) & (l <= thresh_h[1])] = 1
    s_binary = np.zeros_like(s)
    s_binary[(s >= thresh_s[0]) & (s <= thresh_s[1])] = 1
    
    combined_hs = np.zeros_like(h)
    #combined_hs[((h_binary == 1) | (s_binary == 1)) | (l_binary==1) ] = 1
    combined_hs[((s_binary == 1)) | (l_binary==1) ] = 1
    #combined_hs[((h_binary == 1) | (s_binary == 1))] = 1
    #combined_hs[((s_binary == 1))] = 1
    return combined_hs
```

![alt text][image3]

As I said before, because I am not satisfied of the above results, I have implemented the combination of all methods. The function I have written and the result are as follows:  

```python
def combinedThresh(img):
    kernel_size = 9
    grad_binaryX = abs_sobel_thresh(img, orient='x', sobel_kernel=kernel_size,  thresh=(20, 100))
    grad_binaryY = abs_sobel_thresh(img, orient='y', sobel_kernel=kernel_size, thresh=(20, 100))
    mag_binary = mag_thresh(img, sobel_kernel=kernel_size, thresh=(50, 100))
    dir_binary = dir_thresh(img, sobel_kernel=kernel_size, thresh=(0.7, 1.3))
    hls_binary = hls_thresh(img)
    
    combined = np.zeros_like(dir_binary)
    combined[(grad_binaryX == 1) | ((mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1) ] = 1
    return combined
```
![alt text][image4]

As it is clearly seen, the combined thresholding algorithm provides us much better edge detection result. Now it is time to move the next section.

By the way it has taken a lot time to find the best version :). 

### 4. Apply a perspective transform

In order to calculate the lane curvature the prespective transform is needed to apply, which basically trasnforms the image in a bird-eye view (top-down view). The basic idea is to grab four points forming rectangular area from the source image, and also four points forming rectangular area that will be destionation points. The selecting procedure was intuitive in my case, I have tried may points combination to find the best source and destination points.

For the perspective transform I have implemented another function, taking an image as input, then delivers the warped image, trasnfomation matrix, and inverse trasnformation matrix for getting image back later. The function and the result are as follows. I have used the starting point from the gevien write_up, and the I have changed it with respect to resulting image.


```python
def perspectiveTransform(img):

    src = np.float32([
        [700, 450], # top right
        [1100, img.shape[0] ], # bottom right
        [180, img.shape[0]], # bottom left
        [600, 450]]) # top left   
            # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    dst = np.float32([
        [980,0],             # top right
        [980,img.shape[0]],  # bottom right
        [300,img.shape[0]],  # bottom left
        [300,0]])            # top left  
    '''
    src = np.float32([
        [680, 460],            # top right
        [1050, img.shape[0] ], # bottom right
        [200, img.shape[0]],   # bottom left
        [550, 460]])           # top left   
    #  define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    dst = np.float32([
        [900,0],             # top right
        [980,img.shape[0]],  # bottom right
        [440,img.shape[0]],  # bottom left
        [410,0]])            # top left  
    '''
    # get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst) 
    # get inverse M, the transform matrix, to transform the perspective image back to original image
    M_inv = cv2.getPerspectiveTransform(dst, src)
    # warp an image to a top-down view
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped, M, M_inv
```

![alt text][image5]

It is important to mention at this point that I used the combined thresholded image as input from the previous step. The result looks very good im my opinion and we can see the curvature from top. Let's continue to dive more with next steps.

### 5. Detect lane pixels and fit to find the lane boundary 

It is now time to detect the lane lines pixels and fit them into the polynom which generates the curve line that will help to calculate the curvature, after applying calibration, thresholdingm and a perspective transform to a road image. Here the first step is to decide which line is the right and which one is the left using histogram which shows where the binary activations occur across the image. With this histogram we are adding up the pixel values along each column in the image. The result of histogram is as follows:
```python 
histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
plt.plot(histogram)
```
![alt text][image6]

If the figure in previous section and the figure in this section are compared, one can see that the histogram value is getting high where the lane lines occured. 

The next point is to use sliding windows moving upward in the image to determine where the lines go. For this purpose I have written three functions to implement my algorithm as folows:

```python
def find_lane_pixels(binary_warped):
     # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
   
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
   
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
          
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
           
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(leftx, lefty, rightx, righty):   
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit

def drawImage(binary_warped, left_fit, right_fit, out_img):  
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    plt.imshow(out_img.astype('uint8'))
    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    return out_img
```
The resulting image is as follows:

![alt text][image7]

With this result, we can now calculate the curvature in the next section.

### 6. Determine the curvature of the lane and vehicle position with respect to center

In this section the radius of the curvature is calculated. I have implemented the following formulas to calculate the curvature:

![image.png](attachment:image.png)
![alt text][image8]

As soon as we find the left and right lines we can calculate the center of the lane, then we can use the middle of the image on the axis and the center of the lines to be able to calculate the vehicle position with respect to the center of the line. This idea is also implemented into the function given below.

So far only the lane pixels' values are used for all calculation, but in the real world it needs one mor transformation, which is basically the conversion of the `pixels` to the SI `meter`. For that puprpose as we have learned in the class, I had efined the conversion coeefficients `ym_per_pix = 30/720`, 30 meters per 720 pixels, in vertical direction, and `xm_per_pix = 3.7/700` , 3.7 meters per 700 pixels, in horizantal direction. Theseb conversion is used for both calculation of radius of curve and vehicle position.

For the vehicle position calculation I took the center of the lane as reference with assuming that the vehicle's position is on the middle of the horizantal axis, and then I have calculated the center of the found left and right line. Then I checked how big is the offset between these two.

I have implemented the following function to calculate the curvature and the vehicle position with respect to the center.

```python
def cal_Curvature_VehPos(binary_warped):
    '''
    Calculates the curvature of polynomial functions and vehicle position with respect to center in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Find lane pixels and fit in polynom to find polynom's coeefficents
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)  
    left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty)

    # Fit new polynomials to x,y in world space    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty    
    
    ##### Utilize `ym_per_pix` & `xm_per_pix` here #####
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = binary_warped.shape[0]
    
    # Implement the calculation of R_curve on the left and on the right (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    curvature = (left_curverad + right_curverad)//2
    
     # Calculate left and right line positions at the bottom of the image
    left_line_x_pos = (left_fit[0]*(y_eval**2))+(left_fit[1]*y_eval)+left_fit[2]
    right_line_x_pos = (right_fit[0]*(y_eval**2))+(right_fit[1]*y_eval)+right_fit[2]

    mid_lanes_x_pos = (left_line_x_pos + right_line_x_pos)//2 # calculate the average curvature
        
    x_mid = binary_warped.shape[1]//2
        
    vehPos_offset = (mid_lanes_x_pos- x_mid) * xm_per_pix # find the vehicle position on the real world
    
    return ploty, left_fitx, right_fitx, left_curverad, right_curverad, curvature, veh_pos
```
This function takes the warped image as input and returns the curvature and vehicle position. The result for the above given warped image is as below:

![alt text][image9]


### 7. Warp the detected lane boundaries back onto the original image

This section enables us to trasnform the image back to the original view and write all information on it , which are the radius of the curvature and the vehicle position with respect to the center. I have written the following function to implement my algorithm as follows.

```python
def pipeline(img):
    mtx, dist = calibCamera()
    undistImg = undistImage(img, mtx, dist)
    
    combined = combinedThresh(img)
    
    binary_warped, M, M_inv = perspectiveTransform(combined)
    ploty, left_fitx, right_fitx, left_curverad, right_curverad, curvature, veh_pos = cal_Curvature_VehPos(binary_warped)
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undistImg, 1, newwarp, 0.3, 0)
    cv2.putText(result,'Radius of Curve in m: '+str(curvature)[:6],(350,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255),2,cv2.LINE_AA)
    cv2.putText(result,'VehPos Offset to Center in m: '+str(veh_pos)[:6],(350,150), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(255,255,255),2,cv2.LINE_AA)
    return result
``` 
The resulting image is given in Figure below:

![alt text][image10]

It looks nice, isn't it? :)

### 8. Video Clip Pipeline

The last step is to put all above into a pipeline which works not only on image, but also on video. As we know, the video is basically a set of image frame. However we need to track some information from the previous frame like the last lane line pixels. As given in Tips and Tricks for the project, i have decided to implement the class with some important attributes. 

I need to calculate some section only once, for example, camera calibration matrix and distortion matrix, since the same calibration chessboard images are used all the time. And also I need to calculate the initial lane line pixels at the begining to store them later as the last pixel point for the next calculations. 

I have implemented the following Class with attributes and one static method to integrate my all functions to process the given video.

```python
class Line():
    def __init__(self):
        # if it is first Run
        self.firstRun = True # to calculate the initial line pixels only once   
        self.cameraCal = True # to calibrate camera only once 
        self.right_fit=[np.array([False])] # store preivous right line pixels
        self.left_fit=[np.array([False])]  # store preivous left line pixels
        self.mtx = [np.array([False])]  # store the camera calibration matrix after the first run to use in next runs
        self.dist = [np.array([False])] # store the distortion coeeficient matrix after the first run to use in next runs
    @staticmethod
    def vidPipeline(img):
        if Line.cameraCal == True:
            print('First run/frame: Camera Calibration Done ')
            mtx, dist = calibCamera() #done only once
            Line.mtx = mtx 
            Line.dist = dist
            Line.cameraCal = False 
        else:
            mtx = Line.mtx
            dist = Line.dist
    
        undistImg = undistImage(img, mtx, dist) # undistort the image

        combined = combinedThresh(img) # calculate the combined thresholding : hls, sobel-,magnitude,direcetional

        binary_warped, M, M_inv = perspectiveTransform(combined) # genrater wapred image, and calculate transfromation matrix
        
        if Line.firstRun == True: 
            print('First run/frame: Starting Lane Pixels Calculated ')
            leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
            left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty)
            Line.right_fit = right_fit
            Line.left_fit = left_fit
            Line.firstRun = False
        else:
            
            left_fit=Line.left_fit # take the previos lane pixel to continue
            right_fit=Line.right_fit

            margin = 100
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                            left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                            left_fit[1]*nonzeroy + left_fit[2] + margin))).nonzero()[0]
            right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                            right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                            right_fit[1]*nonzeroy + right_fit[2] + margin))).nonzero()[0]

            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]      

            # Fit a second order polynomial to pixel positions in each lane line
            left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty)
                        
            Line.right_fit = right_fit # store the new polynaomial pixel position for the next frame
            Line.left_fit = left_fit

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty    

        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        # Measure the curvature in meter
        left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = binary_warped.shape[0]

        # Implement the calculation of R_curve on the left and on the right (radius of curvature)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        curvature = (left_curverad + right_curverad)//2

        # Calculate left and right line positions at the bottom of the image
        left_line_x_pos = (left_fit[0]*(y_eval**2))+(left_fit[1]*y_eval)+left_fit[2]
        right_line_x_pos = (right_fit[0]*(y_eval**2))+(right_fit[1]*y_eval)+right_fit[2]

        mid_lanes_x_pos = (left_line_x_pos + right_line_x_pos)//2 # calculate the average curvature
        
        x_mid = binary_warped.shape[1]//2
        
        vehPos_offset = (mid_lanes_x_pos- x_mid) * xm_per_pix # find the vehicle position on the real world

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, M_inv, (img.shape[1], img.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undistImg, 1, newwarp, 0.3, 0)
        cv2.putText(result,'Radius of Curve in m: '+str(curvature)[:6],(350,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255),2,cv2.LINE_AA)
        cv2.putText(result,'VehPos Offset to center in m: '+str(vehPos_offset)[:6],(350,150), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(255,255,255),2,cv2.LINE_AA)
        return result
```

### 9. Result

Here's a [link to my video result](./output_videos/output_project_video.mp4). I should say that the result looks nice by detecting the lanes correctly during the entire video.


### 10. Discussion

For the other given two challenging videos my algorithm was not good enough, especially, for hard_challenge_video. I am going to find out another approaches to make my algorithm more robust against the different road conditions, for example, for very bright road condition better edge detection algorithm, for very curvy roads better lane pixels' fiting polynoms and so on. I am not expert but as soon as I learn neural networks, I think it would also help more to find the edges much more efefective and precise. I am also open for any suggestions to make my code alse better, since I am doing this nonedegree for my hobby and interest, not to find a job :)

### References

- Udacity self driving cars nonedegree 
- https://www.intmath.com/applications-differentiation/8-radius-curvature.php
- https://docs.opencv.org/2.4/index.html

#### Important folders
I have put the results of the images and the video into the following folders:

[link to my image results](./output_images)

[link to my video results](./output_videos)

[link to my images for WriteUp](./WriteUp)
