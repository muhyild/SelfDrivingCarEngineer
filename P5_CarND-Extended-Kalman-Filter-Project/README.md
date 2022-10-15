# Extended Kalman Filter Project Starter Code

## 1. Introduction

In this project I programmed a kalman filter in C++ pragramming language to estimate the state of a moving object of interest with noisy lidar and radar measurements.I have obtained obtaining RMSE values that are lower than the tolerance outlined in the project rubric, which are:
- px, py, vx, and vy RMSE should be less than or equal to the values [.11, .11, 0.52, 0.52].

I finished this project using Workspace, which was faster for me to proceed. Therefore I did not perform any installation of simulation and socket.

[image1]: ./Dataset1_both.PNG "Dataset1 with Radar and Laser"
[image2]: ./Dataset1_onlyLaser.PNG "Dataset1 with only Laser"
[image3]: ./Dataset1_onlyRadar.PNG "Dataset1 with only Radar"
[image4]: ./Dataset2_both.PNG "Dataset2 with Radar and Laser"
[image5]: ./Dataset2_onlyLaser.PNG "Dataset2 with only Laser"
[image6]: ./Dataset2_onlyRadar.PNG "Dataset2 with only Radar"

## 2. Compiling codes and running simulation

The main program can be built and run by doing the following from the project top directory. 

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./ExtendedKF
6. Run simulation by pressing simulation button at the bottom right

I have modified the following files,which are provided by Udacity in src folder :

- main.cpp: Interfaces with the simulator using uWebSocketIO to recieve measurements and transmit estimates and RMSE values output by the EKF.
- FusionEKF.cpp: Instantiates the EKF, initial object state (x), covariance (P) and calls the appropriate Predict() or Update() methods depending on whether LIDAR or RADAR measurements are being processed.
- kalman_filter.cpp: Implements the Predict(), Update() and UpdateEKF() methods for prediction, LIDAR updates & RADAR updates respectively.
- tools.cpp: Contains tools for calculating the Jacobian and RMSE values.

Here is the main protocol that main.cpp uses for uWebSocketIO in communicating with the simulator.

INPUT: values provided by the simulator to the c++ program

["sensor_measurement"] => the measurement that the simulator observed (either lidar or radar)


OUTPUT: values provided by the c++ program to the simulator

["estimate_x"] <= kalman filter estimated position x
["estimate_y"] <= kalman filter estimated position y
["rmse_x"]
["rmse_y"]
["rmse_vx"]
["rmse_vy"]

---
## 3. Result

The Extended Kalman Filter which I have implemented uses the sates
- px : position of the car in x coordinate -> initialized with measurement provided from radar and laser sensor data
- py : position of the car in y coordinate -> initialized with measurement provided from radar and laser sensor data
- vx : velocity of the car in x coordinate -> initialized with constant 0 for laser, with rhodot data for radar 
- vy : velocity of the cat in y coordinate -> initialized with constant 0 for laser, with rhodot data for radar 

The covariance matrix of the EKF `P` (4x4 matrix, for 4 states) is diagonally initialized with a variance of `1,1,500,500` that presents the high uncertainty in the initial velocity in x and y coordinate, and high certainty in the initial position in x and y coordinate.

I have tested my EKF with both datasets, and also with only laser data and with only radar data. The following screenshots shows the result of my algorithm on the simulator. 

The following image shows the result with `Dataset1 consisting both Radar and Laser sensors` data. This image shows good estimations of state for the vehicle with the final RMSE values over all timesteps which meets the required specification mentioned above. 
![alt text][image1] 

The following image shows the result with `Dataset1 consisting only Laser sensor` data. This image shows poor estimations of  the vehicle velocity compared to the result obtained using both sensors by looking at final RMSE values.
![alt text][image2] 

The following image shows the result with `Dataset1 consisting only Radar sensor` data. This image shows poor estimations of  the vehicle position compared to the result obtained using both sensors by looking at final RME values.
![alt text][image3]

The following image shows the result with `Dataset2 consisting both Radar and Laser` sensors' data.This image shows good estimations of state for the vehicle with the final RMSE values over all timesteps which meets the required specification mentioned above. 
![alt text][image4] 

The following image shows the result with `Dataset2 consisting only Laser sensor` data. This image shows poor estimations of the vehicle velocity compared to the result obtained using both sensors by looking at final RME values.
![alt text][image5] 

The following image shows the result with `Dataset2 consisting only Radar sensor` data.This image shows poor estimations of  the vehicle position compared to the result obtained using both sensors by looking at final RME values.
![alt text][image6]

The overall results show us that the combined sensor data will give much better estimation accuracy than using only one sensor data at a time. 
