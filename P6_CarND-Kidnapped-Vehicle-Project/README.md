# Kidnapped Vehicle Project: Particle Filter Localization Code

## 1. Introduction

The localization of the self-driving cars are one of the most challenging problem. In this project I programmed a two-dimensional Particle Filter in C++ pragramming language to solve the localization problem.

I finished this project using Workspace, which was faster for me to proceed. Therefore I did not perform any installation of simulation and socket.

[image1]: ./start.png "compile and run"
[image2]: ./result.png "the process of tuning parameter"


## 2. Compiling codes and running simulation

The main program can be built and run by doing the following from the project top directory. 

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./particle_fileter
6. Run simulation by pressing simulation button at the bottom right

I have modified the following file, which are provided by Udacity in src folder :

- particle_filter.cpp: the logic of the particle controller

INPUTS: values provided by the simulator to the c++ program

- ["map data"] => position in x and y coordinate, landmark id (noisy GPS data)
- ["control data"] => vehicle speed and vehicle yaw rate
- ["observation data"] => the landmark positions in x,y coordinate whihch are closed to the vehicle with respect to the vehicle which should be converted to map coordinate system

I have compiled my program and run in the simulator without any error as shown in screenshot below

![alt text][image1] 

---
## 3. Result

I have evaluated my simulation results based on the rubric points provided in porject specification. The results are as follows:

- [Q1] Accuracy: Does your particle filter localize the vehicle to within the desired accuracy? 
- [A1] Yes, it was able to localize the vehicle within desired accuracy. At the end the simulation ends with "Success! Your particle filter passed!" as shown in screenshot below

![alt text][image2] 

- [Q2] Peformance: Does your particle run within the specified time of 100 seconds?
- [A2] Yes, it has finished within specified time, which was 59.94s less than required 100s. It is also shown in screenshot provided above.

- [Q3] General: Does your code use a particle filter to localize the robot?
- [A3] Yes, it uses the particle filter to localize the robot. But there are many other ways to do it, eg. SLAM, Extended Kalman Filter ...

Attached Files: 
- />home>workspace>CarND-Kidnapped-Vehicle-Project>start.png
- />home>workspace>CarND-Kidnapped-Vehicle-Project>result.png