# Project PID Controller

## 1. Introduction

In this project I programmed a PID controller in C++ pragramming language to drive the car in the given track safely. I

I finished this project using Workspace, which was faster for me to proceed. Therefore I did not perform any installation of simulation and socket.

[image1]: ./run.png "compile and run"
[image2]: ./pidtuning.png "the process of tuning parameter"


## 2. Compiling codes and running simulation

The main program can be built and run by doing the following from the project top directory. 

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./pid
6. Run simulation by pressing simulation button at the bottom right

I have modified the following files,which are provided by Udacity in src folder :

- PID.cpp: the logic of the PID controller is implemented here to compensate the error.
- PID.h: the parameters that are used in PID logic are modified.
- main.cpp: the PID parameters are tuned and called in this file.

INPUT: values provided by the simulator to the c++ program

["steering angle"] => steering angle
["cte"] => cross track error 
["speed"] => the speed of the vehicle

OUTPUT: values provided by the c++ program to the simulator

["controlled_output_steering_angle"] <= control output 
["constant_speed] <= speed 30 mph 

I have compiled my program and run in the simulator without any error. My program is able to keep the vehicle on the track.

![alt text][image1] 

---
## 3. State of art: PID

PID, which stands for Proportional, Integral, Derivative, control method is one of the most used control method in the industry. The implemenatation of the PID is straightforward and easy, but the most tricky part is to decide how to tune the coeeficients Kp (proportional), Ki( integral), Kd(derivative). 

The proportional term ($P_{out}$) results in the control output which is proportional to the error, which is the difference beetween calculated and measured value of the varible that need to be controlled (e.g. steering angle, speed, temparature ....). The proportional output of the controller to compensate the error (cte) is calculated as follows:

$ P_{out} = -Kp * cte $ 

here ($-$) means that it tries to reduce the error. 

The inegral term ($ I_{out}$) results in the control output which not only gets the current error but also the old errors into account, so that if the generated control input is not enough to compensate the error, the old past errors are also used to make the controlled output more precise and powerful for the future errors. It is calculated as follows:

$ I_{out} = - Ki * \int_0^t cte(t) dt$

The derivative coeefficents/term (Kd) results in the control output which uses the slope of the error over time. This part helps the system predict the system behaviour and helps to reduce the oscillation, thus improves the stability of the system. It is calculated as follows:

$ D_{out} = - Kd * \frac{dcet(t)}{dt}$

The total error for PID contoller is calculatead as follows:

$TotalError = P_{out} + I_{out} + D_{out}$

The hardest part of the implementation of this method is tuning the coeefficents Kp, Ki, and Kd. There are many different ways to tune them, for example, manual tuning(trial and error), twiddle, Zeigler-Nichols ....

I have chosen the manual tuning as a tuning method of my controller. I have followed the following approach: 

- Set all gains to 0.
- Set only Kp to 1, then decrease by a factor of 2 till reaching oscillation as less as possible 
- Set Kd to 1, and factor by 2 or multiply by two to get better settling time and more stable system
- Set Ki to about 0.1% of Kd, and play with the values around.
- Then check Kp if it can be reduced a bit more to reduce the agressive behavior of the controller.

I have recorded the process of tuning parameters, which is provided also in my program, as follows:

![alt text][image2] 

I had four different solutions at the end, which were working very well to keep the vehicle on the track. The possible combination of the PID coeeficients are provided as follows:

- Kp: 0.125, Ki: 0.0, Kd: 1.0
- Kp: 0.125, Ki: 0.001, Kd: 1.0
- Kp: 0.0625,Ki: 0.001, Kd: 2.0
- Kp: 0.0625,Ki: 0.003, Kd: 2.0

The result is recorde in the video and provided in project file. The name of the video file is `result.mp4`.