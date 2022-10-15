# Capstone Project: System Integration

## 1. Introduction

As a last step of the self driving cars project, the system integration is the core of whole project. In this project I wrote/completed the python scripts to communicate the modules to work with each other to be able to drive the car autonomusly in Carla's simulation provided by Udacity.

I finished this project using Workspace, which was faster for me to proceed. Therefore I did not perform any local installation of simulation and socket.

[image1]: ./graphNode.png "compile and run"
[image2]: ./start1.PNG "the start process 1"
[image3]: ./start2.PNG "the start process 2"
[image4]: ./start3.PNG "the start process 3"
[image5]: ./start4.PNG "the start process 4"
[image6]: ./run1.PNG "the run process 1"
[image7]: ./run2.PNG "the run process 2"

## 2. Compiling codes and running simulation

The main program can be built and run by doing the following from the project top directory. 

1. cd CarND-Capstone
2. pip install -r requirements.txt
3. cd ros
4. catkin_make
5. source devel/setup.sh
6. roslaunch launch/styx.launch
7. Run simulation by pressing simulation button at the bottom right

I have modified the following file, which are provided by Udacity in src folder :

- tl_detector.py: This package contains the traffic light detection node. This node takes in data from the /image_color, /current_pose, and /base_waypoints topics and publishes the locations to stop for red traffic lights to the /traffic_waypoint topic. The /current_pose topic provides the vehicle's current position, and /base_waypoints provides a complete list of waypoints the car will be following.

- waypoint_updater.py:  This package contains the waypoint updater node. The purpose of this node is to update the target velocity property of each waypoint based on traffic light and obstacle detection data. This node will subscribe to the /base_waypoints, /current_pose, /obstacle_waypoint, and /traffic_waypoint topics, and publish a list of waypoints ahead of the car with target velocities to the /final_waypoints topic.

- twist_controller.py: Carla is equipped with a drive-by-wire (dbw) system, meaning the throttle, brake, and steering have electronic control. This package contains the files that are responsible for control of the vehicle: the node dbw_node.py and the file twist_controller.py, along with a pid and lowpass filter that you can use in your implementation. The dbw_node subscribes to the /current_velocity topic along with the /twist_cmd topic to receive target linear and angular velocities. Additionally, this node will subscribe to /vehicle/dbw_enabled, which indicates if the car is under dbw or driver control. This node will publish throttle, brake, and steering commands to the /vehicle/throttle_cmd, /vehicle/brake_cmd, and /vehicle/steering_cmd topics.


The system architecture design is provided as below which is taken from Udacity's project instructure.

![alt text][image1] 

---
## 3. Result

I have evaluated my simulation results based on the rubric points provided in porject specification. The results are as follows:

- [Q1] Running the code: The code is built successfully and connects to the simulator.
- [A1] Yes, the code is built successfully and it connected to the simulator without any errors as shown in screenshots as below.

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

- [Q2] Control and Planning: Waypoints are published to plan Carla’s route around the track.
- [A2] Waypoints has been published to /final_waypoints to plan the vehicle’s path around the track. No unnecessary moves (excessive lane changes, unnecessary turning, unprompted stops) occured. As in the Path Planning project, acceleration did not exceed 10 m/s^2 and jerk did not exceed 10 m/s^3. The top speed of the vehicle was set to the km/h velocity set by the velocity rosparam in waypoint_loader. Controller commands were published to operate Carla’s throttle, brake, and steering. dbw_node.py has been implemented to calculate and provide appropriate throttle, brake, and steering commands. The commands were published to /vehicle/throttle_cmd, /vehicle/brake_cmd and /vehicle/steering_cmd, as applicable.I have added some screenshots during simulation to be able to demonstrate that this rubric point was also satisfied.

![alt text][image6]
![alt text][image7]

- [Q3] Meets Specifications: Successfully navigate the full track more than once.
- [A3] The car was able to navigate the full track more than once without any unwanted actions.