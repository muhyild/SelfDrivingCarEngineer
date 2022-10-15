# Highway Driving Project: Path Planning Code

## 1. Introduction

In this project I programmed a path planning algorithm in C++ pragramming language to drive a car on a highway built on simulator made by Udacity.

I finished this project using Workspace, which was faster for me to proceed. Therefore I did not perform any installation of simulation and socket.

[image1]: ./start.PNG "compile and run"
[image2]: ./result.PNG "Result 1"
[image3]: ./result2.PNG "Result 2"
[image4]: ./result3.PNG "Result 3"
[image5]: ./result4.PNG "Result 4"
[image6]: ./result5.PNG "Result 5"
[image7]: ./outline.PNG "Result 5"


## 2. Compiling codes and running simulation

The main program can be built and run by doing the following from the project top directory. 

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./path_planning
6. Run simulation by pressing simulation button at the bottom right

I have modified the following files,which are provided by Udacity in src folder :

- main.cpp: the logic of prediction of other cars in the traffic, behvaiour planning, and path planner

Here is the data provided from the Simulator to the C++ Program

#### Main car's localization Data (No Noise)

["x"] The car's x position in map coordinates

["y"] The car's y position in map coordinates

["s"] The car's s position in frenet coordinates

["d"] The car's d position in frenet coordinates

["yaw"] The car's yaw angle in the map

["speed"] The car's speed in MPH

#### Previous path data given to the Planner

//Note: Return the previous list but with processed points removed, can be a nice tool to show how far along
the path has processed since last time. 

["previous_path_x"] The previous list of x points previously given to the simulator

["previous_path_y"] The previous list of y points previously given to the simulator

#### Previous path's end s and d values 

["end_path_s"] The previous list's last point's frenet s value

["end_path_d"] The previous list's last point's frenet d value

#### Sensor Fusion Data, a list of all other car's attributes on the same side of the road. (No Noise)

["sensor_fusion"] A 2d vector of cars and then that car's [car's unique ID, car's x position in map coordinates, car's y position in map coordinates, car's x velocity in m/s, car's y velocity in m/s, car's s position in frenet coordinates, car's d position in frenet coordinates. 

#### Some Details

1. The car uses a perfect controller and will visit every (x,y) point it recieves in the list every .02 seconds. The units for the (x,y) points are in meters and the spacing of the points determines the speed of the car. The vector going from a point to the next point in the list dictates the angle of the car. Acceleration both in the tangential and normal directions is measured along with the jerk, the rate of change of total Acceleration. The (x,y) point paths that the planner recieves should not have a total acceleration that goes over 10 m/s^2, also the jerk should not go over 50 m/s^3. (NOTE: As this is BETA, these requirements might change. Also currently jerk is over a .02 second interval, it would probably be better to average total acceleration over 1 second and measure jerk from that.

2. There will be some latency between the simulator running and the path planner returning a path, with optimized code usually its not very long maybe just 1-3 time steps. During this delay the simulator will continue using points that it was last given, because of this its a good idea to store the last points you have used so you can have a smooth transition. previous_path_x, and previous_path_y can be helpful for this transition since they show the last points given to the simulator controller with the processed points already removed. You would either return a path that extends this previous path or make sure to create a new path that has a smooth transition with this last path.


---
## 3. Result

I have evaluated my result based on the rubric points provided by Udacity. Based on these given rubric points my result was successful. In the following section I will go through all rubric points and show the result accordingly. 

[R1] Compilation: the code compiles correctly?
Yes, it has compiled without any errors as shown in screenshot below.
![alt text][image1] 

[R2] Valid Trajectories: The car fulfills the following requirements which are shown the screenshots taken during simulation below.
- The car is able to drive at least 4.32 miles without incident
-- The car was able to drive almost 6.14 miles without any incident. The process along the way were shown in screenshots below
- The car drives according to the speed limit
-- The car follows always the speed limit shown in screenshot
- Max Acceleration and Jerk are not Exceeded
-- No warning were shown if the max acc. and Jerk are exceeded
- Car does not have collisions
-- The car did not have any collissions with other cars duirng simulation
- The car stays in its lane, except for the time between changing lanes
-- The car stays alway in its lane 
- The car is able to change lanes
-- The car wass able to change its lanes smoothly
![alt text][image3] 
![alt text][image2]
![alt text][image4]
![alt text][image5] 
![alt text][image6] 

[R3] Criteria: There is a reflection on how to generate paths
Based on the instructions during the course there are three main parts which I need to implement that are:
- Predictiion
- Behaviour Planning
- Trajectory Planning

![alt text][image7] 

Now I am going to explain a bit more about these three states and where I have implemented the in the given infrastructure by Udacity.

##### Prediction
In this part the aim is to use the sensor fusion data to be able to make a prediction about the environment in which the vehicle moving to prevent vehicle from collisions. The sensor fusion data format consist of  [ id, x, y, vx, vy, s, d] where the id is an unique identifier for the other car moving in the same highway. The x, y values are in global map coordinates, and the vx, vy values are the velocity components, also in reference to the global map. Finally s and d are the Frenet coordinates for that car. Based on these provided datas it is possible to check if there is a car blocking the fronth of my car, or if the right or left lane of my car is free to change its lane to pass the car which blocks the lane. if there is vehicle moving around closer than 30m are considered to be too close and thus dangerous. If the situation is dangerous then I have created three conditions to check if it is [car_too_close]. If it is [car_too_close] check if the left side of the car free [car_left_free] or [car_right_free]. This part was implemented between line [109] and line [145]. For more detail I have commented my codes in the source file.

##### Behaviour Planning 
Based on the result of the predicition state, the car now make a decision if the car is closed the car, or the left or the right side of the car is free to change the lane to pass the car which is moving slowly or blocking the fronth of my car. The car can only decelerate or accelerate 0.224mph at each speed point. I have implemented the lgoc for behavioural planinng between the line [148] and line [180].

##### Trajectory Planning
Based on the result of behevioral planning state which delivers the speed and the lane output, this part aims to generate 50 points of the path to follow, every time the car starts a new path. After generating the points, the spline fitting method is used to generate a continous path as given in project instructions. This part was implemented between line [182] and line [283] in main.cpp source file in the project.