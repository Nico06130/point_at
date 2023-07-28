## Description

This package is used to identify objects pointed by a user in 3D.

It is a ROS service that returns the ID of the corresponding boundingbox  of the object pointed.
To do so, hand landmarks are generated using MediaPipe. The 2D coordinates of the top and the bottom keypoint of the index are extracted. Their depth is calculated using the RealSense API. Then a Yolov8 service implemented in the yolov8_ros package is called. This service returns the 2D coordinates of the bounding boxes detected. Then the depth of the center of the boundingboxes are calculated using the RealSense API. Finally, the colinearity of the vectors bottom finger/top finger and top finger/center boundingbox is checked, with an error margin. If it is colinear, then the id of the boundingbox of the object detected is returned.
## Architecture
![image](https://github.com/Nico06130/point_at/assets/78531005/08199b1a-8e1a-47c0-88ac-9b1552b4bb92)

## Dependencies

    -ROS
    -mediapipe
    -pyrealsense2
    -opencv
    -numpy
    -yolov8_ros_msgs package


## Execution

In a terminal:        

    rosrun point_at_node call_point_at_service.py

In an other terminal: 

    rosrun point_at_node point_at_realsense.py


## Outputs

Displays frame with hand detection, bounding boxes and fills the bounding box when the object is pointed in a topic called /point_at_frame
Service returns PoseStamped() of the pointed object.

![image](https://github.com/Nico06130/point_at/assets/78531005/552adbfc-445f-47a3-8fab-0cd12ab96142)

## Imporvements

Mediapipe is running on the CPU when using Python. The project can be transferred to C++ in order for Mediapipe to run on the GPU to enhance performances.



