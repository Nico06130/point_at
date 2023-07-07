# Point at using Realsense D455

This package is used to identify objects pointed by a user in 3D.

It is a ROS service that returns the ID of the corresponding boundingbox  of the object pointed.
To do so, hand landmarks are generated using MediaPipe. The 2D coordinates of the top and the bottom keypoint of the index are extracted. Their depth is calculated using the RealSense API. Then a Yolov8 service implemented in the yolov8_ros package is called. This service returns the 2D coordinates of the bounding boxes detected. Then the depth of the center of the boundingboxes are calculated using the RealSense API. Finally, the colinearity of the vectors bottom finger/top finger and top finger/center boundingbox is checked, with an error margin. If it is colinear, then the id of the boundingbox of the object detected is returned.


Requirements:

-ROS
-mediapipe
-pyrealsense2
-opencv

Usage:

In a terminal:        rosrun point_at_node call_point_at_service.py
In an other terminal: rosrun point_at_node point_at_realsense.py


