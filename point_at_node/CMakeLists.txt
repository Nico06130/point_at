cmake_minimum_required(VERSION 3.0.2)
project(point_at_node)

## Add support for C++11, supported in ROS Kinetic and newer
# add_definitions(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    rospy
    cv_bridge
    std_msgs
    sensor_msgs
    point_at_msgs
    point_at_srvs
)


catkin_package(
   # INCLUDE_DIRS include
#  LIBRARIES people_face_identification
 # CATKIN_DEPENDS message_runtime
#  DEPENDS system_lib
)


include_directories(
  ${catkin_INCLUDE_DIRS}
)