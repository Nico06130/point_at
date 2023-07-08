#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32, UInt16
import cv2
import numpy as np
from point_at_srvs.srv import Pointed

class CallPointAtService:

    def __init__(self):

        rospy.init_node("point_at_service_call",anonymous=True)
        self.yolo_model_name = "yolov8m.pt" 
        self.yolo_class = [39]

    def call(self):

        rospy.wait_for_service('point_at_service')

        try:

            rospy.loginfo("Attente du service Point At")

            get_service = rospy.ServiceProxy('point_at_service',Pointed)  
            pose_detecte = get_service(self.yolo_model_name,self.yolo_class).pointed_msg
            rospy.loginfo(pose_detecte)

        except rospy.ServiceException as e:

                print("Erreur dans l'appel du service: %s"%e)


if __name__=="__main__":

    if __name__ == '__main__':

        a=CallPointAtService()

        while not rospy.is_shutdown():

            a.call()
            rospy.sleep(1)