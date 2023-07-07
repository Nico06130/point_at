#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
import cv2
import numpy as np
from point_at_srvs.srv import PointedId

class CallPointAtService:

    def __init__(self):

        rospy.init_node("point_at_service_call",anonymous=True)

    def call(self):

        rospy.wait_for_service('point_at_service')

        try:

            rospy.loginfo("Attente du service Point At")

            get_service = rospy.ServiceProxy('point_at__service',PointedId)  
            id_pointe = get_service().pointed_msg
            rospy.loginfo(id_pointe)

        except rospy.ServiceException as e:

                print("Erreur dans l'appel du service: %s"%e)


if __name__=="__main__":

    if __name__ == '__main__':

        a=CallPointAtService()
        a.call()

        while not rospy.is_shutdown():

            a.test()
            rospy.sleep(1)