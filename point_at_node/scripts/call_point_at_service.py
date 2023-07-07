#!/usr/bin/python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import mediapipe as mp

class TestNode:

    def __init__(self):

        self.image_sub = rospy.Subscriber('/kinect2/hd/image_color', Image, self.callback)
        self.point_at_pub = rospy.Publisher('mediapipe_test',Image,queue_size=10)
        self.bridge = CvBridge()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        
    def callback(self,req):
	
        frame = self.bridge.imgmsg_to_cv2(req, "bgr8")
        
        with self.mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.1,
                min_tracking_confidence=0.1,
                max_num_hands=1) as hands:

            results = hands.process(frame)

            if results.multi_hand_landmarks:

                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    rospy.loginfo(hand_landmarks)

            else:

                rospy.loginfo("pas de main dans le champ")
                    
        #cv2.imshow("test",frame)
        self.point_at_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        

    def run(self):

        rospy.init_node('point_at_detection')
        
        rospy.spin()
        
if __name__ == '__main__':
    try:
        node = TestNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
