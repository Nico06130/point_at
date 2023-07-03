#!/usr/bin/python3

import rospy
import mediapipe as mp
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np

class PointAtNode:

    def __init__(self):
        self.image_sub = rospy.Subscriber('/kinect2/hd/image_color', Image, self.callback)
        self.bridge = CvBridge()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_utils_styles
        self.mp_hands = mp.solutions.hands

    def callback(self, req):
        frame = self.bridge.imgmsg_to_cv2(req, "bgr8")
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        with self.mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                max_num_hands=1) as hands:

            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Print hand coordinates
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        z = int(landmark.z * 1000)  # Scale z-coordinate for better visibility
                        rospy.loginfo(f"Hand Landmark {idx}: ({x}, {y}, {z})")

        cv2.imshow("Point at", frame)
        cv2.waitKey(1)


    def run(self):
        rospy.init_node('point_at_detection')
        rospy.spin()

if __name__ == '__main__':
    try:
        node = PointAtNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
