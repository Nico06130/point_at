#!/usr/bin/python3

import rospy
import mediapipe as mp
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np

class PointAtNode:

    def __init__(self):

        self.image_sub = rospy.Subscriber('/kinect2/hd/image_color', Image, self.image_callback)
        self.point_cloud_sub = rospy.Subscriber('/kinect2/sd/points', PointCloud2, self.pointcloud_callback)
        self.bridge = CvBridge()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        #Service

        self.model_name = "yolov8"
        self.model_class = 0

    def image_callback(self, req):

        self.flux = req

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
                    # for idx, landmark in enumerate(hand_landmarks.landmark):
                    #     x = int(landmark.x * w)
                    #     y = int(landmark.y * h)
                    #     z = int(landmark.z * 1000)  # Scale z-coordinate for better visibility
                    #     rospy.loginfo(f"Hand Landmark {idx}: ({x}, {y}, {z})")

        cv2.imshow("Point at", frame)
        cv2.waitKey(1)

    def pointcloud_callback(self,data):

        self.pointcloud = data
    
    def get_3D_bboxes(self):

        rospy.wait_for_service('boxes3Dservice')

        try:
            get_boxes = rospy.ServiceProxy('boxes3Dservice',Boxes3D)
            bbox = get_boxes(self.model_name,self.model_class,self.flux,self.pointcloud)
        
        except rospy.ServiceException as e:

            print("Service call failed: %s"%e)

    def are_collinear(point1, point2, point3, tolerance):
        """
        Vérifie si trois points sont approximativement collinéaires dans l'espace.
        """
        # Calcul des vecteurs formés par les points

        vector1 = (point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2])
        vector2 = (point3[0] - point1[0], point3[1] - point1[1], point3[2] - point1[2])

        # Calcul des produits en croix des vecteurs
        cross_product = (
            vector1[1] * vector2[2] - vector1[2] * vector2[1],
            vector1[2] * vector2[0] - vector1[0] * vector2[2],
            vector1[0] * vector2[1] - vector1[1] * vector2[0]
        )
        # Calcul de la norme du produit en croix
        norm = (cross_product[0] ** 2 + cross_product[1] ** 2 + cross_product[2] ** 2) ** 0.5

        # Vérification de la colinéarité en comparant la norme avec la tolérance
        if norm <= tolerance:
            return True
        else:
            return False
        
    def run(self):

        rospy.init_node('point_at_detection')
        rospy.spin()

if __name__ == '__main__':
    try:
        node = PointAtNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

