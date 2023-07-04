#!/usr/bin/python3

import rospy
import mediapipe as mp
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
from boxes_3D.srv import boxes3D
import struct

class PointAtNode:

    def __init__(self):

        self.image_sub = rospy.Subscriber('/kinect2/hd/image_color', Image, self.image_callback)
        self.point_cloud_sub = rospy.Subscriber('/kinect2/sd/image_depth', Image, self.image_depth_callback)
        self.bridge = CvBridge()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.image_depth = None

        #Service

        self.model_name = "yolov8m.pt"
        self.model_class = []
        
        #Coordonnees

        self.point_index = [0,0,0]
        self.point_paume = [0,0,0]
        self.point_objet = [0,0,0]


    def image_depth_callback(self,data):

        self.image_depth = data

        # #Recuperation de la profondeur de la paume de la main

        # depth_modified = self.bridge.imgmsg_to_cv2(data)
        # x = 0
        # y = 0
        # z_value = depth_modified[y, x]

        # rospy.loginfo("La profondeur de la paume vaut: ",z_value)




    def image_callback(self, req):

        self.flux = req

        frame = self.bridge.imgmsg_to_cv2(req, "bgr8")
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        self.get_3D_bboxes()

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



        #cv2.imshow("Point at", frame)
        #cv2.waitKey(1)

    
    def get_3D_bboxes(self):

        rospy.loginfo("Entree dans la callback")
        rospy.wait_for_service('boxes_3d_service')

        if self.image_depth != None:

            try:
                rospy.loginfo("Attente du service")
                get_boxes = rospy.ServiceProxy('boxes_3d_service',boxes3D)
                
                bbox = get_boxes(self.model_name,self.model_class,self.flux,self.image_depth).boxes3D
                rospy.loginfo(bbox)
            
            except rospy.ServiceException as e:

                print("Erreur dans l'appel du service: %s"%e)

        else:
            rospy.loginfo("Pas de depth frame")


    def are_collinear(self,point1, point2, point3, tolerance):
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
        

    # def get_pointed_objects(self):

    #     if self.are_collinear(self.point_paume,self.point_index,self.point_objet,1):
    #         rospy.loginfo("Les points sont coliineaires")

    
    def run(self):

        rospy.init_node('point_at_detection')
        
        rospy.spin()

if __name__ == '__main__':
    try:
        node = PointAtNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

