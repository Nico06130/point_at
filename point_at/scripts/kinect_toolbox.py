#!/usr/bin/python3

import rospy
import mediapipe as mp
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
from boxes_3D.srv import boxes3D
from yolov8_ros.srv import Yolov8

class PointAtNode:

    def __init__(self):

        self.image_sub = rospy.Subscriber('/kinect2/hd/image_color', Image, self.image_callback)
        self.point_at_pub = rospy.Publisher('point_at_frame',Image,queue_size=10)
        self.point_cloud_sub = rospy.Subscriber('/kinect2/hd/image_depth_rect', Image, self.image_depth_callback)
        self.bridge = CvBridge()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.image_depth = None

        #Service

        self.model_name = "yolov8m.pt"
        self.model_class = []
        
        # #Coordonnees

        self.x_index = 0
        self.y_index = 0
        self.z_index = 0
        self.x_wrist = 0
        self.y_wrist = 0
        self.z_wrist = 0
        self.x_objet = 0
        self.y_objet = 0
        self.z_objet = 0

        self.landmark_coord = [] #2D coord for wrist and index
        self.point_index = [self.x_index,self.y_index,self.z_index]
        self.point_paume = [self.x_wrist,self.y_wrist,self.z_wrist  ]
        self.point_objet = [self.x_objet,self.y_objet,self.z_objet]


    def image_depth_callback(self,data):

        self.image_depth = data

        # #Recuperation de la profondeur de la paume et de l'index

        depth_modified = self.bridge.imgmsg_to_cv2(data,"16UC1")

        if self.y_index != 0 and self.x_index != 0:

            self.z_index = depth_modified[int(self.y_index),int(self.x_index)]
            self.z_wrist = depth_modified[int(self.y_wrist),int(self.x_wrist)]
            # self.z_index = depth_modified[int(960),int(540)]
            rospy.loginfo("Profondeur wrist en m: ",self.z_wrist/100)

        else:

            rospy.loginfo("Pas de coordonnées")

    def image_callback(self, req):

        self.flux = req

        # self.get_3D_bboxes()
        self.draw_on_frame(req)

    def are_collinear_espace(self,point1, point2, point3, tolerance):
        """
        Vérifie si trois points sont approximativement collinéaires dans l'espace.
        """
        # Calcul des vecteurs formés par les points

        vector1 = (point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2])
        vector2 = (point3[0] - point2[0], point3[1] - point2[1], point3[2] - point2[2])

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

    def draw_on_frame(self,image):

        frame = self.bridge.imgmsg_to_cv2(image, "bgr8")
        image_rgb = frame
        h, w, _ = frame.shape

        # Appel du service boxes_3d_service pour recuperer les coordonnes 3D des boundingbpx

        rospy.loginfo("Entree dans la callback d'appel de boxes 3D")
        rospy.wait_for_service('boxes_3d_service')

        if self.image_depth != None:

            try:
                rospy.loginfo("Attente du service boxes 3D")

                get_boxes_espace = rospy.ServiceProxy('boxes_3d_service',boxes3D)             
                bbox_espace = get_boxes_espace(self.model_name,self.model_class,self.flux,self.image_depth).boxes3D

        ### Dessiner les boundingbox des objets ###

                for boite in bbox_espace:

                    self.x1_rect = boite.xmin
                    self.x2_rect = boite.xmax
                    self.y1_rect = boite.ymin
                    self.y2_rect = boite.ymax

                    self.x_objet =  (self.x1_rect+ self.x2_rect)/2
                    self.y_objet = (self.y1_rect + self.y2_rect)/2
                    self.z_objet = boite.centerz

                    #rospy.loginfo(self.z_objet)
                    cv2.rectangle(frame, (int(self.x1_rect), int(self.y1_rect)), (int(self.x2_rect), int(self.y2_rect)), (0, 255, 0), 2)
         
            except rospy.ServiceException as e:

                print("Erreur dans l'appel du service: %s"%e)

        else:

            rospy.loginfo("Pas de depth frame")

        ### Dessiner les landmarks de la main ###

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

                    for landmark in [self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.WRIST]:

                        self.landmark_coord.append((
                            hand_landmarks.landmark[landmark].x*w,
                            hand_landmarks.landmark[landmark].y*h,
                            )
                        )



                        rospy.loginfo(self.landmark_coord)
                        cv2.circle(frame,(int(self.landmark_coord[0][0]),int(self.landmark_coord[0][1])),10,(255,0,0),2)
                
                    #rospy.loginfo(self.landmark_coord)

                    # rospy.loginfo("Coords de lindex:",self.landmark_coord[0])

        self.point_at_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))


    # def get_z_landmarks(self,coord):
    #     """
    #     Calcul de la profondeur avec le pointcloud
    #     """
    #     depth_array=self.bridge.imgmsg_to_cv2(self.image_depth,"16UC1")
    #     z_coord=depth_array[int(coord[0])][int(coord[1])]
    #     rospy.loginfo(z_coord)

    #     if z_coord==0:
    #         z_coord=-1

    #     return z_coord
    

    def run(self):

        rospy.init_node('point_at_detection')
        
        rospy.spin()

if __name__ == '__main__':
    try:
        node = PointAtNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

