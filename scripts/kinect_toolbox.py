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
        self.model_class = [0]
        
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
        self.point_index = []
        self.point_paume = []
        self.point_objet = []

    def image_depth_callback(self,data):

        self.image_depth = data
                    
        #else:

            #rospy.loginfo("Pas de coordonnées")

    def image_callback(self, req):

        self.flux = req
        self.draw_on_frame(req)

    def are_collinear_espace(self,point1, point2, point3, tolerance):
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

    def draw_on_frame(self,image):

        
        frame = self.bridge.imgmsg_to_cv2(image, "bgr8")
        frame_copy = frame
        h, w, _ = frame.shape

        point_wrist = (0,0,0)
        point_index=(0,0,0)
        rospy.wait_for_service('boxes_3d_service')

        if self.image_depth != None:

            try:
                #rospy.loginfo("Attente du service boxes 3D")

                get_boxes_espace = rospy.ServiceProxy('boxes_3d_service',boxes3D)             
                bbox_espace = get_boxes_espace(self.model_name,self.model_class,self.flux,self.image_depth).boxes3D


### Dessiner les landmarks de la main ###

                with self.mp_hands.Hands(
                        model_complexity=0,
                        min_detection_confidence=0.01,
                        min_tracking_confidence=0.01,
                        max_num_hands=1) as hands:

                    results = hands.process(frame)

                    if results.multi_hand_landmarks:

                        rospy.loginfo("Main detectee")

                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )

                            x_index = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x*w
                            y_index = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y*h
                            rospy.loginfo(x_index)
                            z_index = self.get_z_landmarks(x_index,y_index)
                            point_index = (x_index,y_index,z_index)

                            x_wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x*w
                            y_wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y*h
                            z_wrist = self.get_z_landmarks(x_wrist,y_wrist)

                            point_wrist = (x_wrist,y_wrist,z_wrist)

### Dessiner les boundingbox des objets ###

                for boite in bbox_espace:

                    self.x1_rect = boite.xmin
                    self.x2_rect = boite.xmax
                    self.y1_rect = boite.ymin
                    self.y2_rect = boite.ymax

                    self.x_objet =  (self.x1_rect+ self.x2_rect)/2
                    self.y_objet = (self.y1_rect + self.y2_rect)/2
                    self.z_objet = boite.centerz /100 #en cm

                    objet = (self.x_objet,self.y_objet,self.z_objet)
                    #rospy.loginfo(self.z_objet)

                    cv2.rectangle(frame, (int(self.x1_rect), int(self.y1_rect)), (int(self.x2_rect), int(self.y2_rect)), (0, 255, 0), 2)

                    if point_wrist != (0,0,0) and point_index != (0,0,0):

                        if self.are_collinear_espace(point_wrist,point_index,objet,5) :

                            rospy.loginfo("Pointe vers:",boite.bbox_class)
                            cv2.rectangle(frame, (int(self.x1_rect), int(self.y1_rect)), (int(self.x2_rect), int(self.y2_rect)), (0, 255, 0), -1)
                        


            except rospy.ServiceException as e:

                print("erreur d'appel du service")


                    #rospy.loginfo(point_wrist)

        else:

            rospy.loginfo("Pas de depth frame")


        # Appel du service boxes_3d_service pour recuperer les coordonnes 3D des boundingbpx

        # rospy.loginfo("Entree dans la callback d'appel de boxes 3D")
        # rospy.wait_for_service('boxes_3d_service')

        # if self.image_depth != None:
        #     try:
        #         #rospy.loginfo("Attente du service boxes 3D")

        #         get_boxes_espace = rospy.ServiceProxy('boxes_3d_service',boxes3D)             
        #         bbox_espace = get_boxes_espace(self.model_name,self.model_class,self.flux,self.image_depth).boxes3D

        # ### Dessiner les boundingbox des objets ###

        #         for boite in bbox_espace:

        #             self.x1_rect = boite.xmin
        #             self.x2_rect = boite.xmax
        #             self.y1_rect = boite.ymin
        #             self.y2_rect = boite.ymax

        #             self.x_objet =  (self.x1_rect+ self.x2_rect)/2
        #             self.y_objet = (self.y1_rect + self.y2_rect)/2
        #             self.z_objet = boite.centerz /100 #en cm

        #             #rospy.loginfo(self.z_objet)
        #             cv2.rectangle(frame, (int(self.x1_rect), int(self.y1_rect)), (int(self.x2_rect), int(self.y2_rect)), (0, 255, 0), 2)
         
        #     except rospy.ServiceException as e:

        #         print("Erreur dans l'appel du service: %s"%e)


        #     #rospy.loginfo(point_wrist)

        # else:

        #     rospy.loginfo("Pas de depth frame")

                
                    #rospy.loginfo(self.landmark_coord)

                    # rospy.loginfo("Coords de lindex:",self.landmark_coord[0])

        self.point_at_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))


    def get_z_landmarks(self,coordx,coordy):
        """
        Calcul de la profondeur avec le pointcloud
        """

        if coordx != None and coordy != None:
        
            depth_array=self.bridge.imgmsg_to_cv2(self.image_depth,"16UC1")
            z_coord=depth_array[int(coordy)][int(coordx)]
        #rospy.loginfo(z_coord)

            if z_coord==0:
                z_coord=-1

        else:

            z_coord = -1

        return z_coord
        
    

    def run(self):

        rospy.init_node('point_at_detection')
        
        rospy.spin()

if __name__ == '__main__':
    try:
        node = PointAtNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

# # Print hand coordinates
# for idx, landmark in enumerate(hand_landmarks.landmark):
#     x = int(landmark.x * w)
#     y = int(landmark.y * h)
#     z = int(landmark.z * 1000)  # Scale z-coordinate for better visibility
#     rospy.loginfo(f"Hand Landmark {idx}: ({x}, {y}, {z})")
