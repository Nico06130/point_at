#!/usr/bin/python3

import rospy
import mediapipe as mp
from sensor_msgs.msg import Image 
from std_msgs.msg import String, Float32, UInt16
from cv_bridge import CvBridge
import cv2
import numpy as np
from boxes_3D.srv import boxes3D
from yolov8_ros.srv import Yolov8
from point_at.srv import Pointed,PointedResponse

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

        self.myService = rospy.Service('Point at service', Pointed, self.myServiceCallback)
        self.model_name = "yolov8m.pt"
        self.model_class = [0]
        self.MAX_ERROR = 50


    def image_depth_callback(self,data):

        self.image_depth = data
                    
    def image_callback(self,data):

        self.flux = data
        vectors = self.mediapipe(data)

        if vectors != None : 

            self.index = vectors[0]
            self.wrist = vectors[1]

        # self.myServiceCallback()

    def mediapipe(self,image):

        
        frame = self.bridge.imgmsg_to_cv2(image, "bgr8")
        h, w, _ = frame.shape

        point_wrist = (0,0,0)
        point_index=(0,0,0)

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
                    z_index = self.get_z_landmarks(x_index,y_index)

                    x_wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x*w
                    y_wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y*h
                    z_wrist = self.get_z_landmarks(x_wrist,y_wrist)

                    point_index = (x_index,y_index,z_index)
                    point_wrist = (x_wrist,y_wrist,z_wrist)

                    landmarks = (point_index,point_wrist)

                    return landmarks

            else:

                landmarks = None

            return landmarks
        
                # cv2.rectangle(frame, (int(self.x1_rect), int(self.y1_rect)), (int(self.x2_rect), int(self.y2_rect)), (0, 255, 0), 2)

                # if point_wrist != (0,0,0) and point_index != (0,0,0):

                #     if self.are_collinear_espace(point_wrist,point_index,objet,5) :

                #         rospy.loginfo("Pointe vers:",boite.bbox_class)
                #         cv2.rectangle(frame, (int(self.x1_rect), int(self.y1_rect)), (int(self.x2_rect), int(self.y2_rect)), (0, 255, 0), -1)
                    

    def myServiceCallback(self,req):


        #rospy.loginfo("Request: "+str(req))
        rate = rospy.Rate(10) # 10hz
        isEnd = False
        error_count = 0

        while not rospy.is_shutdown() or not isEnd or error_count<self.MAX_ERROR: 

        # get the current vector self._vector 

            point1 = self.wrist
            point2 = self.index

        # service call boxes_3d

            try:

                rospy.loginfo("Attente du service boxes 3D")

                get_boxes_espace = rospy.ServiceProxy('boxes_3d_service',boxes3D)             
                bbox_espace = get_boxes_espace(self.model_name,self.model_class,self.flux,self.image_depth).boxes3D

                for boite in bbox_espace:

                    x1_rect = boite.xmin
                    x2_rect = boite.xmax
                    y1_rect = boite.ymin
                    y2_rect = boite.ymax
                    classe = boite.bboxlass
                    id_class = boite.ID

                    x_objet =  (self.x1_rect+ self.x2_rect)/2
                    y_objet = (self.y1_rect + self.y2_rect)/2
                    z_objet = boite.centerz /100 #en cm

                    point3 = (x_objet,y_objet,z_objet)

                    if self.are_collinear_espace(point1,point2,point3):
                        rospy.loginfo("Pointe vers: %s",classe)
                        isEnd = True

            except rospy.ServiceException as e:

                print("erreur d'appel du service")

            error_count +=1

            rate.sleep()

        #self.point_at_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

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

    def get_z_landmarks(self,coordx,coordy):
        """
        Calcul de la profondeur avec le pointcloud
        """

        depth_array=self.bridge.imgmsg_to_cv2(self.image_depth,"16UC1")
        z_coord=depth_array[int(coordy)][int(coordx)]

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
