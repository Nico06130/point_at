#!/usr/bin/python3

import rospy
import mediapipe as mp
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import cv2
import pyrealsense2 as rs
import numpy as np
from yolov8_ros_msgs.srv import Yolov8
from point_at_srvs.srv import Pointed,PointedResponse


### Python class to detect pointed objects using Realsense camera, Mediapipe for hand tracking and Yolov8 for object detection ###


class RealSenseNode:

    def __init__(self):

        rospy.init_node('point_at_detection')
        self.point_at_pub = rospy.Publisher('point_at_frame',Image,queue_size=1)
        self.bridge = CvBridge()

        #Arguments for calling Yolov8
        self.yolo_model_name = "yolov8m.pt" 
        self.yolo_class = []

        #Server 
        self.server = rospy.Service('point_at_service',Pointed,self.serviceCallback)

        #Mediapipe init
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        #RealSense init
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)

    def mediapipe(self):
        """
        Recupere les coordonnes des landmarks de la main et les dessine sur la frame  
        """

        index_finger_landmarks = []

        with self.mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                max_num_hands = 1) as hands:

            while not rospy.is_shutdown():

                # Recuperation du flux couleur de la RealSense ainsi que du flux de profondeur 
                # pour calculer la coordonnee en z des pixels souhaites
                frames = self.pipeline.wait_for_frames()        
                color_frame = frames.get_color_frame()          
                depth_frame = frames.get_depth_frame()         

                if not color_frame:

                    continue

                color_image = np.asanyarray(color_frame.get_data())
                image = cv2.flip(color_image, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                h, w, _ = image.shape

                if results.multi_hand_landmarks:

                    for hand_landmarks in results.multi_hand_landmarks:

                        # Dessin des landmarks sur la frame
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                image_rgb,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )
                                
                        # Recuperation des coordonnees normalisees des landmarks du haut et du bas de l'index et
                        #  calcul de leur profondeur pour avoir une coordonnee en z
                        for landmark in [self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_DIP]:

                            if 0 < hand_landmarks.landmark[landmark].x < 1:
                                index_finger_landmarks.append((
                                    hand_landmarks.landmark[landmark].x,
                                    hand_landmarks.landmark[landmark].y,
                                    depth_frame.get_distance(
                                        int(hand_landmarks.landmark[landmark].x * w),
                                        int(hand_landmarks.landmark[landmark].y * h)
                                    )
                                ))

                        x1_norm, y1_norm,z1 = index_finger_landmarks[0]
                        x2_norm, y2_norm,z2 = index_finger_landmarks[1]

                        x1,y1 = x1_norm*w, y1_norm*h
                        x2,y2 = x2_norm*w, y2_norm*h

                        #Coordonnees 3D des landmarks ajustes a la frame
                        point_index_haut = (x1,y1,z1)
                        point_index_bas = (x2,y2,z2)

                        #Recuperation des parametres des objets detectes et dessin des bounding box 
                        image_yolo = self.bridge.cv2_to_imgmsg(image_rgb, "bgr8")
                        liste_objets_yolo = self.yolo_service(image_yolo)

                        for objet in liste_objets_yolo.boxes:

                            classe = objet.bbox_class
                            id_classe = objet.ID
                            x1_rect = objet.xmin
                            x2_rect = objet.xmax
                            y1_rect = objet.ymin
                            y2_rect = objet.ymax

                            cv2.rectangle(image_rgb, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), (0, 255, 0), 2) #dessin des boundingbox

                            x_centre_objet =  (x1_rect+ x2_rect)/2 
                            y_centre_objet = (y1_rect + y2_rect)/2 
                            z_centre_ojet = depth_frame.get_distance(int(x_centre_objet),int(y_centre_objet))
                            coord_centre_objet = (x_centre_objet,y_centre_objet,z_centre_ojet)

                            if self.are_collinear(point_index_bas,point_index_haut,coord_centre_objet,10):

                                cv2.rectangle(image_rgb, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), (0, 255, 0), -1)
                                rospy.loginfo("Pointe vers %s, d'ID %i",classe,id_classe)
                                self.serviceCallback(id_classe)

                else:

                    image_yolo = self.bridge.cv2_to_imgmsg(image_rgb, "bgr8")
                    liste_objets_yolo = self.yolo_service(image_yolo)

                    for objet in liste_objets_yolo.boxes:

                        x1_rect = objet.xmin
                        x2_rect = objet.xmax
                        y1_rect = objet.ymin
                        y2_rect = objet.ymax

                        cv2.rectangle(image_rgb, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), (0, 255, 0), 2) #dessin des boundingbox


            self.point_at_pub.publish(self.bridge.cv2_to_imgmsg(image_rgb, "bgr8"))

        self.pipeline.stop()

    def yolo_service(self,frame):
        """
        Appel du service Yolov8 pour recuperer les objets detectes et leurs parametres associes

        """

        rospy.wait_for_service('yolov8_on_unique_frame')

        try:

            rospy.loginfo("Attente du service Yolov8")

            get_boxes_espace = rospy.ServiceProxy('yolov8_on_unique_frame',Yolov8)             
            bbox_espace = get_boxes_espace(self.yolo_model_name,self.yolo_class,frame).boxes

            return bbox_espace  

        except rospy.ServiceException as e:

            print("Erreur dans l'appel du service Yolov8: %s"%e)

    def serviceCallback(self,box_id):

        return(PointedResponse(box_id))

    def are_collinear(self,point1, point2, point3, tolerance):
        """
        Verifie si trois points sont approximativement collineaires dans l'espace.
        """
        # Calcul des vecteurs formes par les points
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

        # Verification de la colinearite en comparant la norme avec la tolerance
        if norm <= tolerance:
            return True
        else:
            return False
        

if __name__ == '__main__':

    try:
        node = RealSenseNode()
        node.mediapipe()
    except rospy.ROSInterruptException:
        pass
