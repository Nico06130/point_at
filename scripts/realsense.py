#!/usr/bin/python3

import rospy
import mediapipe as mp
#from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import cv2
import pyrealsense2 as rs
import numpy as np
import torch
#import tf2_ros
#from geometry_msgs.msg import TransformStamped
from math import sin, pi

class RealSenseNode:

    def __init__(self):

        rospy.init_node('pôint_at_detection')
        self.label_pub = rospy.Publisher('pointe', String, queue_size=10)
        self.bridge = CvBridge()

        self.mp_hands = mp.solutions.hands

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)

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

    def getRealXY(self, x_ref, y_ref, distance, img_w=640, img_h=480, HFovDeg=90, VFovDeg=65):
        
        HFov = HFovDeg * pi / 180.0  # Horizontal field of view of the RealSense D455
        VFov = VFovDeg * pi / 180.0
        #Phi = (HFov / 2.0) * ( (2*neck_x)/self.image_w + 1)  #Angle from the center of the camera to neck_x
        PhiX = (HFov / 2.0) *  (x_ref - img_w/2) / (img_w/2) #Angle from the center of the camera to neck_x
        PhiY = (VFov / 2.0) *  (y_ref - img_h/2) / (img_h/2)
        return (    distance * sin(PhiX)  ,     distance * sin(PhiY)   )


    def run(self):
        
        rate = rospy.Rate(10)  # 10Hz

        with self.mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                max_num_hands = 1) as hands:

            while not rospy.is_shutdown():

                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame:

                    continue

                color_image = np.asanyarray(color_frame.get_data())
                image = cv2.flip(color_image, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                results_yolo = self.model(image)

                h, w, _ = image.shape

                if results.multi_hand_landmarks:

                    for hand_landmarks in results.multi_hand_landmarks:

                        index_finger_landmarks = []

                        for landmark in [self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_DIP]:
                            index_finger_landmarks.append((
                                hand_landmarks.landmark[landmark].x,
                                hand_landmarks.landmark[landmark].y,
                                depth_frame.get_distance(
                                    float(hand_landmarks.landmark[landmark].x * w),
                                    float(hand_landmarks.landmark[landmark].y * h)
                                )
                            ))

                        x1_raw, y1_raw,z1 = index_finger_landmarks[0]
                        x2_raw, y2_raw,z2 = index_finger_landmarks[1]

                        point1 = (x1_raw,y1_raw,z1)
                        point2 = (x2_raw,y2_raw,z1)

                        
                        
                        x1,y1 = x1_raw*w, y1_raw*h
                        x2,y2 = x2_raw*w, y2_raw*h

                        # cv2.circle(image,(int(x1),int(y1)),10,(255,0,0),2)

                else:

                    point1 = (0,0,0)
                    point2 = (1,1,1)

                for detection in results_yolo.xyxy[0]:

                    x1_rect, y1_rect, x2_rect, y2_rect, confidence, class_id = detection
                    label = self.model.names[int(class_id)]
                    x_milieu = (x1_rect+x2_rect)/2
                    y_milieu = (y1_rect + y2_rect)/2
                    z_milieu = depth_frame.get_distance(int(x_milieu), int(y_milieu))

                    x_real, y_real = self.getRealXY(x_milieu, y_milieu, z_milieu, w, h)

                    point3 = (x_real,y_real,z_milieu)

                    coord_milieu = (int(x_milieu),int(y_milieu))

                    if  label == "bottle":

                        if self.are_collinear(point1, point2, point3,10):

                            self.label_pub.publish(label)

            rate.sleep()

     
        self.pipeline.stop()

if __name__ == '__main__':
    try:
        node = RealSenseNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
