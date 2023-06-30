#!/usr/bin/python3

import rospy
import ktb
import mediapipe as mp
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import tf2_ros
from geometry_msgs.msg import TransformStamped
from math import sin, pi

class PointAtNode:

    def __init__(self):

        rospy.init_node('pôint_at_detection')
        #self.image_pub = rospy.Publisher('image', Image, queue_size=10)
        self.label_pub = rospy.Publisher('pointage', String, queue_size=10)

        self.bridge = CvBridge()
        # self.mp_drawing = mp.solutions.drawing_utils
        # self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

        self.k = ktb.Kinect()


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

        rate = rospy.Rate(10)  # 10Hz

        with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                max_num_hands = 1) as hands:
            

            while not rospy.is_shutdown():
                # Specify as many types as you want here

                color_frame = self.k.get_frame(ktb.RAW_COLOR)
                depth_frame = self.k.get_frame(ktb.RAW_DEPTH)
                height, width = depth_frame.shape

                if not color_frame:
                    continue
                    
                # Get the original frame dimensions
                height, width, _ = color_frame.shape

                # Calculate the new dimensions while maintaining the aspect ratio
                aspect_ratio = width / height
                new_width = 512
                new_height = 424

                # Calculate the target aspect ratio
                target_aspect_ratio = new_width / new_height

                # Resize the frame while preserving the aspect ratio
                if aspect_ratio > target_aspect_ratio:
                    # Add vertical padding
                    padding = int((width / target_aspect_ratio - height) / 2)
                    resized_frame = cv2.copyMakeBorder(color_frame, padding, padding, 0, 0, cv2.BORDER_CONSTANT)
                else:
                    # Add horizontal padding
                    padding = int((height * target_aspect_ratio - width) / 2)
                    resized_frame = cv2.copyMakeBorder(color_frame, 0, 0, padding, padding, cv2.BORDER_CONSTANT)

                # Resize the frame to the target dimensions
                resized_frame = cv2.resize(resized_frame, (new_width, new_height))

                height_resized, width_resized, _ = resized_frame.shape

                color_image = np.asanyarray(resized_frame)
                image = cv2.flip(color_image, 1)

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                results = hands.process(image_rgb)

                results_yolo = self.model(image)

                h, w, _ = image.shape

                if results.multi_hand_landmarks:

                    for hand_landmarks in results.multi_hand_landmarks:
                        # mp_drawing.draw_landmarks(
                        #     image_rgb,
                        #     hand_landmarks,
                        #     mp_hands.HAND_CONNECTIONS,
                        #     mp_drawing_styles.get_default_hand_landmarks_style(),
                        #     mp_drawing_styles.get_default_hand_connections_style()
                        # )

                        index_finger_landmarks = []

                        for landmark in [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_DIP]:
                            index_finger_landmarks.append((
                                hand_landmarks.landmark[landmark].x,
                                hand_landmarks.landmark[landmark].y,
                                )
                            )
                        z1 = depth_frame[(int(hand_landmarks.landmark[landmark].y * height_resized)),(int(hand_landmarks.landmark[landmark].x * width_resized))]
                                        
                        x1_raw, y1_raw = index_finger_landmarks[0]
                        x2_raw, y2_raw = index_finger_landmarks[1]
                        # print(index_finger_landmarks)



                        point1 = (x1_raw,y1_raw,z1)
                        point2 = (x2_raw,y2_raw,z1)

                        
                        
                        x1,y1 = x1_raw*w, y1_raw*h
                        x2,y2 = x2_raw*w, y2_raw*h


                for detection in results_yolo.xyxy[0]:

                    # if confidence > 0.5:

                    x1_rect, y1_rect, x2_rect, y2_rect, confidence, class_id = detection
                    label = "Pointe vers:" + self.model.names[int(class_id)]
                    x_milieu = (x1_rect+x2_rect)/2
                    y_milieu = (y1_rect + y2_rect)/2
                    z_milieu = depth_frame[(int(y_milieu)),(int(x_milieu))]/100 # en cm


                    coord_milieu = (int(x_milieu),int(y_milieu))

                    point3 = (x_milieu,y_milieu,z_milieu)

                    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                    if  label == "bottle":


                        if self.are_collinear(point1, point2, point3,10):
                            self.label_pub.publish(label)

                    rate.sleep()

if __name__ == '__main__':

    try:
        node = PointAtNode()
        node.run()
    except rospy.ROSInterruptException:
        pass