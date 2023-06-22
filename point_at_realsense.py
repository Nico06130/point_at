import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
import torch

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

m = 0
b = 0

def are_collinear(point1, point2, point3, tolerance):
    """
    Vérifie si trois points sont approximativement collinéaires dans l'espace.
    Les points sont représentés sous forme de tuples (x, y, z).
    La tolérance spécifie la marge d'erreur acceptable.
    Retourne True si les points sont approximativement collinéaires, False sinon.
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


    

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands = 1) as hands:

    while True:

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        image = cv2.flip(color_image, 1)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image_rgb)

        results_yolo = model(image)

        h, w, _ = image.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                index_finger_landmarks = []

                for landmark in [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_DIP]:
                    index_finger_landmarks.append((
                        hand_landmarks.landmark[landmark].x,
                        hand_landmarks.landmark[landmark].y,
                        depth_frame.get_distance(
                            int(hand_landmarks.landmark[landmark].x * w),
                            int(hand_landmarks.landmark[landmark].y * h)
                        )
                    ))

                x1_raw, y1_raw,z1 = index_finger_landmarks[0]
                x2_raw, y2_raw,z2 = index_finger_landmarks[1]

                point1 = (x1_raw,y1_raw,z1)
                point2 = (x2_raw,y2_raw,z1)

                
                
                x1,y1 = x1_raw*w, y1_raw*h
                x2,y2 = x2_raw*w, y2_raw*h

                cv2.circle(image,(int(x1),int(y1)),10,(255,0,0),2)

                x_dir = x2 - x1
                y_dir = y2 - y1

                if x_dir != 0 :

                    m = y_dir/x_dir                     
                    b = y1 - m*x1
                    vect_dir = (int(x_dir),int(y_dir))

                else:

                    m = 0
                    b = 0
                    vect_dir = (0,0)

                if x2 - x1 >= 0:

                    x_edge = -w

                elif x2 - x1 < 0:

                    x_edge = w

                y_edge = x_edge*m + b

                #cv2.line(image, (int(x1),int(y1)),(int(x_edge),int(y_edge)),(255,0,0), 3)


        for detection in results_yolo.xyxy[0]:

            x1_rect, y1_rect, x2_rect, y2_rect, confidence, class_id = detection
            label = model.names[int(class_id)]
            x_milieu = (x1_rect+x2_rect)/2
            y_milieu = (y1_rect + y2_rect)/2
            z_milieu = depth_frame.get_distance(int(x_milieu), int(y_milieu))

            point3 = (x_milieu,y_milieu,z_milieu)

            coord_milieu = (int(x_milieu),int(y_milieu))

            if  label == "suitcase":

                cv2.rectangle(image, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), (0, 255, 0), 2)                            
                text = cv2.putText(image, label + " " + str(confidence), (int(x1_rect), int(y1_rect - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                for x in range(int(x1_rect),int(x2_rect)) :
                
                    y = x * m + b

                    if y1_rect <= y <= y2_rect and m>0 :
                    # if y1_rect - 50 <= y <= y2_rect + 50 :

                        if are_collinear(point1, point2, point3,1):

                            cv2.rectangle(image, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), (0, 255, 0), -1)
                            text = cv2.putText(image, label + " " + str(confidence), (int(x1_rect), int(y1_rect - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                            print("¨Pointe vers:",label)

                    elif y1_rect <= y <= y2_rect and m<0 :
                    # if y1_rect - 50 <= y <= y2_rect + 50 :

                        if are_collinear(point1, point2, point3,1):

                            cv2.rectangle(image, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), (0, 255, 0), -1)
                            text = cv2.putText(image, label + " " + str(confidence), (int(x1_rect), int(y1_rect - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                            print("¨Pointe vers:",label)

                # if are_collinear(point1, point2, point3,1):

                #     cv2.rectangle(image, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), (0, 255, 0), -1)
                #     text = cv2.putText(image, label + " " + str(confidence), (int(x1_rect), int(y1_rect - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                #     print("¨Pointe vers:",label)              

        cv2.imshow('Point at', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

pipeline.stop()
cv2.destroyAllWindows()
