import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_TEST_PATH = os.path.join(CURRENT_DIR, "..", "yolo_test")
sys.path.append(YOLO_TEST_PATH)

MOVINGS_PATH = os.path.join(CURRENT_DIR, "..")
sys.path.append(MOVINGS_PATH)

from results import run_yolo
from background import get_movings

buffer_drone = []

values = []
fig, ax = plt.subplots()

plt.ion()

def derivate(max_derivate):
    if(len(buffer_drone)>=2):
        # print(f"{buffer_drone[len(buffer_drone)-1]}")

        position_box_t0 = buffer_drone[len(buffer_drone)-1][0]
        position_box_t1 = buffer_drone[len(buffer_drone)-2][0]

        position_t0 = np.array([(position_box_t0[0] + position_box_t0[2])/2, (position_box_t0[1] + position_box_t0[3])/2])
        position_t1 = np.array([(position_box_t1[0] + position_box_t1[2])/2, (position_box_t1[1] + position_box_t1[3])/2])

        

        derivate = (position_t1-position_t0)[0]**2 + (position_t1-position_t0)[1]**2


        values.append(derivate)

        ax.clear()
        ax.plot(values)
        ax.set_title("derivee sur 1 frame")
        ax.set_xlabel("frame")
        ax.set_ylabel("derivee sur 1 frame")
        plt.pause(0.001)

        if(derivate>=max_derivate):
            return False
        return True

        # print(derivate)

def nothing():
    return 1


gen_yolo = run_yolo()
gen_motion = get_movings()

for (frame_number, frame_resized, height_ratio_resize, width_ratio_resize, detections, confidences, class_ids), (frame_id, centroids, motion_frame) in zip(gen_yolo, gen_motion):

    x1, y1, x2, y2 = 0,0,0,0

    if len(detections) == 0:
        print("Aucune détection YOLO sur cette frame")
    else:
        x1, y1, x2, y2 = detections[0]
        centroid_yolo = [(x1+x2)/2, (y1+y2)/2]
        print(f"{x1*width_ratio_resize}, {y1*height_ratio_resize}")
        cv2.circle(frame_resized, (int(centroid_yolo[0]* width_ratio_resize), int(centroid_yolo[1]* height_ratio_resize)), 10, (0, 255, 0), -1)

    for  det, conf, cls in zip( detections, confidences, class_ids):
        x1, y1, x2, y2 = det
        # print(f"Classe: {int(cls)}, Conf: {conf:.2f}, x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    
        sys.stdout.flush()
        buffer_drone.append([det, conf, cls])

    if(not(derivate(50000))):
           print(f"detection YOLO {frame_number} derivee non valide")
    # print(f"detection YOLO {frame_number} derivee valide")

    
    cv2.putText(frame_resized, f"derivee={12}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Resultat YOLO", frame_resized)  