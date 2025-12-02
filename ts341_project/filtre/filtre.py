import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_TEST_PATH = os.path.join(CURRENT_DIR, "..", "yolo_test")
sys.path.append(YOLO_TEST_PATH)

from results import run_yolo

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

frame_number = 0

for frame_resized, detections, confidences, class_ids in run_yolo():

    cv2.putText(frame_resized, f"derivee={12}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
    cv2.imshow("Resultat YOLO", frame_resized)

    frame_number+=1
    if len(detections) == 0:
        print("Aucune détection YOLO sur cette frame")
    for fra, det, conf, cls in zip(frame_resized, detections, confidences, class_ids):
        x1, y1, x2, y2 = det
        # print(f"Classe: {int(cls)}, Conf: {conf:.2f}, x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        if(not(derivate(50000))):
           print(f"frame YOLO {frame_number} non valide")

        sys.stdout.flush()
        buffer_drone.append([det, conf, cls])