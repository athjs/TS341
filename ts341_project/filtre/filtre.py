import os
import sys
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_TEST_PATH = os.path.join(CURRENT_DIR, "..", "yolo_test")
sys.path.append(YOLO_TEST_PATH)

from results import run_yolo

buffer_drone = []

values = []
fig, ax = plt.subplots()

plt.ion()

def derivate():
    if(len(buffer_drone)>=2):
        # print(f"{buffer_drone[len(buffer_drone)-1]}")

        position_box_t0 = buffer_drone[len(buffer_drone)-1][0]
        position_box_t1 = buffer_drone[len(buffer_drone)-2][0]

        position_t0 = np.array([(position_box_t0[0] + position_box_t0[2])/2, (position_box_t0[1] + position_box_t0[3])/2])
        position_t1 = np.array([(position_box_t1[0] + position_box_t1[2])/2, (position_box_t1[1] + position_box_t1[3])/2])

        print()

        derivate = (position_t1-position_t0)[0]**2 + (position_t1-position_t0)[1]**2

        values.append(derivate)

        ax.clear()
        ax.plot(values)
        ax.set_title("derivee sur 1 frame")
        ax.set_xlabel("frame")
        ax.set_ylabel("derivee sur 1 frame")
        plt.pause(0.001)

        # print(derivate)



for detections, confidences, class_ids in run_yolo():
    if len(detections) == 0:
        print("Aucune détection sur cette frame")
    for det, conf, cls in zip(detections, confidences, class_ids):
        x1, y1, x2, y2 = det
        # print(f"Classe: {int(cls)}, Conf: {conf:.2f}, x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        derivate()
        sys.stdout.flush()
        buffer_drone.append([det, conf, cls])