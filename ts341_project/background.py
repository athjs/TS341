"""Videos trating packages.

The utils package provides an API call to ease video opening etc...
"""

import cv2 as cv
import utils as u
import numpy as np
from threading import Thread
from queue import Queue


# def remove_background(path: str) -> dict[int, list[tuple[int, int]]]:
#     """Open the video and remove background.
#
#     Return the video without background
#     only with moving objects
#     """
#     cap: cv.VideoCapture = u.openvideo(path)
#     kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
#     fgbg = cv.createBackgroundSubtractorMOG2()
#     # For the creation of the video without backgorund.
#     # frame_width: int = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
#     # frame_height: int = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
#     # fourcc = cv.VideoWriter_fourcc(*"mp4v")
#     while 1:
#         ret: bool
#         frame: cv.Mat
#         ret, frame = cap.read()
#         if not ret:
#             raise Exception("An error occured during the video")
#         # Video without background
#         fgmask: cv.Mat = fgbg.apply(frame)
#         # thresholding the image to remove some noises
#         threshold: cv.Mat
#         fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
#         ret, threshold = cv.threshold(fgmask.copy(), 120, 255, cv.THRESH_BINARY)
#         # removing some useless points (KNN like)
#         contours: list[np.ndarray] = get_contours(threshold)
#         centroids: list[tuple[int, int]] = get_moving_centroïds(contours)
#         frame_id: int = int(cap.get(1))
#         frame_centroids: dict[int, list[tuple[int, int]]] = {}
#         frame_centroids[frame_id] = centroids
#         yield frame_centroids
#         print(frame_centroids)
#         # add the frame to the video without background
#         # open the raw video with rectangles on moving objects
#         cv.imshow("Frame", frame)
#         if cv.waitKey(30) & 0xFF == ord("q"):
#             break
#
#     cap.release()
#     cv.destroyAllWindows()
#     return frame_centroids


display_q = Queue(maxsize=1)

def display_loop():
    while True:
        item = display_q.get()
        if item is None:
            return
        frame = item
        cv.imshow("Frame", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            return

def frame_generator(cap, fgbg, kernel):
    while True:
        ret, frame = cap.read()
        if not ret:
            return

        fgmask = fgbg.apply(frame)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        _, thr = cv.threshold(fgmask, 120, 255, cv.THRESH_BINARY)

        contours = get_contours(thr)
        centroids = get_moving_centroïds(contours)
        frame_id = int(cap.get(1))

        yield frame_id, centroids, frame

def main():
    cap = u.openvideo("videos/video3")
    fgbg = cv.createBackgroundSubtractorMOG2()
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    t = Thread(target=display_loop, daemon=True)
    t.start()

    for frame_id, centroids, frame in frame_generator(cap,fgbg,kernel):
        try:
            display_q.put_nowait(frame)
        except:
            pass   # si l'afficheur n’a pas consommé, on écrase l’ancien

        # traitement en parallèle
        print(frame_id, centroids)

    display_q.put(None)
    t.join()
    cap.release()
    cv.destroyAllWindows()

def get_contours(threshold: cv.VideoCapture) -> list[np.ndarray]:
    """Returns contours of objects in a video thresholded."""
    contours: list[np.ndarray]
    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def get_moving_centroïds(contours: list[np.ndarray]) -> list[tuple[int, int]]:
    """Returns centroids of each object contour of the video."""
    centroids: list[tuple[int, int]] = []
    for contour in contours:
        area = cv.contourArea(contour)
        # draw a contour only if it is big enough
        if 10 < area < 1500:
            M = cv.moments(contour)
            # shows x and y coordinates
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
    return centroids

main()
# remove_background("videos/video3")
