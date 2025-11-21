"""Videos trating packages.

The utils package provides an API call to ease video opening etc...
"""

import cv2 as cv
import utils as u
import numpy as np


def remove_background(path: str) -> None:
    """Open the video and remove background.

    Return the video without background
    only with moving objects
    """
    cap: cv.VideoCapture = u.openvideo(path)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    fgbg = cv.createBackgroundSubtractorMOG2()
    # For the creation of the video without backgorund.
    # frame_width: int = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    # frame_height: int = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # fourcc = cv.VideoWriter_fourcc(*"mp4v")
    while 1:
        ret: bool
        frame: cv.Mat
        ret, frame = cap.read()
        if not ret:
            raise Exception("An error occured during the video")
        # Video without background
        fgmask: cv.Mat = fgbg.apply(frame)
        # thresholding the image to remove some noises
        threshold: cv.Mat
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        ret, threshold = cv.threshold(fgmask.copy(), 120, 255, cv.THRESH_BINARY)
        # removing some useless points (KNN like)
        contours: list[np.ndarray] = get_contours(threshold)
        centroids: list[tuple[int, int]] = get_moving_centroïds(contours)
        print(centroids)
        # add the frame to the video without background
        # open the raw video with rectangles on moving objects
        cv.imshow("Frame", frame)
        if cv.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
    return


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


remove_background("videos/video3")
