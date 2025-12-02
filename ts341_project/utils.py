"""Importing cv2."""

import cv2 as cv


def openvideo(path: str) -> cv.VideoCapture:
    """Open a video and returns it or raise an error."""
    cap: cv.VideoCapture = cv.VideoCapture(path + ".mp4")
    if not cap.isOpened():
        raise Exception("The video could not be opened !")
    return cap
