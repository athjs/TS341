"""Utils for opening videos."""

import cv2 as cv
import logging

logger = logging.Logger(__name__)


def openvideo(path: str) -> cv.VideoCapture:
    """Open a video and returns it or raise an error."""
    logging.basicConfig(filename="myapp.log", level=logging.INFO)
    logger.info("Openning Video")
    cap: cv.VideoCapture = cv.VideoCapture(path + ".mp4")
    if not cap.isOpened():
        raise Exception("The video could not be opened !")
    return cap
