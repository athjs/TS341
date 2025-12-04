"""Videos trating packages.

The utils package provides an API call to ease video opening etc...
"""

from typing import Generator
from typing import Sequence
import cv2 as cv
import utils as u
import numpy as np
from threading import Thread
from queue import Queue
import logging

logger = logging.getLogger(__name__)

display_q: Queue = Queue(maxsize=1)


def display_loop() -> None:
    """Displays the current frame."""
    item: np.ndarray
    frame: np.ndarray
    while True:
        item = display_q.get()
        if item is None:
            return
        frame = item
        cv.imshow("Frame", frame)
        if cv.waitKey(30) & 0xFF == ord("q"):
            return


def get_movings(
    cap: cv.VideoCapture, fgbg, kernel
) -> Generator[tuple[int, list[tuple[int, int]], np.ndarray], None, None]:
    """Take video stream, kernel for the operations and video without background.

    Return the frame id, centroids for each moving object and a frame to show
    """
    logger.info("Running centroids.")
    while True:
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()
        if not ret:
            raise Exception("An error occured during the video")

        fgmask: np.ndarray = fgbg.apply(frame)
        threshold: np.ndarray
        frame_id = int(cap.get(1))
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        _, threshold = cv.threshold(fgmask.copy(), 121, 255, cv.THRESH_BINARY)
        contours: Sequence[np.ndarray] = get_contours(threshold)
        centroids: list[tuple[int, int]] = get_moving_centroïds(contours)
        frame_id: int = int(cap.get(1))

        yield frame_id, centroids, frame


def write_movings(path: str):
    """Open the video and remove background.

    Return the video without background
    only with moving objects
    """
    cap: cv.VideoCapture = u.openvideo(path)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    fgbg = cv.createBackgroundSubtractorMOG2()
    t: Thread = Thread(target=display_loop, daemon=True)
    t.start()

    for frame_id, centroids, frame in get_movings(cap, fgbg, kernel):
        try:
            display_q.put_nowait(frame)
        except:
            pass
    display_q.put(None)
    t.join()
    cap.release()
    cv.destroyAllWindows()


def get_contours(threshold: np.ndarray) -> Sequence[np.ndarray]:
    """Returns contours of objects in a video thresholded."""
    contours: Sequence[np.ndarray]
    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def get_moving_centroïds(contours: Sequence[np.ndarray]) -> list[tuple[int, int]]:
    """Returns centroids of each object contour of the video."""
    centroids: list[tuple[int, int]] = []
    logging.basicConfig(filename="myapp.log", level=logging.INFO)
    for contour in contours:
        area: float = cv.contourArea(contour)
        # draw a contour only if it is big enough
        if 10 < area < 1500:
            M: dict[str, float] = cv.moments(contour)
            # shows x and y coordinates
            cx: int = int(M["m10"] / M["m00"])
            cy: int = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
    return centroids


write_movings("videos/video3")
# remove_background("videos/video3")
