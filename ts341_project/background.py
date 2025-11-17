import cv2 as cv
import utils


def removeBackground(path: str) -> cv.VideoCapture:
    """Open the video and remove background.

    Return the video without background
    only with moving objects
    """
    cap: cv.VideoCapture = utils.openvideo(path)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    fgbg = cv.createBackgroundSubtractorMOG2()
    frame_width: int = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height: int = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out: cv.VideoWriter = cv.VideoWriter(
        "videos/video2wb.mp4", fourcc, 30.0, (frame_width, frame_height), isColor=False
    )
    while 1:
        ret, frame = cap.read()
        if not ret:
            raise Exception("An error occured during the video")
        fgmask = fgbg.apply(frame)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        out.write(fgmask)
        cv.imshow("Frame", fgmask)
        if cv.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()
    return


removeBackground("videos/video2")
