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
    # For the creation of the video without backgorund.
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
        # Video without background
        fgmask = fgbg.apply(frame)
        # thresholding the image to remove some noises
        ret, threshold = cv.threshold(fgmask.copy(), 120, 255, cv.THRESH_BINARY)
        # removing some useless points (KNN like)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        # find countours of moveable objects
        contours, hier = cv.findContours(
            threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        # check every contour if are exceed certain value draw bounding boxes
        for contour in contours:
            # if area exceed certain value then draw bounding boxes
            if cv.contourArea(contour) > 20 and cv.contourArea(contour) < 2500 :
                (x, y, w, h) = cv.boundingRect(contour)
                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        # add the frame to the video without background
        out.write(fgmask)
        # open the raw video with rectangles on moving objects
        cv.imshow("Frame", frame)
        if cv.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()
    return


removeBackground("videos/video2")
