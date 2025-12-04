import os
from ultralytics import YOLO
import cv2
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(CURRENT_DIR, "..", "..", "videos")
sys.path.append(VIDEO_PATH)

def run_yolo(video_path=os.path.join(VIDEO_PATH, "video2.mp4")):
    
    MODEL_PATH = os.path.join(CURRENT_DIR, "kaggle_training.pt")

    model = YOLO(MODEL_PATH)

    frame_number=0

    for result in model.predict(source=video_path, stream=True, verbose=False):
        frame = result.plot()
        original_height, original_width = frame.shape[:2]
        frame_resized = cv2.resize(frame, (1280, 720))  
        height, width = frame_resized.shape[:2]

        height_ratio_resize = height/original_height
        width_ratio_resize = width/original_width

        # cv2.imshow("Resultat YOLO", frame_resized)

        # Récupérer les coordonnées et infos
        detections = result.boxes.xyxy.cpu().numpy()  
        confidences = result.boxes.conf.cpu().numpy() 
        class_ids = result.boxes.cls.cpu().numpy()  

        # On renvoie le tout **pour chaque frame**
        yield frame_number, frame_resized, height_ratio_resize, width_ratio_resize, detections, confidences, class_ids
        frame_number+=1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()