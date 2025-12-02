import os
from ultralytics import YOLO
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_yolo(video_path=os.path.join(CURRENT_DIR, "capture_cloudy-daylight_True_10_03_14_35_15_cam1.mp4")):
    
    MODEL_PATH = os.path.join(CURRENT_DIR, "kaggle_training.pt")

    model = YOLO(MODEL_PATH)

    for result in model.predict(source=video_path, stream=True, verbose=False):
        frame = result.plot()
        frame_resized = cv2.resize(frame, (1280, 720))  

        cv2.imshow("Resultat YOLO", frame_resized)

        # Récupérer les coordonnées et infos
        detections = result.boxes.xyxy.cpu().numpy()  
        confidences = result.boxes.conf.cpu().numpy() 
        class_ids = result.boxes.cls.cpu().numpy()  

        # On renvoie le tout **pour chaque frame**
        yield detections, confidences, class_ids

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()