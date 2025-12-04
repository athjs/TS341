"""Permet d'essayer un modele pour générer les résultats formaté à l'évaluation."""

from ultralytics import YOLO
import cv2



start_frame = 0
end_frame = 2285

start_frame = int(start_frame)
end_frame = int(end_frame)

def write_result(nom_model, frame, x, y, f):
    f.write(str(frame)+","+str(x)+","+str(y)+"\n")

def test_model(conf_min, model, video,nom_model):
    f = open("ts341_project/benchmark/model_results/"+nom_model+".csv","w")
    while True:
        frame_id = int(video.get(cv2.CAP_PROP_POS_FRAMES))

        if frame_id > end_frame:
            break

        ret, frame = video.read()
        if not ret:
            break

        results = model.predict(frame, verbose=False)
        # S'il n'y a aucune prédiction dans results
        if len(results) == 0 or len(results[0].boxes) == 0:
            print("Aucune prédiction")
            write_result(nom_model, frame_id, -1, -1, f)
        else:
            # Parcourt des prédictions
            for box in results[0].boxes:
                # Probabilité (confiance)
                conf = float(box.conf[0])

                # Coordonnées (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                if conf > conf_min :
                    print(f"Proba: {conf:.3f} | Coordonnées: x1={x1}, y1={y1}")
                    write_result(nom_model, frame_id, x1, y1, f)
                else :
                    print("Prediction too low")
                    write_result(nom_model, frame_id, -1, -1, f)
                    
model = YOLO("ts341_project/model_weights/kaggle_dataset.pt")
video = cv2.VideoCapture("videos/capture_cloudy-daylight_True_10_03_14_35_15_cam1.mp4")
video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
test_model(0.3, model, video,"Kaggle_dataset")

video.release()
