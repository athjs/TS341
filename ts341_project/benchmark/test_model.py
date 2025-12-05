"""Permet d'essayer un modele pour générer les résultats formaté à l'évaluation."""

from ultralytics import YOLO
import cv2

start_frame = 0
end_frame = 2295

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
            write_result(nom_model, frame_id, -1, -1, f)
        else:
            box = results[0].boxes 
            # Probabilité (confiance)
            conf = float(box.conf[0])

            # Coordonnées (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy, w, h = box.xywh[0].tolist()
            if conf > conf_min :
                write_result(nom_model, frame_id, int(cx), int(cy), f)
            else :
                write_result(nom_model, frame_id, -1, -1, f)
        print("Conf_min :",conf_min, "| Progression :", int((frame_id-start_frame)/(end_frame-start_frame)*100),"%")
    f.close()
                    
model = YOLO("ts341_project/model_weights/kaggle_dataset.pt")
video = cv2.VideoCapture("videos/capture_cloudy-daylight_True_10_03_14_35_15_cam1.mp4")
video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
for seuil in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    print("########################### ",seuil)
    test_model(seuil, model, video,"Kaggle_dataset_"+str(seuil))
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

video.release()
