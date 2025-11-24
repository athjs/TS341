# Code généré avec l'aide de ChatGPT

import cv2
import csv

# --- paramètres ---
video_path = "videos/capture_cloudy-daylight_True_10_03_14_35_15_cam1.mp4"
output_csv = "ts341_project/test_results.csv"

# --- CSV ---
scale = 0.5  # ← fenêtre 50% de la taille originale

csv_file = open(output_csv, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["frame", "x", "y"])

cap = cv2.VideoCapture(video_path)

frame_id = 0
clicked = False
click_x, click_y = None, None

def mouse_callback(event, x, y, flags, param):
    """Récupère le click de la souris sur l'image.

    Args:
        x (int): x du pixel cliqué
        y (int): y du pixel cliqué
    """
    global clicked, click_x, click_y, scale
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True
        # conversion vers coordonnées réelles
        click_x = int(x / scale)
        click_y = int(y / scale)

cv2.namedWindow("Annotation")
cv2.setMouseCallback("Annotation", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    clicked = False

    # redimension pour affichage
    display = cv2.resize(frame, None, fx=scale, fy=scale)

    while True:
        cv2.putText(display,
                    f"Frame {frame_id} - clic=pos | d=no det | n=skip | q=quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Annotation", display)
        key = cv2.waitKey(10)

        # clic sur la version réduite
        if clicked:
            # point affiché seulement sur la version réduite
            cv2.circle(display,
                       (int(click_x * scale), int(click_y * scale)),
                       5, (0, 0, 255), -1)
            cv2.imshow("Annotation", display)
            cv2.waitKey(150)

            writer.writerow([frame_id, click_x, click_y])
            break

        # 'd' = pas de détection
        if key == ord('d'):
            writer.writerow([frame_id, -1, -1])
            break

        # 'n' = skip
        if key == ord('n'):
            break

        # 'q' = quit
        if key == ord('q'):
            cap.release()
            csv_file.close()
            cv2.destroyAllWindows()
            exit()

    frame_id += 1

cap.release()
csv_file.close()
cv2.destroyAllWindows()

print("Annotations saved in:", output_csv)