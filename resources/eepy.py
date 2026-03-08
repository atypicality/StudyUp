import cv2
import random
import mediapipe
import numpy as np
import torch
import pygame

pygame.mixer.init()
pygame.mixer.music.load("resources/diddyblud.mp3")
ALARM_PLAYING = False 

from ultralytics import YOLO

device = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO("yolov8n.pt")
model.to(device)

mp_face = mediapipe.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

camera = cv2.VideoCapture(0)

knownObjects = {}

sleep_frames = 0
missing_frames = 0
baseline_head = None

EAR_THRESHOLD = 0.20
FRAME_THRESHOLD = 30
HEAD_DROP_THRESHOLD = 80

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]


def ear(eye_points, landmarks, frame):
    h, w, _ = frame.shape

    points = []
    for i in eye_points:
        x = int(landmarks[i].x * w)
        y = int(landmarks[i].y * h)
        points.append((x, y))

    p1,p2,p3,p4,p5,p6 = points

    vertical1 = np.linalg.norm(np.array(p2) - np.array(p6))
    vertical2 = np.linalg.norm(np.array(p3) - np.array(p5))
    horizontal = np.linalg.norm(np.array(p1) - np.array(p4))

    if horizontal == 0:
        return 0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def drawDetections(img, detections, threshold):

    global knownObjects
    person_detected = False

    for box in detections.boxes:

        if float(box.conf) > threshold:

            objClass = int(box.cls[0])

            if model.names[objClass] == "person":
                person_detected = True

            if objClass not in knownObjects:
                knownObjects[objClass] = (
                    random.randint(0,255),
                    random.randint(0,255),
                    random.randint(0,255)
                )

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = knownObjects[objClass]

            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)

            label = f"{model.names[objClass]} {box.conf[0]:.2f}"
            cv2.putText(img,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,color,2)

    return person_detected


torch.set_grad_enabled(False)

while True:

    ret, frame = camera.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # YOLO detection
    yolo_results = model.predict(frame, device=device, verbose=False)
    detections = yolo_results[0]

    person_detected = drawDetections(frame, detections, 0.5)

    if person_detected:
        missing_frames = 0
    else:
        missing_frames += 1

    # Face detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb)

    head_drop = 0

    if face_results.multi_face_landmarks:

        for face in face_results.multi_face_landmarks:

            # ----- (eye detection) -----
            left = ear(LEFT_EYE, face.landmark, frame)
            right = ear(RIGHT_EYE, face.landmark, frame)
            avg_ear = (left + right) / 2

            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            if avg_ear < EAR_THRESHOLD:
                sleep_frames += 1
            else:
                sleep_frames = 0

            # ----- Head Drop Detection fuh -----
            top_head = face.landmark[10]

            head_x = int(top_head.x * w)
            head_y = int(top_head.y * h)

            cv2.circle(frame,(head_x,head_y),5,(255,0,0),-1)

            if baseline_head is None:
                baseline_head = head_y

            head_drop = head_y - baseline_head

            cv2.putText(frame, f"HEAD DROP: {head_drop}", (30,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # ----- Status detection :)-----
    if sleep_frames > FRAME_THRESHOLD or missing_frames > 20 or head_drop > HEAD_DROP_THRESHOLD:
        status = "SLEEPING"
        if not ALARM_PLAYING:
            pygame.mixer.music.play(-1)
            ALARM_PLAYING = True
    else:
        status = "Doing Thingies"
        if ALARM_PLAYING:
            pygame.mixer.music.stop()
            ALARM_PLAYING = False

    cv2.putText(frame, status, (50,120),
                cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0,0,255) if status != "Doing Thingies" else (0,255,0),
                3)

    cv2.imshow("Sleep Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()