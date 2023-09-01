import cv2
import numpy as np
import os




save_dir = "images1"
os.makedirs(save_dir,exist_ok=True)


video_path = "video1.mkv"

cap = cv2.VideoCapture(video_path)

i = 0
while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break
    cv2.imencode(".jpg", frame)[1].tofile(os.path.join(save_dir, "{}.jpg".format(i)))
    i += 1
    print(i)