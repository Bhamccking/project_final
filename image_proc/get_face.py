import cv2
import mediapipe as mp
import numpy as np
import os

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

img_dir = "C:\\Users\\aord23\\Desktop\\images\\侧脸\\"
save_dir = "../dataset/face4"
os.makedirs(save_dir,exist_ok=True)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(model_selection=0,
                                     min_detection_confidence=0.5
                                     ) as face_detection:
    files = os.listdir(img_dir)
    if len(files):
        for i in range(len(files)):
            # file = "{}.jpg".format(i)
            try:
                file = files[i]
                path = os.path.join(img_dir,file)
                buf = np.fromfile(path,np.uint8)
                img = cv2.imdecode(buf,1)
                img.flags.writeable = False
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                imh,imw = img.shape[:2]
                # img1 = rgb[0:imh,0:imw//2]
                # img2 = rgb[0:imh,imw//2:]
                # for j,image in enumerate([img1,img2]):
                results = face_detection.process(rgb)
                # Draw the face detection annotations on the image.
                img.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.detections:
                    for j,detection in enumerate(results.detections):
                        # mp_drawing.draw_detection(image, detection)
                        # print(type(detection))
                        location_data = detection.location_data.relative_bounding_box
                        x = int(location_data.xmin*(imw))
                        y = int(location_data.ymin*(imh))
                        w = int(location_data.width*(imw))
                        h = int(location_data.height*(imh))

                        try:
                            imface = img[y:y+h,x:x+w]
                            path = os.path.join(save_dir,"{}_{}.jpg".format(i,j))
                            cv2.imencode(".jpg", imface)[1].tofile(path)
                        except:
                            pass
                print(i)
            except:
                pass
            # cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
            # cv2.waitKey(25)