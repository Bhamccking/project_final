from PyQt5 import QtCore,QtGui,QtWidgets
import os
import sys
import cv2
from video_ui import Ui_Form
import copy
import math
import numpy as np
import time
import warnings
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import mediapipe as mp

log_dir = "log"
os.makedirs(log_dir,exist_ok=True)


class_names = ["close","open"]
pause_idx = 1
input_channels = 1
model_path = "./weights/model.onnx"

COUNT_FRAME = 25
PAUSE_THRESH = 15

ATT_THRESH = 0.8


def softmax(scores):
    p = np.exp(scores) / np.sum(np.exp(scores))
    return p


class clsModel():
    def __init__(self,model_path):
        if not os.path.exists(model_path):
            self.cls_model = None
        else:
            self.cls_model = cv2.dnn.readNetFromONNX(model_path)
        self.classes = class_names

    def infer(self, roi_img):
        if self.cls_model is None:
            return (0, 0)
        """识别"""
        if roi_img is None:
            return (0, 0)
        img_input = cv2.resize(roi_img, (224, 224))
        if len(img_input.shape)==3:
            if input_channels==1:
                img_input = cv2.cvtColor(img_input,cv2.COLOR_BGR2GRAY)
            else:
                pass
        else:
            if input_channels==3:
                img_input = cv2.cvtColor(img_input,cv2.COLOR_GRAY2BGR)
            else:
                pass
        blob = cv2.dnn.blobFromImage(img_input, 1 / 255.0, (224, 224), crop=False)
        self.cls_model.setInput(blob)
        scores = self.cls_model.forward()[0]
        scores = softmax(scores)
        id = np.argmax(scores)
        prob = scores[id]
        if id==1:
            if prob<ATT_THRESH:
                id = 0
                prob = 1 - prob
        return (id, prob)


class VideoThread(QtCore.QThread):
    signal_log = QtCore.pyqtSignal(str)
    def __init__(self,parent=None):
        super(VideoThread, self).__init__(parent)
        self.ui = parent
        self.play = False


    def set_path(self,video_path):
        self.videl_path = video_path

    def slot_play(self,flag):
        self.play = flag

    def stop(self):
        self.stop_run = False

    def run(self):
        """run video"""
        try:
            if not os.path.exists(self.videl_path):
                self.signal_log.emit("{} not found".format(self.videl_path))
                return
            cap = cv2.VideoCapture(self.videl_path)
            current_frame_id = 0
            self.stop_run = False
            while cap.isOpened():
                if self.stop_run:
                    break
                if not self.play:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_id)
                else:
                    ret,frame = cap.read()
                    if not ret:
                        # self.signal_log.emit("get frame error")
                        self.ui.label_video.setPixmap(
                            QtGui.QPixmap(os.path.join(log_dir, self.ui.video_temp)).scaled(self.ui.video_w,
                                                                                            self.ui.video_h))
                    else:
                        cv2.imencode(".jpg", frame)[1].tofile(os.path.join(log_dir, self.ui.video_temp))
                        self.ui.label_video.setPixmap(
                            QtGui.QPixmap(os.path.join(log_dir, self.ui.video_temp)).scaled(self.ui.video_w,
                                                                                             self.ui.video_h))

                        current_frame_id += 1
                time.sleep(1/25.0) #
            cap.release()
        except Exception as e:
            print("VideoThread Error：",e.args)



class CameraThread(QtCore.QThread):
    signal_log = QtCore.pyqtSignal(str)
    signal_play = QtCore.pyqtSignal(bool)
    def __init__(self,parent=None):
        super(CameraThread, self).__init__(parent)
        self.ui = parent
        self.model = clsModel(model_path=model_path)
        self.playing = False
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

        
    def set_cap(self,cam):
        self.cam_cap = cam

    def stop(self):
        self.stop_run = False

    def run(self):
        """run camera and detect"""
        try:
            self.stop_run = False
            with self.mp_face_detection.FaceDetection(model_selection=0,min_detection_confidence=0.5) as face_detection:
                flags = []
                while self.cam_cap.isOpened():
                    if self.stop_run:
                        break
                    ret,frame = self.cam_cap.read()
                    if not ret:
                        continue
                    """detect face"""
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    imh,imw = rgb.shape[:2]
                    results = face_detection.process(rgb)
                    imface = None
                    if results.detections:
                        for detection in results.detections:
                            # self.mp_drawing.draw_detection(rgb, detection)
                            location_data = detection.location_data.relative_bounding_box
                            x = int(location_data.xmin * imw)
                            y = int(location_data.ymin * imh)
                            w = int(location_data.width * imw)
                            h = int(location_data.height * imh)
                            imface = frame[y:y+h,x:x+w]
                            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                    # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    """classify face: open or close video"""
                    if imface is not None:
                        try:
                            (idx,prob) = self.model.infer(imface)
                            flags.append(idx)
                        except Exception as e:
                            idx = 0
                            prob = 0
                            # self.signal_log.emit("model infer error")
                            # print(e.args)
                        cv2.putText(frame,"{}".format(class_names[idx]),(x,y),1,2,(0,0,255),2)

                        if len(flags)>COUNT_FRAME:
                            flags.pop(0)

                        pause = False
                        if flags.count(0)>PAUSE_THRESH:
                            pause = True
                        if pause:
                            if self.playing:
                                self.signal_play.emit(False)
                                self.signal_log.emit("pause")
                                self.playing = False
                        else:
                            if not self.playing:
                                self.signal_play.emit(True)
                                self.signal_log.emit("play")
                                self.playing = True


                    """show image"""
                    cv2.imencode(".jpg", frame)[1].tofile(os.path.join(log_dir, self.ui.camera_temp))
                    self.ui.label_camera.setPixmap(
                        QtGui.QPixmap(os.path.join(log_dir, self.ui.camera_temp)).scaled(self.ui.cam_w, self.ui.cam_h))

        except Exception as e:
            print("CAM ERROR:",e.args)


#
class VideoWin(QtWidgets.QWidget,Ui_Form):
    signal_exit = QtCore.pyqtSignal(bool)
    def __init__(self, parent=None,
                 win_title="",
                 video_temp="",
                 camera_temp="",
                 cam_id=0,
                 cam = None,
                 ):
        super(VideoWin, self).__init__(parent)
        # setup ui
        self.setupUi(self)
        # init win
        self.setWindowTitle(win_title)
        self.video_w = self.label_video.width()
        self.video_h = self.label_video.height()
        self.cam_w = self.label_camera.width()
        self.cam_h = self.label_camera.height()
        self.thread_video = VideoThread(self)
        self.thread_camera = CameraThread(self)
        self.pushButton_loadVideo.clicked.connect(self.slot_load)

        self.video_temp = video_temp
        self.camera_temp = camera_temp
        self.cam_id = cam_id
        self.cam = cam

        # signal
        self.thread_camera.signal_play.connect(self.slot_play)
        self.thread_video.signal_log.connect(self.slot_log)
        self.thread_camera.signal_log.connect(self.slot_log)


    def slot_play(self,flag):
        self.thread_video.slot_play(flag)

    def slot_load(self):
        """load video file"""
        # self.video_path = ""
        video_path = QtWidgets.QFileDialog.getOpenFileName(self, "video", "", "*.mp4")[0]
        if video_path != "":
            self.lineEdit_loadVideo.setText(video_path)
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                ret,frame = cap.read()
                if not ret:
                    QtWidgets.QMessageBox.critical(self,"wrong","video error!!!")
                    return
                path = os.path.join(log_dir,self.video_temp)
                cv2.imencode(".jpg", frame)[1].tofile(path)
                self.label_video.setPixmap(QtGui.QPixmap(os.path.join(log_dir,self.video_temp)).scaled(self.video_w, self.video_h))
                # os.remove(os.path.join(log_dir,self.video_temp))

                self.thread_video.set_path(video_path)
                self.thread_video.play = False
                self.thread_video.start()

                self.thread_camera.set_cap(self.cam)
                self.thread_camera.start()


    def slot_log(self,string):
        try:
            time_string = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(time.time()))
            line = "[{}]{}".format(time_string,string)
            self.textBrowser_log.append(line)
        except:
            pass

    def closeEvent(self, event):
        """quit"""
        exit_flag = QtWidgets.QMessageBox.warning(self, 'info', "quit now？",QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if exit_flag == QtWidgets.QMessageBox.Yes:
            self.thread_camera.stop()
            self.thread_video.stop()
            # self.thread_camera.wait()
            # self.thread_video.wait()
            self.signal_exit.emit(self.cam_id)
            self.close()
            event.accept()
        else:
            event.ignore()


# if __name__ == "__main__":
#     print("")
#     app = QtWidgets.QApplication(sys.argv)
#     login = VideoWin()
#     login.show()
#     sys.exit(app.exec_())