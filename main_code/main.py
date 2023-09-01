


"""

main win code

"""


from video_win import VideoWin
from PyQt5 import QtCore,QtGui,QtWidgets
import os
import sys
import cv2
from main_ui import Ui_Form
import copy
import math
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

log_dir = "log"
os.makedirs(log_dir,exist_ok=True)


# DISPLAY_NUM = 2

#
class MainWin(QtWidgets.QWidget,Ui_Form):
    signal_exit = QtCore.pyqtSignal(bool)
    def __init__(self, parent=None):
        super(MainWin, self).__init__(parent)
        # setup ui
        self.setupUi(self)
        # init win
        self.setWindowTitle("Ai Attention")
        self.label_title.setText("Deep Learning Attention for Mmulti Screen")

        self.pushButton_start.clicked.connect(self.slot_start)
        self.pushButton_quit.clicked.connect(self.slot_quit)

        self.subwins = {
            0: None,
            1: None,
            2: None,
            3: None,
            4: None,
        }
        try:
            self.cams = []
            for i in range(5):
                cam = cv2.VideoCapture(i)
                self.cams.append(cam)
        except Exception as e:
            print(e.args)

    def slot_start(self):
        """"""
        display_num = self.spinBox_displayNum.value()
        if display_num<1:
            QtWidgets.QMessageBox.warning(self,"warning","screen num must >1 ")
            return
        self.subwins = {
            0: None,
            1: None,
            2: None,
            3: None,
            4: None,
        }
        for i in range(display_num):
            self.subwins[i] = VideoWin(win_title="win_{}".format(i),
                           video_temp="video_{}.jpg".format(i),
                           camera_temp="camera_{}.jpg".format(i),
                           cam_id=i,
                           cam = self.cams[i]
                           )
            self.subwins[i].move(150*i,150*i)
            self.subwins[i].signal_exit.connect(self.slot_subquit)
            self.subwins[i].show()

        self.hide()

    def slot_subquit(self,id):
        self.subwins[id].close()
        self.subwins[id] = None

        all_quit = True
        for i in range(5):
            if self.subwins[i] is not None:
                all_quit = False

        if all_quit:
            self.show()


    def closeEvent(self, event):
        """quit"""
        # exit_flag = QtWidgets.QMessageBox.warning(self, 'info', "quit now？",QtWidgets.QMessageBox.Yes | (QtWidgets.QMessageBox.No))
        # if exit_flag == QtWidgets.QMessageBox.Yes:
        #     self.signal_exit.emit(True)
        #     self.hide()
        #     event.accept()
        # else:
        #     event.ignore()
        self.close()

    def slot_quit(self):
        self.close()


if __name__ == "__main__":
    print("")
    app = QtWidgets.QApplication(sys.argv)
    win = MainWin()
    win.show()
    sys.exit(app.exec_())