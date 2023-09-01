# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'video_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1040, 547)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.label_video = QtWidgets.QLabel(Form)
        self.label_video.setMinimumSize(QtCore.QSize(640, 480))
        self.label_video.setFrameShape(QtWidgets.QFrame.Box)
        self.label_video.setAlignment(QtCore.Qt.AlignCenter)
        self.label_video.setObjectName("label_video")
        self.gridLayout.addWidget(self.label_video, 0, 0, 1, 1)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_camera = QtWidgets.QLabel(Form)
        self.label_camera.setMinimumSize(QtCore.QSize(320, 240))
        self.label_camera.setFrameShape(QtWidgets.QFrame.Box)
        self.label_camera.setAlignment(QtCore.Qt.AlignCenter)
        self.label_camera.setObjectName("label_camera")
        self.verticalLayout_3.addWidget(self.label_camera)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_loadVideo = QtWidgets.QPushButton(Form)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_loadVideo.setFont(font)
        self.pushButton_loadVideo.setObjectName("pushButton_loadVideo")
        self.horizontalLayout.addWidget(self.pushButton_loadVideo)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.lineEdit_loadVideo = QtWidgets.QLineEdit(Form)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_loadVideo.setFont(font)
        self.lineEdit_loadVideo.setReadOnly(True)
        self.lineEdit_loadVideo.setObjectName("lineEdit_loadVideo")
        self.verticalLayout.addWidget(self.lineEdit_loadVideo)
        self.verticalLayout_3.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_3 = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.textBrowser_log = QtWidgets.QTextBrowser(Form)
        self.textBrowser_log.setEnabled(True)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.textBrowser_log.setFont(font)
        self.textBrowser_log.setObjectName("textBrowser_log")
        self.verticalLayout_2.addWidget(self.textBrowser_log)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.gridLayout.addLayout(self.verticalLayout_3, 0, 1, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_video.setText(_translate("Form", "video"))
        self.label_camera.setText(_translate("Form", "camera"))
        self.pushButton_loadVideo.setText(_translate("Form", "load video"))
        self.label_3.setText(_translate("Form", "log"))

