# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(451, 238)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.label_title = QtWidgets.QLabel(Form)
        self.label_title.setMaximumSize(QtCore.QSize(16777215, 60))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setAlignment(QtCore.Qt.AlignCenter)
        self.label_title.setObjectName("label_title")
        self.gridLayout.addWidget(self.label_title, 0, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.spinBox_displayNum = QtWidgets.QSpinBox(Form)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.spinBox_displayNum.setFont(font)
        self.spinBox_displayNum.setMinimum(1)
        self.spinBox_displayNum.setMaximum(5)
        self.spinBox_displayNum.setObjectName("spinBox_displayNum")
        self.horizontalLayout.addWidget(self.spinBox_displayNum)
        self.gridLayout.addLayout(self.horizontalLayout, 1, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton_start = QtWidgets.QPushButton(Form)
        self.pushButton_start.setMinimumSize(QtCore.QSize(150, 50))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton_start.setFont(font)
        self.pushButton_start.setObjectName("pushButton_start")
        self.horizontalLayout_2.addWidget(self.pushButton_start)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.pushButton_quit = QtWidgets.QPushButton(Form)
        self.pushButton_quit.setMinimumSize(QtCore.QSize(150, 50))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton_quit.setFont(font)
        self.pushButton_quit.setObjectName("pushButton_quit")
        self.horizontalLayout_2.addWidget(self.pushButton_quit)
        self.gridLayout.addLayout(self.horizontalLayout_2, 2, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 3, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_title.setText(_translate("Form", "Ai Attention for Mmulti Screen"))
        self.label.setText(_translate("Form", "Display number"))
        self.pushButton_start.setText(_translate("Form", "Start"))
        self.pushButton_quit.setText(_translate("Form", "Quit"))

