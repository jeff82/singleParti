# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(633, 412)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(270, 90, 54, 12))
        self.label.setObjectName("label")
        self.pbOpen = QtWidgets.QPushButton(self.centralwidget)
        self.pbOpen.setGeometry(QtCore.QRect(20, 20, 75, 23))
        self.pbOpen.setObjectName("pbOpen")
        self.pb_pause = QtWidgets.QPushButton(self.centralwidget)
        self.pb_pause.setGeometry(QtCore.QRect(20, 120, 75, 23))
        self.pb_pause.setObjectName("pb_pause")
        self.pbStart = QtWidgets.QPushButton(self.centralwidget)
        self.pbStart.setGeometry(QtCore.QRect(20, 80, 75, 23))
        self.pbStart.setObjectName("pbStart")
        self.pbdefault = QtWidgets.QPushButton(self.centralwidget)
        self.pbdefault.setGeometry(QtCore.QRect(20, 160, 75, 23))
        self.pbdefault.setObjectName("pbdefault")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(30, 60, 54, 12))
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 633, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "TextLabel"))
        self.pbOpen.setText(_translate("MainWindow", "open"))
        self.pb_pause.setText(_translate("MainWindow", "pause"))
        self.pbStart.setText(_translate("MainWindow", "start"))
        self.pbdefault.setText(_translate("MainWindow", "default"))
        self.label_2.setText(_translate("MainWindow", "TextLabel"))

