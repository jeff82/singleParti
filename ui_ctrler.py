# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 22:27:42 2018

@author: Administrator
"""

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from uis.mainwindow import Ui_MainWindow


class UiCtrler:
    
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()