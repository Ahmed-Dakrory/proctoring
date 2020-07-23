from PyQt5.QtWidgets import QApplication
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from PyQt5.QtCore import QUrl
from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox

import subprocess

#Check VMWARE
batcmd='systeminfo /s %computername% | findstr /c:"Model:" /c:"Host Name" /c:"OS Name"'
result = subprocess.check_output(batcmd, shell=True)
print(result)


class WebEnginePage(QWebEnginePage):
    def __init__(self, *args, **kwargs):
        QWebEnginePage.__init__(self, *args, **kwargs)
        self.featurePermissionRequested.connect(self.onFeaturePermissionRequested)

    def onFeaturePermissionRequested(self, url, feature):
        if feature in (QWebEnginePage.MediaAudioCapture, 
            QWebEnginePage.MediaVideoCapture, 
            QWebEnginePage.MediaAudioVideoCapture):
            self.setFeaturePermission(url, feature, QWebEnginePage.PermissionGrantedByUser)
        else:
            self.setFeaturePermission(url, feature, QWebEnginePage.PermissionDeniedByUser)

def showDialog():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText("Error")
    msg.setInformativeText('Please Dont Run From Vmware')
    msg.setWindowTitle("Error")
    msg.exec_()


app = QApplication([])
if "VMware" in str(result):
    showDialog()
else:
    
    view = QWebEngineView()

    view.setWindowFlags(QtCore.Qt.FramelessWindowHint)
    view.showFullScreen()

    page = WebEnginePage()
    view.setPage(page)
    view.load(QUrl("http://127.0.0.1:8000/"))
    view.show()
    app.exec_()