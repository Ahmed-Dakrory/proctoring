# python -m fbs freeze --debug
# fbs freeze
from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic

import sys
import cv2
import base64
import imutils
import tensorflow as tf
import numpy as np
import requests


# Recognition Thread
import os
import cv2
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
import pickle
import numpy as np
# from keras.models import load_model
from tensorflow.keras.models import model_from_json


# from imutils import face_utils
from imutils.face_utils import FaceAligner

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer




################################
# Those for the Devices Checking
import pyaudio
import subprocess
import re
import wmi
import urllib
import ctypes
from ctypes import *

kernel32 = ctypes.WinDLL('kernel32')
user32 = ctypes.WinDLL('user32')




######################################################
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC

# file to save model parameters in
classifierfilename = 'classifier/svmrbf.sav'

#######################################################
#folder to store face images
facepath = 'images'
CNN_INPUT_SIZE = 128
ANGLE_THRESHOLD = 0.15
IMAGE_PER_POSE=5
FACE_WIDTH = 160




mark_detector = MarkDetector() 
        
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=FACE_WIDTH)

class GUI(QMainWindow):
    def __init__(self,appctxt):
        super(GUI,self).__init__()
        self.stepNow = 0
        self.checkInternetTrial = 0

        uic.loadUi('main.ui', self) # Load the .ui file
        self.setWindowFlags(Qt.FramelessWindowHint)

        # Adapte Views
        # set Internet gid
        movie = QMovie("imageface/internet.gif")
        self.internetChecking.setMovie(movie)
        movie.start()

        # set Success Devices
        self.bar_cam.setVisible(False)
        self.bar_mouse.setVisible(False)
        self.bar_key.setVisible(False)
        self.bar_speaker.setVisible(False)
        self.bar_micro.setVisible(False)
        self.bar_system.setVisible(False)

        self.success_cam.setVisible(False)
        self.success_mouse.setVisible(False)
        self.success_key.setVisible(False)
        self.success_speaker.setVisible(False)
        self.success_micro.setVisible(False)

        self.predictButton.clicked.connect(self.goNextStep)
        self.exitButton.clicked.connect(self.close)
        self.minimizeButton.clicked.connect(lambda: self.showMinimized())
        
        self.show()
        self.exit_code = appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()
        

    def close(self):
        QCoreApplication.exit(0)

    def goToErrorPage(self):
        error_dialog = QErrorMessage()
        error_dialog.showMessage('Oh no! Problem With Internet')
        error_dialog.exec_()

    def goNextStep(self):
        self.predictButton.setStyleSheet("""QPushButton{background-color: rgb(190, 188, 188);
        border-style: outset;
        border-width: 1px;
        border-radius: 5px;
        border-color: #e8e8e8;
        padding: 4px;
        color: #fbfbfb;
        font-size: 15px;
        font-weight: 700;}""")
        self.predictButton.setEnabled(False)
        self.pagerWindow.setCurrentIndex(self.stepNow)
        if self.stepNow == 0:
            print("Checking Internet...")

            loop = QEventLoop()
            QTimer.singleShot(1000, loop.quit)
            loop.exec_()
            if self.checkInternetConnection():
                self.stepNow +=1
                self.goNextStep()
            else:
                # check internet to 5 times
                if self.checkInternetTrial < 5:
                    self.goNextStep()
                else:
                    self.checkInternetTrial
                    self.goToErrorPage()

        elif self.stepNow == 1:
            print("Internet Success...")
            loop = QEventLoop()
            QTimer.singleShot(2000, loop.quit)
            loop.exec_()
            self.stepNow +=1
            self.goNextStep()
        elif self.stepNow == 2:
            print("Device Checking...")
            self.CheckDevices()
            self.stepNow +=1
            self.goNextStep()
        elif self.stepNow == 3:
            self.predict()
        elif self.stepNow == 4:
            self.predictButton.setEnabled(True)
            self.predictButton.setStyleSheet("""QPushButton{background-color: #0095ff;
            border-style: outset;
            border-width: 1px;
            border-radius: 5px;
            border-color: #e8e8e8;
            padding: 4px;
            color: #fbfbfb;
            font-size: 15px;
            font-weight: 700;}
            
            QPushButton:hover{background-color: #0095ff;
            border-style: outset;
            border-width: 1px;
            border-radius: 5px;
            border-color: #e8e8e8;
            padding: 4px;
            color: #565050;
            font-size: 15px;
            font-weight: 700;}""")

    def checkInternetConnection(self):
        self.checkInternetTrial+=1
        try:
            requests.get('https://www.google.com/').status_code
            return True
        except:
            return False

    #Check VMWARE
    def checkVmWare(self):
        batcmd='systeminfo /s %computername% | findstr /c:"Model:" /c:"Host Name" /c:"OS Name"'
        result = subprocess.check_output(batcmd, shell=True)
        # print(result)

        if re.search('VirtualBox', str(result), re.IGNORECASE):
            return (True)
        else:
            return (False)

    def checkMicrophone(self):
        winmm= windll.winmm
        if winmm.waveInGetNumDevs()>0:
            return True
        else:
            return False

    def checkSpeaker(self):
        p = pyaudio.PyAudio()

        for i in range(0,10):
            try:
                if p.get_device_info_by_index(i)['maxOutputChannels']>0:
                    return True
            except Exception as e:
                print (e)
                return False

    def CheckDevices(self):
        # Check Camera
        cap = cv2.VideoCapture(0) 
        if not (cap is None or not cap.isOpened()):
            self.bar_cam.setVisible(True)
            self.success_cam.setVisible(True)
        

        loop = QEventLoop()
        QTimer.singleShot(500, loop.quit)
        loop.exec_()

        # Check Mouse
        wmiService = wmi.WMI()
        PointingDevices = wmiService.query("SELECT * FROM Win32_PointingDevice")
        if len(PointingDevices)>= 1:
            self.bar_mouse.setVisible(True)
            self.success_mouse.setVisible(True)
            
        loop = QEventLoop()
        QTimer.singleShot(500, loop.quit)
        loop.exec_()

        # Check Keyboard
        keyboards = wmiService.query("SELECT * FROM Win32_Keyboard")
        if len(keyboards) >= 1:
            self.bar_key.setVisible(True)
            self.success_key.setVisible(True)
       
        loop = QEventLoop()
        QTimer.singleShot(500, loop.quit)
        loop.exec_()

        if self.checkSpeaker():
            self.bar_speaker.setVisible(True)
            self.success_speaker.setVisible(True)
        
        loop = QEventLoop()
        QTimer.singleShot(500, loop.quit)
        loop.exec_()


        if self.checkMicrophone():
            self.bar_micro.setVisible(True)
            self.success_micro.setVisible(True)
        
        loop = QEventLoop()
        QTimer.singleShot(500, loop.quit)
        loop.exec_()


        if not self.checkVmWare():
            self.bar_system.setVisible(True)
        
        loop = QEventLoop()
        QTimer.singleShot(1000, loop.quit)
        loop.exec_()



    @pyqtSlot(QImage)
    def setImage(self, image):
        source = QPixmap.fromImage(image)
        output = QPixmap(source.size())
        
        output.fill(Qt.transparent)
        # # create a new QPainter on the output pixmap
        qp = QPainter(output)
        qp.setBrush(QBrush(source))
        qp.setPen(Qt.NoPen)
        qp.drawRoundedRect(output.rect(), 70, 70)
        qp.end()
        self.cameraHolder.setPixmap(output)

    @pyqtSlot(str)
    def setCameraPose(self, title):
        self.followPose.setText(title)
        

    @pyqtSlot(str)
    def goCheckingForPose(self,statues):
        self.stepNow +=1
        self.goNextStep()

    def predict(self):
        print("Start Prediction")
        # self.camera.startCap()
        self.Thread_Of_Prediction_Is_Run = True
        th = ThreadCamera(self)
        th.changePixmap.connect(self.setImage)
        th.setPose.connect(self.setCameraPose)
        th.checkingEnded.connect(self.goCheckingForPose)
        th.start()

        

class ThreadCamera(QThread):
    changePixmap = pyqtSignal(QImage)
    setPose = pyqtSignal(str)
    checkingEnded = pyqtSignal(str)

    def run(self):
        cap = cv2.VideoCapture(0)
        
        
        name = '3'

        poses=['frontal','right','left','up','down']
        file=0
        
        ret, sample_frame = cap.read()
        i = 0
        count = 0
        if ret==False:
            return    
            
        # Introduce pose estimator to solve pose. Get one frame to setup the
        # estimator according to the image size.
        height, width = sample_frame.shape[:2]
        pose_estimator = PoseEstimator(img_size=(height, width))
        
        # Introduce scalar stabilizers for pose.
        pose_stabilizers = [Stabilizer(
            state_num=2,
            measure_num=1,
            cov_process=0.1,
            cov_measure=0.1) for _ in range(6)]
        images_saved_per_pose=0
        number_of_images = 0
        
        while i<5:
            saveit = False
            # Read frame, crop it, flip it, suits your needs.
            ret, frame = cap.read()
            if ret is False:
                break
            if count % 5 !=0: # skip 5 frames
                count+=1
                continue
            if images_saved_per_pose==IMAGE_PER_POSE:
                i+=1
                images_saved_per_pose=0

            # If frame comes from webcam, flip it so it looks like a mirror.
            if file == 0:
                frame = cv2.flip(frame, 2)
            original_frame=frame.copy()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            facebox = mark_detector.extract_cnn_facebox(frame)
        
            if facebox is not None:
                # Detect landmarks from image of 128x128.
                x1=max(facebox[0]-0,0)
                x2=min(facebox[2]+0,width)
                y1=max(facebox[1]-0,0)
                y2=min(facebox[3]+0,height)
                
                face = frame[y1: y2,x1:x2]
                face_img = cv2.resize(face, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
                marks = mark_detector.detect_marks([face_img])
        
                # Convert the marks locations from local CNN to global image.
                marks *= (facebox[2] - facebox[0])
                marks[:, 0] += facebox[0]
                marks[:, 1] += facebox[1]
            
                # Try pose estimation with 68 points.
                pose = pose_estimator.solve_pose_by_68_points(marks)
        
                # Stabilize the pose.
                steady_pose = []
                pose_np = np.array(pose).flatten()
                for value, ps_stb in zip(pose_np, pose_stabilizers):
                    ps_stb.update([value])
                    steady_pose.append(ps_stb.state[0])
                steady_pose = np.reshape(steady_pose, (-1, 3))
        
                if i==0:
                    if abs(steady_pose[0][0])<ANGLE_THRESHOLD and abs(steady_pose[0][1])<ANGLE_THRESHOLD:
                        images_saved_per_pose+=1
                        saveit = True                    
                if i==1:
                    if steady_pose[0][0]>ANGLE_THRESHOLD:
                        images_saved_per_pose+=1
                        saveit = True
                if i==2:
                    if steady_pose[0][0]<-ANGLE_THRESHOLD:
                        images_saved_per_pose+=1
                        saveit = True
                if i==3:
                    if steady_pose[0][1]<-ANGLE_THRESHOLD:
                        images_saved_per_pose+=1
                        saveit = True
                if i==4:
                    if steady_pose[0][1]>ANGLE_THRESHOLD:
                        images_saved_per_pose+=1
                        saveit = True
                # Show preview.
                if i>=5:
                    self.setPose.emit('Thank you')
                    self.checkingEnded.emit('Success')
                    # if self.outputLabel!=None:
                    #     self.outputLabel.destroy()
                    
                    # self.outputLabel = Label(self.framePic, text="Thanks")
                    # self.outputLabel.config(font=("Courier", 44))
                    # self.outputLabel.place(x=400, y=100)
                    break

                # frame = cv2.rectangle(frame, (x1,y1), (x2,y2),(255,255,0),2)
                
                # Show the image
                height, width = frame.shape[:2] 
                dim = (width, height)

                frame =frame[int(height/4):int(3/4*height),int(width/3):int(2/3*width)]
                
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                # p = convertToQtFormat.scaled(350, 250)
                self.changePixmap.emit(convertToQtFormat)
                self.setPose.emit(str(poses[i] +' : '+ str(images_saved_per_pose)+'/'+str(IMAGE_PER_POSE)))
                
                        
                if saveit:                        
                    face = dlib.rectangle(x1, y1, x2, y2)
                    face_aligned = face_aligner.align(original_frame, frame_gray, face)
                    
                    # Save the image to the server with this id
                    retval, buffer = cv2.imencode('.jpg', face_aligned)
                    jpg_as_text = base64.b64encode(buffer) 
                    fileName =  str(name)+'_'+str(number_of_images)+'.jpg'
                    
                    # files = {'image': jpg_as_text,
                    #         'filePath': str(name),
                    #         'fileName' : '/'+fileName}
                    # response = requests.post('http://127.0.0.1:8000/imageUploading/', data=files)

                    # print(response.text)

                    number_of_images+=1
                
        cap.release()
            


class RoundPixmapStyle(QProxyStyle):
    def __init__(self, radius=10, *args, **kwargs):
        super(RoundPixmapStyle, self).__init__(*args, **kwargs)
        self._radius = radius

    def drawItemPixmap(self, painter, rectangle, alignment, pixmap):
        painter.save()
        pix = QPixmap(pixmap.size())
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setBrush(QBrush(pixmap))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(pixmap.rect(), self._radius, self._radius)
        p.end()
        super(RoundPixmapStyle, self).drawItemPixmap(painter, rectangle, alignment, pix)
        painter.restore()


class Camera:
    def __init__(self):
        self.camera = None


if __name__ == '__main__':
    
    appctxt = ApplicationContext()       # 1. Instantiate ApplicationContext
    mainApp = GUI(appctxt)
    sys.exit(mainApp.exit_code)