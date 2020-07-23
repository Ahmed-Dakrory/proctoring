import pyaudio
import subprocess
import re

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import multiprocessing 
import cv2
from PIL import Image
from PIL import ImageTk
import threading
import imutils
import tensorflow as tf
import numpy as np
import requests

#CyKit Reader
import time
import socket
import os


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
import ServerRecognitionModel as serverModel


model_json='model/facenet_keras.json'
model_weights='model/facenet_keras_weight.h5'


# load the training dataset to map the targets

FACE_WIDTH = 160
FACE_HEIGHT = 160

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=FACE_WIDTH)

#Console controller
import ctypes
from ctypes import *

kernel32 = ctypes.WinDLL('kernel32')
user32 = ctypes.WinDLL('user32')

SW_HIDE = 0

hWnd = kernel32.GetConsoleWindow()

ImagesFrames = []

Sample_OF_10_SecVideo = 1


#Check VMWARE
def checkVmWare():
    batcmd='systeminfo /s %computername% | findstr /c:"Model:" /c:"Host Name" /c:"OS Name"'
    result = subprocess.check_output(batcmd, shell=True)
    # print(result)

    if re.search('VirtualBox', str(result), re.IGNORECASE):
        return (True)
    else:
        return (False)

def checkCamera():
    cap = cv2.VideoCapture(0) 
    if cap is None or not cap.isOpened():
        return False
    else:
        return True
    
def checkMicrophone():
    winmm= windll.winmm
    if winmm.waveInGetNumDevs()>0:
        return True
    else:
        return False

def checkSpeaker():
    p = pyaudio.PyAudio()

    for i in range(0,10):
        try:
            if p.get_device_info_by_index(i)['maxOutputChannels']>0:
                return True
        except Exception as e:
            print (e)
            return False


def prewhiten(x):
    if x.ndim==4:
        axis=(1, 2, 3)
        size=x[0].size
    elif x.ndim==3:
        axis=(0, 1, 2)
        size=x.size
    else:
        raise ValueError('Dimension should be 3 or 4.')

    mean=np.mean(x,axis=axis,keepdims=True)
    std=np.std(x,axis=axis,keepdims=True)
    std_adj=np.maximum(std,1.0/np.sqrt(size))
    y=(x-mean)/std_adj
    return y


def intializeModel():
    with open(model_json, 'r') as json_file:
        loaded_json = json_file.read()
    model = model_from_json(loaded_json)
    model.load_weights(model_weights)
    return model



def push(mainNumpyArray,vectorArray,len):
    size = np.shape(mainNumpyArray)
    array = mainNumpyArray
    if size[0] >= len:
        array.pop(0)
        array.append(vectorArray)
    else:
        array.append(vectorArray)
    return array




class GUI:
    def __init__(self):
        self.appWindow = Tk()
        self.appWindow.geometry("390x400")
        self.appWindow.wm_title("Proctoring")
        self.appWindow.iconbitmap('logo.ico')
        self.appWindow.wm_protocol("WM_DELETE_WINDOW", self.closingWindow)
        self.camlabel = None
        self.Speakerlabel = None
        self.Microphonelabel = None

        self.camimage = None
        self.speakimage = None
        self.microimage = None

        self.personLabel= None
        self.personId = None

        self.cameraStateMain = "True"
        self.graph = tf.compat.v1.get_default_graph()
        

        self.notExistimage = Image.open('false.png')
        self.notExistimage = self.notExistimage.resize((30, 30)) 
        self.notExistimage.save("false.ppm", "ppm")
        self.notExistimageImageTk = PhotoImage(file='false.ppm')

        self.Existimage = Image.open('correct.png')
        self.Existimage = self.Existimage.resize((30, 30)) 
        self.Existimage.save("correct.ppm", "ppm")
        self.ExistImageTk = PhotoImage(file='correct.ppm')
        # with self.graph.as_default():
        # Load the models
        self.model = intializeModel()
        

        #First Tab Data
        if self.cameraStateMain == "True":
            self.framePic = Frame(self.appWindow, relief=RAISED, borderwidth=1)
            self.framePic.pack(fill=BOTH, expand=True)

            self.panelButton = Frame(self.appWindow, relief=RAISED, borderwidth=1)
            self.panelButton.pack(fill=BOTH, expand=True)
            

            self.panel = None
            
            self.camera = CameraApp(self.panel,self.framePic,self.graph)



            self.btn = Button(self.panelButton, text = "Start Prediction", command = self.predict)
            self.btn.pack(side=RIGHT, padx=5, pady=5)

            self.btn2 = Button(self.panelButton, text = "Stop Prediction", command = self.closePredict)
            self.btn2.pack(side=RIGHT, padx=5, pady=5)

            self.btn3 = Button(self.panelButton, text = "Check Devices", command = self.checkDevices)
            self.btn3.pack(side=RIGHT, padx=5, pady=5)



        if hWnd:
            user32.ShowWindow(hWnd, SW_HIDE)

    def checkDevices(self):

        threadCheck = threading.Thread(target=self.checkDevicesThread, args=())
        threadCheck.start()
        

    def checkDevicesThread(self):
        if self.camlabel!=None:
            self.camlabel.destroy()
            self.Speakerlabel.destroy()
            self.Microphonelabel.destroy()
            self.Virtuallabel.destroy()

            self.camimage.destroy()
            self.speakimage.destroy()
            self.microimage.destroy()
            self.virtualimage.destroy()
        
        self.camlabel = Label(self.framePic, text="Camera: ")
        self.camlabel.place(x=350, y=10)
        self.Speakerlabel = Label(self.framePic, text="Speaker: ")
        self.Speakerlabel.place(x=350, y=50)
        self.Microphonelabel = Label(self.framePic, text="MicroPhone: ")
        self.Microphonelabel.place(x=350, y=90)
        self.Virtuallabel = Label(self.framePic, text="Not Virtualized: ")
        self.Virtuallabel.place(x=350, y=130)

        if checkCamera():
            self.camimage = Label(self.framePic,image = self.ExistImageTk)
            self.camimage.place(x=450, y=10)
        else:
            self.camimage = Label(self.framePic,image = self.notExistimageImageTk)
            self.camimage.place(x=450, y=10)

        if checkSpeaker():
            self.speakimage = Label(self.framePic,image = self.ExistImageTk)
            self.speakimage.place(x=450, y=50)
        else:
            self.speakimage = Label(self.framePic,image = self.notExistimageImageTk)
            self.speakimage.place(x=450, y=50)
        

        if checkMicrophone():
            self.microimage = Label(self.framePic,image = self.ExistImageTk)
            self.microimage.place(x=450, y=90)
        else:
            self.microimage = Label(self.framePic,image = self.notExistimageImageTk)
            self.microimage.place(x=450, y=90)

        
        if not checkVmWare():
            self.virtualimage = Label(self.framePic,image = self.ExistImageTk)
            self.virtualimage.place(x=450, y=130)
        else:
            self.virtualimage = Label(self.framePic,image = self.notExistimageImageTk)
            self.virtualimage.place(x=450, y=130)

    def make_label(self,master, x, y, w, h, img):
        f = Frame(master, height = h, width = w)
        f.pack_propagate(0) 
        f.place(x = x, y = y)
        label = Label(f, image = img)
        label.pack(fill = BOTH, expand = 1)
        return label


    def closePredict(self):
        print("Close Prediction")
        self.Thread_Of_Prediction_Is_Run = False
        self.camera.endCap()

    def predict(self):
        print("Start Prediction")
        self.camera.startCap()
        self.Thread_Of_Prediction_Is_Run = True

        threadPrediction = threading.Thread(target=self.makePredictionImage, args=())
        threadPrediction.start()


    def closingWindow(self):
        print("Close App")
        self.Thread_Of_Prediction_Is_Run = False
        os._exit(0)
            
    def makePredictionImage(self):
        while self.Thread_Of_Prediction_Is_Run:
            try:
                if self.camera.face_alignedFlag:
                    #Here the model Prediction
                    dataToPredict = prewhiten(self.camera.face_aligned).reshape(-1,FACE_WIDTH,FACE_HEIGHT,3)
                    embedding=self.model.predict(dataToPredict)
                    features = {'features':str(embedding.tolist())}
                    url = 'http://127.0.0.1:8000/imageRecognition'
                    out = requests.post(url,data = features)
                    print(out.json()['FacePerson'])
                    if self.personId!=None:
                        self.personId.destroy()
                        self.personLabel.destroy()

                    self.personLabel = Label(self.framePic, text='Person: ')
                    self.personLabel.config(font=("Courier", 44))
                    self.personLabel.place(x=350, y=200)
                    self.personId = Label(self.framePic, text=str(out.json()['FacePerson']))
                    self.personId.config(font=("Courier", 44))
                    self.personId.place(x=590, y=200)
            except Exception as e:
                print(e)
                



class CameraApp:
    def __init__(self,panel,framePic,graph):
       
        self.isCameraThreadEnded = True
        self.framePic = framePic
        self.panel = panel
        self.graph = graph
        self.face_alignedFlag = False
        

    def cameraCap(self):
       
        while True:
            dim = (650,650)
            if self.isCameraThreadEnded:
                break

            ret,self.imgCap=self.cap.read()
            
            if ret==True:
                
                self.imgCap = cv2.resize(self.imgCap, dim, interpolation = cv2.INTER_AREA)
                self.imgCap = cv2.flip(self.imgCap, 1)
                with self.graph.as_default():
                    #push(ImagesFrames,self.imgCap,Sample_OF_10_SecVideo)
                    
                    #print(np.array(FeaturesForVideo10SecEngagment).shape)
                    #print(np.array(FeaturesForVideo10Sec).shape)
                    # try:
                    height, width = self.imgCap.shape[:2]
                    frame_gray = cv2.cvtColor(self.imgCap, cv2.COLOR_BGR2GRAY)
                
                    faces = detector(frame_gray)
                    self.face_alignedFlag = False
                    if len(faces)> 0:
                        face = faces[0]
                        (x, y, w, h) = face_utils.rect_to_bb(face)
                        x1=max(x-0,0)
                        x2=min(x + w+0,width)
                        y1=max(y-0,0)
                        y2=min(y + h+0,height)
                        
                        face_img = frame_gray[y1:y2, x1:x2]
                        self.face_aligned = face_aligner.align(self.imgCap, frame_gray, face)
                        self.face_alignedFlag = True
                        cv2.rectangle(self.imgCap, (x1, y1), (x2,y2), (0, 255, 0), 2)
                        
                    # except Exception as e:
                    #     print("Ahmed:"+str(e))
                    
                    self.imgCap = imutils.resize(self.imgCap, width=300,height=300)
                    # OpenCV represents images in BGR order; however PIL
                    # represents images in RGB order, so we need to swap
                    # the channels, then convert to PIL and ImageTk format
                    image = cv2.cvtColor(self.imgCap, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    image = ImageTk.PhotoImage(image)
                    # if the panel is not None, we need to initialize it
                    if self.panel is None:
                        self.panel = Label(self.framePic, image=image)
                        self.panel.image = image
                        self.panel.pack(side=LEFT, padx=0, pady=0)
                        print("Done")
                    # otherwise, simply update the panel
                    else:
                        if self.imgCap.shape[0] == 300:
                            self.panel.configure(image=image)
                            self.panel.image = image
                

                
            
                

    def startCap(self):
        print("Start...")
        self.cap=cv2.VideoCapture(0)
        self.thread = threading.Thread(target=self.cameraCap, args=())
        self.isCameraThreadEnded = False
        self.thread.start()

    def endCap(self):
        print("End...")
        self.isCameraThreadEnded = True
        self.cap.release()
        

    def onClose(self):
        print("[INFO] closing...")
        self.isCameraThreadEnded = True
        self.cap.release()
        print("[Camera] closed...")

if __name__ == "__main__":
    print('Camera: '+str(checkCamera()))
    print('MicroPhone: '+str(checkMicrophone()))
    print('Speaker: '+str(checkSpeaker()))
    
    mainApp = GUI()
    mainApp.appWindow.mainloop()