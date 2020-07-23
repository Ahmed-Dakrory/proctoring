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


# from imutils import face_utils
from imutils.face_utils import FaceAligner

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer



########################################################
####For Embeded Function
import os,errno
import glob
# from keras.models import load_model
from numpy import savez_compressed


# model_path='D:/face2/data/model/facenet_keras.h5'
model_json='model/facenet_keras.json'
model_weights='model/facenet_keras_weight.h5'

#folder to store embeddings
embedfolder = 'embed'

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


# load the training dataset to map the targets

FACE_WIDTH = 160
FACE_HEIGHT = 160

#Console controller
import ctypes
from ctypes import *

kernel32 = ctypes.WinDLL('kernel32')
user32 = ctypes.WinDLL('user32')

SW_HIDE = 0

hWnd = kernel32.GetConsoleWindow()

ImagesFrames = []

Sample_OF_10_SecVideo = 1




def Create_Embedding(outputLabel,framePic):
    if outputLabel!=None:
        outputLabel.destroy()
            
    outputLabel = Label(framePic, text="Here We Start Learning By Embedding")
    outputLabel.config(font=("Courier", 33))
    outputLabel.place(x=400, y=100)
    # model=load_model(model_path)
    with open(model_json, 'r') as json_file:
        loaded_json = json_file.read()
    model = model_from_json(loaded_json)
    model.load_weights(model_weights)
    
    
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
    
    faceSamples=[]
    # ids=[]
    nclasses=0
    
    for foldername in os.listdir(facepath):
        f=foldername.split('.')
        if len(f)>1:
            continue
        print('class = ',nclasses+1)
        nimages = len(glob.glob(os.path.join(facepath,foldername,'*.jpg')))
        em=os.path.join(embedfolder,foldername+'.npz')
        if len(glob.glob(em))==0:            
            if not os.path.exists(embedfolder):
                try:
                    os.makedirs(embedfolder, exist_ok = 'True')
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        print('Access denied')
                        return    
            for filename in os.listdir(os.path.join(facepath,foldername)):
                face_pixels = Image.open(os.path.join(facepath,foldername,filename))
                face_pixels = np.asarray(face_pixels)
                embedding=model.predict(prewhiten(face_pixels).reshape(-1,FACE_WIDTH,FACE_HEIGHT,3))
        
                faceSamples.append(embedding.reshape(-1))
                ## faceSamples.append(embedding)
                ## ids.append(int(foldername))
                # ids.append(foldername)
            savez_compressed(em, faceSamples)
            faceSamples=[]
            # ids=[]
        nclasses+=1
    
    print('Number of classes processed = ',nclasses)
    if outputLabel!=None:
        outputLabel.destroy()
    
    outputLabel = Label(framePic, text='Number of classes processes = %s' % (nclasses))
    outputLabel.config(font=("Courier", 33))
    outputLabel.place(x=400, y=100)
    # save embeddings
    # savez_compressed(filez, faceSamples, ids)



def Train_Classifier(outputLabel,framePic):
    if outputLabel!=None:
        outputLabel.destroy()
            
    outputLabel = Label(framePic, text="Start Training")
    outputLabel.config(font=("Courier", 33))
    outputLabel.place(x=400, y=100)
    trainX=[]
    ids=[]
    for filename in os.listdir(embedfolder):    
        if filename.endswith('.npz'):
            f=filename.split('.npz')
            foldername=f[0]
            data = np.load(os.path.join(embedfolder,filename))
            trainX.append(data['arr_0'])
            idc=[foldername]*data['arr_0'].shape[0]
            ids.append(idc)
    
    trainX=np.concatenate(trainX,axis=0)
    ids=np.concatenate(ids,axis=0)
    out_encoder = LabelEncoder()
    out_encoder.fit(ids)
    trainy = out_encoder.transform(ids)
    
    # fit model
    
    # scoring = ['accuracy']
    #scoring = {'accuracy': make_scorer(accuracy_score),'prec': 'precision'}
    #grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
    # param_grid = {'C': [10, 100],'gamma' : [0.1,1]}
    param_grid = {'C': [10, 100]}
    grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, cv=10,verbose = 3) 
    
    # param_grid = {'C': [10, 100]} 
    # grid = GridSearchCV(LinearSVC(), param_grid, refit=True, cv=5,verbose = 3) 
    
    
    grid.fit(trainX, trainy)
    
    pickle.dump(grid, open(classifierfilename, 'wb'))
    
    # predict
    yhat_train = grid.predict(trainX)
    # score
    score_train = accuracy_score(trainy, yhat_train)
    # summarize
    if outputLabel!=None:
        outputLabel.destroy()
            
    outputLabel = Label(framePic, text='Accuracy: train=%.3f' % (score_train*100))
    outputLabel.config(font=("Courier", 33))
    outputLabel.place(x=400, y=170)


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
        self.outputLabel = None
       
       
        self.cameraStateMain = "True"
        self.graph = tf.compat.v1.get_default_graph()
        


        

        #First Tab Data
        if self.cameraStateMain == "True":
            self.framePic = Frame(self.appWindow, relief=RAISED, borderwidth=1)
            self.framePic.pack(fill=BOTH, expand=True)

            Label(self.framePic, text='userId', bg='dodgerblue').place(x=400, y=10)
            self.userId = Entry(self.framePic,)

            self.userId.place(x=400, y=50)

            self.panelButton = Frame(self.appWindow, relief=RAISED, borderwidth=1)
            self.panelButton.pack(fill=BOTH, expand=True)
            

            self.panel = None
            self.outputLabel = None
            self.camera = CameraApp(self.panel,self.framePic,self.graph,self.outputLabel,self.userId)



            self.btn = Button(self.panelButton, text = "Start Cap", command = self.predict)
            self.btn.pack(side=RIGHT, padx=5, pady=5)

            self.btn2 = Button(self.panelButton, text = "Stop Prediction", command = self.closePredict)
            self.btn2.pack(side=RIGHT, padx=5, pady=5)

            self.btn3 = Button(self.panelButton, text = "Learn ", command = self.checkDevices)
            self.btn3.pack(side=RIGHT, padx=5, pady=5)



        if hWnd:
            user32.ShowWindow(hWnd, SW_HIDE)

    def checkDevices(self):

        threadCheck = threading.Thread(target=self.checkDevicesThread, args=())
        threadCheck.start()
        

    def checkDevicesThread(self):
        Create_Embedding(self.outputLabel,self.framePic)
        Train_Classifier(self.outputLabel,self.framePic)
        
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

        # threadPrediction = threading.Thread(target=self.makePredictionImage, args=())
        # threadPrediction.start()


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
            except Exception as e:
                print(e)
                



class CameraApp:
    def __init__(self,panel,framePic,graph,outputLabel,userId):
       
        self.isCameraThreadEnded = True
        self.framePic = framePic
        self.panel = panel
        self.graph = graph
        self.face_alignedFlag = False
        self.outputLabel = outputLabel
        self.userId = userId

    def cameraCap(self):
        self.uId = self.userId.get()
        if self.uId != '':
            if self.outputLabel!=None:
                self.outputLabel.destroy()
            
            self.outputLabel = Label(self.framePic, text="Here We Start")
            self.outputLabel.config(font=("Courier", 44))
            self.outputLabel.place(x=400, y=100)

            mark_detector = MarkDetector()
            name = self.uId
            directory = os.path.join(facepath, name)
            
            if not os.path.exists(facepath):
                os.makedirs(facepath, exist_ok = 'True')
            
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok = 'True')
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        print('invalid student id or access denied')
                        return

            poses=['frontal','right','left','up','down']
            file=0
            
            ret, sample_frame = self.cap.read()
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
            
            shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=FACE_WIDTH)
            while i<5:
                saveit = False
                # Read frame, crop it, flip it, suits your needs.
                ret, frame = self.cap.read()
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
            
                    # print(steady_pose[0][0])
                    # if steady_pose[0][0]>0.1:
                    #     print('right')
                    # else: 
                    #     if steady_pose[0][0]<-0.1:
                    #         print('left')
                    # if steady_pose[0][1]>0.1:
                    #     print('down')
                    # else: 
                    #     if steady_pose[0][1]<-0.1:
                    #         print('up')
                    # print(steady_pose[0])
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
                        print ('Thank you')
                        if self.outputLabel!=None:
                            self.outputLabel.destroy()
                        
                        self.outputLabel = Label(self.framePic, text="Thanks")
                        self.outputLabel.config(font=("Courier", 44))
                        self.outputLabel.place(x=400, y=100)
                        break

                    frame = cv2.putText(frame,poses[i] +' : '+ str(images_saved_per_pose)+'/'+str(IMAGE_PER_POSE),(10,50),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),1,cv2.LINE_AA)
                    frame = cv2.rectangle(frame, (x1,y1), (x2,y2),(255,255,0),2)                        
                    frame = imutils.resize(frame, width=300,height=300)
                    # OpenCV represents images in BGR order; however PIL
                    # represents images in RGB order, so we need to swap
                    # the channels, then convert to PIL and ImageTk format
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                        self.panel.configure(image=image)
                        self.panel.image = image
                            
                    if saveit:                        
                        face = dlib.rectangle(x1, y1, x2, y2)
                        face_aligned = face_aligner.align(original_frame, frame_gray, face)
                        cv2.imwrite(os.path.join(directory, str(name)+'_'+str(number_of_images)+'.jpg'), face_aligned)
                        print(images_saved_per_pose)
                        number_of_images+=1
                    
            self.cap.release()
        else:
            if self.outputLabel!=None:
                self.outputLabel.destroy()
            
            self.outputLabel = Label(self.framePic, text="Please Enter a Valid Id")
            self.outputLabel.config(font=("Courier", 44))
            self.outputLabel.place(x=400, y=100)

                
            

                
            
                

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
    
    mainApp = GUI()
    mainApp.appWindow.mainloop()