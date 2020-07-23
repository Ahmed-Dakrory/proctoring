import os
import cv2
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
import pickle
import numpy as np
# from keras.models import load_model
from keras.models import model_from_json
import ServerRecognitionModel as serverModel


model_json='model/facenet_keras.json'
model_weights='model/facenet_keras_weight.h5'


# load the training dataset to map the targets

FACE_WIDTH = 160
FACE_HEIGHT = 160

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=FACE_WIDTH)


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


def Test_Video():


    model = intializeModel()

    en = serverModel.getEnModel()
    grid = serverModel.getGrid()

    
    cameraCap = cv2.VideoCapture(0)
    
    count = 0
    while True:
        ret, frame = cameraCap.read()
        count+= 1
        if ret==False:
            break
        if count % 5==0:
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            faces = detector(frame_gray)
            if len(faces)> 0:
                face = faces[0]
                (x, y, w, h) = face_utils.rect_to_bb(face)
                x1=max(x-0,0)
                x2=min(x + w+0,width)
                y1=max(y-0,0)
                y2=min(y + h+0,height)
                
                face_img = frame_gray[y1:y2, x1:x2]
                face_aligned = face_aligner.align(frame, frame_gray, face)

                #Here the model Prediction
                embedding=model.predict(prewhiten(face_aligned).reshape(-1,FACE_WIDTH,FACE_HEIGHT,3))

                #print(embedding.shape)
                print(serverModel.Recognition(embedding,en,grid))
                cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 2)

               
            cv2.imshow('Video', frame)
    
        if cv2.waitKey(10) == 27:
            break
    
    cameraCap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    Test_Video()