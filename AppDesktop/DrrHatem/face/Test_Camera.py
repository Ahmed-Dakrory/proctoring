import os
import cv2
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import column_or_1d
# from keras.models import load_model
from keras.models import model_from_json

# model_path='./model/facenet_keras.h5'
model_json='model/facenet_keras.json'
model_weights='model/facenet_keras_weight.h5'
embedfolder='embed'
# file='D:/DAiSEE/DataSet/Train/310066/3100661007/3100661007.avi'
# file='D:/DAiSEE/DataSet/Train/110002/1100022019/1100022019.avi'
# file='D:/DAiSEE/DataSet/Train/110002/1100022019/1100022022.avi'
# file='D:/DAiSEE/DataSet/Train/110002/1100022027/1100022027.avi'
#file='D:/DAiSEE/DataSet/Train/110005/1100051028/1100051028.avi'
file='1100051028.avi'
# file='1100022019.avi'
# file=0
classifierfilename = 'classifier/svmrbf.sav'
# load the training dataset to map the targets

th = 0.25
FACE_WIDTH = 160
FACE_HEIGHT = 160

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=FACE_WIDTH)

# Encode target with -1 for the unseen targets
class TolerantLabelEncoder(LabelEncoder):
    def __init__(self, ignore_unknown=False,
                       unknown_original_value='unknown', 
                       unknown_encoded_value=-1):
        self.ignore_unknown = ignore_unknown
        self.unknown_original_value = unknown_original_value
        self.unknown_encoded_value = unknown_encoded_value

    def transform(self, y):
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)

        indices = np.isin(y, self.classes_)
        if not self.ignore_unknown and not np.all(indices):
            raise ValueError("y contains new labels: %s" 
                                         % str(np.setdiff1d(y, self.classes_)))

        y_transformed = np.searchsorted(self.classes_, y)
        y_transformed[~indices]=self.unknown_encoded_value
        return y_transformed

    def inverse_transform(self, y):
        check_is_fitted(self, 'classes_')

        labels = np.arange(len(self.classes_))
        indices = np.isin(y, labels)
        if not self.ignore_unknown and not np.all(indices):
            raise ValueError("y contains new labels: %s" 
                                         % str(np.setdiff1d(y, self.classes_)))

        y_transformed = np.asarray(self.classes_[y], dtype=object)
        y_transformed[~indices]=self.unknown_original_value
        return y_transformed

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

def Test_Video():
    # model=load_model(model_path)
    checkVec = []
    indexOfCheckVec = -1
    itisHim = True
    idOfThisMan =None

    images_saved_per_pose=0
    number_of_images = 0
    IMAGE_PER_POSE=5
    indexForChangePose = 0

    poses=['frontal','right','left','up','down']
    with open(model_json, 'r') as json_file:
        loaded_json = json_file.read()
    model = model_from_json(loaded_json)
    model.load_weights(model_weights)

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
    
    grid = pickle.load(open(classifierfilename, 'rb'))
    
    # get target mapping
    en = TolerantLabelEncoder(ignore_unknown=True)
    en.fit(ids)
    
    cameraCap = cv2.VideoCapture(0)
    
    count = 0
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
        
            #faces = face_cascade.detectMultiScale(frame, 1.3, 5)
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

                if indexForChangePose == 5:
                    if itisHim:
                        print("Hi: %s" % (idOfThisMan))
                    break

                frame = cv2.putText(frame,poses[indexForChangePose] +' : '+ str(images_saved_per_pose)+'/'+str(IMAGE_PER_POSE),(10,50),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),1,cv2.LINE_AA)
                if images_saved_per_pose==IMAGE_PER_POSE:
                    indexForChangePose+=1
                    images_saved_per_pose=0

                

                cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 2)

                #Here the SVM Model Prediction
                yhat_test = grid.predict(embedding)
                yhat_prob = grid.predict_proba(embedding)
                id=np.argmax(yhat_prob)
                # print(yhat_prob[0][id])
                if yhat_prob[0][id]<th:
                    print(yhat_prob[0][id],'unknown')
                    checkVec.append(-1)
                    indexOfCheckVec += 1
                    itisHim = False
                else:
                    print(yhat_prob[0][id],en.inverse_transform(id))
                    images_saved_per_pose+=1
                    idOfThisMan = en.inverse_transform(id)
                    checkVec.append(en.inverse_transform(id))
                    indexOfCheckVec += 1

                if indexOfCheckVec!=0:
                    if checkVec[indexOfCheckVec]!=checkVec[indexOfCheckVec-1]:
                        itisHim = False
                        break
    
            cv2.imshow('Video', frame)
    
        if cv2.waitKey(10) == 27:
            break
    
    cameraCap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    Test_Video()