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




classifierfilename = 'classifier/svmrbf.sav'
# load the training dataset to map the targets

th = 0.25

embedfolder='embed'


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


def getEnModel():
    ids=[]
    for filename in os.listdir(embedfolder):    
        if filename.endswith('.npz'):
            f=filename.split('.npz')
            foldername=f[0]
            data = np.load(os.path.join(embedfolder,filename))
            idc=[foldername]*data['arr_0'].shape[0]
            ids.append(idc)
    
    ids=np.concatenate(ids,axis=0)
    
    
    
    # get target mapping
    en = TolerantLabelEncoder(ignore_unknown=True)
    en.fit(ids)
    return en

def getGrid():
    grid = pickle.load(open(classifierfilename, 'rb'))
    return grid

def Recognition(embedding,en,grid):
     # Here the SVM Model Prediction
    yhat_prob = grid.predict_proba(embedding)
    id=np.argmax(yhat_prob)
    
    if yhat_prob[0][id]<th:
        return None
    else:
        return en.inverse_transform(id)
    # if yhat_prob[0][id]<th:
    #     print(yhat_prob[0][id],'unknown')
    # else:
    #     print(yhat_prob[0][id],en.inverse_transform(id))
    
