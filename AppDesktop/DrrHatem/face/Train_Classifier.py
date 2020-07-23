'''Face Recognition Main File'''
import glob
import os
import numpy as np
import pickle

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC

# file to save model parameters in
classifierfilename = 'classifier/svmrbf.sav'

# load training embeddings
embedfolder = 'embed'

def Train_Classifier():
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
    print('Accuracy: train=%.3f' % (score_train*100))

if __name__ == '__main__':
    Train_Classifier()
