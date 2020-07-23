import os
import numpy as np
import pickle

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import column_or_1d

# Linear svm
# classifierfilename = './classifier/svmlin.sav'
# rbf kernel svm
classifierfilename = 'classifier/svmrbf.sav'
# load the training dataset to map the targets
embedfolder = 'embed'
embedfolder2 = 'embed2'
th = 0.25

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

def Test_Classifier():
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
    
    # testX=trainX
    # ids2=ids
    
    testX=[]
    ids2=[]
    for filename in os.listdir(embedfolder2):
        if filename.endswith('.npz'):
            f=filename.split('.npz')
            foldername=f[0]
            data = np.load(os.path.join(embedfolder2,filename))
            testX.append(data['arr_0'])
            idc=[foldername]*data['arr_0'].shape[0]
            ids2.append(idc)
    
    testX=np.concatenate(testX,axis=0)
    ids2=np.concatenate(ids2,axis=0)
    
    # get target mapping
    en = TolerantLabelEncoder(ignore_unknown=True)
    en.fit(ids)
    # Apply target mapping to test targets
    testy = en.transform(ids2)
    
    # load svm model
    grid = pickle.load(open(classifierfilename, 'rb'))
    
    # predict
    yhat_test = grid.predict(testX)
    
    # print(yhat_test)
    yhat_prob = grid.predict_proba(testX)
    print(np.max(yhat_prob,1))
    yhat_test[np.where(np.max(yhat_prob,1)<th)]=-2
    import matplotlib
    matplotlib.pyplot.plot(yhat_test)
    # score
    score_test = accuracy_score(testy, yhat_test)
    # summarize
    print('Accuracy: test=%.3f' % (score_test*100))

if __name__ == '__main__':
    Test_Classifier()
