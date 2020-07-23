import os,errno
import numpy as np
import glob
# from keras.models import load_model
from numpy import savez_compressed
from PIL import Image
from keras.models import model_from_json

FACE_WIDTH = 160
FACE_HEIGHT = 160

# model_path='D:/face2/data/model/facenet_keras.h5'
model_json='model/facenet_keras.json'
model_weights='model/facenet_keras_weight.h5'

#folder to load face images from
facepath = 'images'
#folder to store embeddings
embedfolder = 'embed'

#test parameters
# facepath = 'D:/face/images2'
# filez = './embed/em2.npz'

def Create_Embedding():
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
    # save embeddings
    # savez_compressed(filez, faceSamples, ids)

if __name__ == '__main__':
    Create_Embedding()
