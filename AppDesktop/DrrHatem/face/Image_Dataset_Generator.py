import cv2
import os,errno
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
from mark_detector import MarkDetector

CNN_INPUT_SIZE = 128
FACE_WIDTH = 160

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=FACE_WIDTH)

file='D:/DAiSEE/DataSet/Train/310066/3100661007/3100661007.avi'

#folder to store face images
facepath = 'images'
showit = False
def Image_Database_Generator():
    mark_detector = MarkDetector()
    video_capture = cv2.VideoCapture(file)
    name = input("Enter student id:")
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
    
    number_of_images = 0
    MAX_NUMBER_OF_IMAGES = 50
    count = 0
    count = 0
    while number_of_images < MAX_NUMBER_OF_IMAGES:
        ret, frame = video_capture.read()
        count+= 1
        if ret==False:
            break
        if count % 5!=0:
            continue
        
        frame = cv2.flip(frame, 1)
        height,width,_=frame.shape
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
        original_frame=frame.copy()
                        
        height, width = frame.shape[:2]        
        
        shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=FACE_WIDTH)

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
        
            # Show preview.
            if showit:
                frame = cv2.rectangle(frame, (x1,y1), (x2,y2),(255,255,0),2)
                cv2.imshow("Preview", frame)

            face = dlib.rectangle(x1, y1, x2, y2)
            face_aligned = face_aligner.align(original_frame, frame_gray, face)
            cv2.imwrite(os.path.join(directory, str(name)+'_'+str(number_of_images)+'.jpg'), face_aligned)
            
            number_of_images+=1
            print(number_of_images)

        if cv2.waitKey(10) == 27:
            break    
    print ('Thank you')
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    Image_Database_Generator()