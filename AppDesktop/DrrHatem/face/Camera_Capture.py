import os,errno
import cv2
import numpy as np
import dlib
# from imutils import face_utils
from imutils.face_utils import FaceAligner

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

#folder to store face images
facepath = 'images'
CNN_INPUT_SIZE = 128
ANGLE_THRESHOLD = 0.15
IMAGE_PER_POSE=5
FACE_WIDTH = 160

def Camera_Capture():
    mark_detector = MarkDetector()
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

    poses=['frontal','right','left','up','down']
    file=0
    cap = cv2.VideoCapture(file)
    
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
    
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=FACE_WIDTH)
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
                break

            frame = cv2.putText(frame,poses[i] +' : '+ str(images_saved_per_pose)+'/'+str(IMAGE_PER_POSE),(10,50),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),1,cv2.LINE_AA)
            frame = cv2.rectangle(frame, (x1,y1), (x2,y2),(255,255,0),2)
            cv2.imshow("Preview", frame)

            if saveit:                        
                face = dlib.rectangle(x1, y1, x2, y2)
                face_aligned = face_aligner.align(original_frame, frame_gray, face)
                cv2.imwrite(os.path.join(directory, str(name)+'_'+str(number_of_images)+'.jpg'), face_aligned)
                print(images_saved_per_pose)
                number_of_images+=1

        if cv2.waitKey(10) == 27:
            break    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    Camera_Capture()
