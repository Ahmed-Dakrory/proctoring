from django.shortcuts import render
from django.http import JsonResponse
from io import BytesIO
from PIL import Image
import re
import base64
import numpy as np
import ast

from .models import profile
from django.http import JsonResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.contrib.auth import logout
from django.shortcuts import redirect

import face_recognition
from django.views.decorators.csrf import csrf_exempt
# Create your views here.
from . import ServerRecognitionModel


import os

def mainPage(request):
    user =request.user
    userProfile = None
    if user.is_authenticated:
        userProfile = profile.objects.get(user = user)
        print("Ok")
    else:
        print("Not Found")


    context = {
        'userProfile':userProfile,
        'systemName':'Proctoring'
    }
    return render(request,'index.html',context)

def getUserByIpGandL(request):
    localIp = request.GET['localIp']
    globalIp = request.GET['ipGlobal']
    userProfile = profile.objects.filter(localIp = localIp,globalIp=globalIp).latest('user__last_login')
    
    # allTransJson = {"result":[]}

    #     allTransJson['result'].append(userProfile.to_json())
    
    return JsonResponse(userProfile.to_json())

def authUser(request):
    
    try:
        username = request.POST['username']
        password = request.POST['password']
        globalIp = request.POST['globalIp']
        localIp = request.POST['localIp']
        user = authenticate(username=username, password=password)
        if user is not None:
            thisProfile = profile.objects.filter(user=user)
            thisProfile.update(globalIp=globalIp,localIp=localIp)
            login(request, user)
            # Redirect to a success page.
        else:
            # Return an 'invalid login' error message.
            print("invalid")
    except:
        pass
    
    return redirect(request.META['HTTP_REFERER'])

def logout_req(request):
    logout(request)
    return redirect(request.META['HTTP_REFERER'])
    
en = ServerRecognitionModel.getEnModel()
grid = ServerRecognitionModel.getGrid()
@csrf_exempt
def imageRecognition(request):
    features = request.POST.get('features', [])
    arr = np.array(ast.literal_eval(features))
    # print(arr.shape)
    
    outputFace = ServerRecognitionModel.Recognition(arr,en,grid)

    data = {'Result':'Ok','FacePerson':str(outputFace)}
    return JsonResponse(data)

def register(request):
    

    if request.method=='POST':
        username = request.POST['username']
        password = request.POST['password']
        firstname = request.POST['firstname']
        lastname = request.POST['lastname']
        email = request.POST['email']
        globalIp = request.POST['globalIp']
        localIp = request.POST['localIp']
        u = None
        try:
            u = User.objects.get(username=username)
        except:
            pass
   
        if u == None:
            user = User.objects.create_user(username=username, email=email, password=password,
            first_name=firstname,last_name=lastname)
            user.save()
            profileNew = profile.objects.create(user=user,localIp=localIp,globalIp=globalIp)
            profileNew.save()

            user = authenticate(username=username,password=password)
            if user is not None:
                login(request,user)
                return redirect("/")
    else:
        pass

    return render(request,'register.html',None)

@csrf_exempt
def imageUploading(request):
    rootMain = 'media/images/'
    imgstr = request.POST['image']
    fileName = request.POST['fileName']
    filePath = request.POST['filePath']
    facepath = rootMain + filePath
    if not os.path.exists(facepath):
        os.makedirs(facepath, exist_ok = 'True')

    image_data = re.sub("^data:image/png;base64,", "", str(imgstr))
    image_data = base64.b64decode(image_data)
    image_data = BytesIO(image_data)
    img = Image.open(image_data)

    rgb_im = img.convert('RGB')
    fileDir=rootMain + filePath + fileName
    rgb_im.save(fileDir)
    data = {
        'profile': filePath,
        'file':fileName,
        'status':'success'
        }
    return JsonResponse(data)

@csrf_exempt
def imagePosting(request):
    imgstr = request.POST['image']
    image_data = re.sub("^data:image/png;base64,", "", str(imgstr))
    image_data = base64.b64decode(image_data)
    image_data = BytesIO(image_data)
    img = Image.open(image_data)

    rgb_im = img.convert('RGB')
    rgb_im.save("img2.jpg") 
    # opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # #Converting to grayscale
    # test_image_gray = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2GRAY)

    # haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    # faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5)
    
    # result = DeepFace.verify("image1.jpg", "img2.jpg")

    known_image = face_recognition.load_image_file("image1.jpg")
    unknown_image = face_recognition.load_image_file("img2.jpg")
    verfied = False
    try:
        biden_encoding = face_recognition.face_encodings(known_image)[0]
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

        results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
        
        verfied = results[0]
    except:
        pass
    x = 1
    y = 1
    w = 1
    h = 1

    # for (x,y,w,h) in faces_rects:
    #     x = x
    #     y = y
    #     w = w
    #     h = h
        
    data = {
        'x': str(x),
        'y': str(y),
        'w': str(w),
        'h': str(h),
        'verfied':str(verfied),
        }
    return JsonResponse(data)