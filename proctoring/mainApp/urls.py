from django.urls import path
from . import views 

app_name = 'mainApp'

urlpatterns = [
    path('',views.mainPage,name = 'mainPage'),
    path('register',views.register,name = 'register'),
    path('auth', views.authUser, name='authUser'),
    path('logout', views.logout_req, name='logout'),
    path('getUserByIpGandL', views.getUserByIpGandL, name='getUserByIpGandL'),
    path('imgPosting',views.imagePosting,name = 'imgPosting'),
    path('imageUploading/',views.imageUploading,name = 'imageUploading'),
    path('imageRecognition',views.imageRecognition,name = 'imageRecognition'),
]
