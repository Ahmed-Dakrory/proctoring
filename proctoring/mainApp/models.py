from django.db import models

from django.contrib.auth.models import User

# Create your models here.
class profile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    localIp = models.CharField(max_length=15,default=None)
    globalIp = models.CharField(max_length=15,default=None)

    def to_json(self):
        return {
            'profile_id' :self.pk,
            'user_id' :self.user.pk,
            'user_name' :self.user.username,
            'user_last_login' :self.user.last_login,
            'user_first_name' :self.user.first_name,
            'user_last_name' :self.user.last_name,
            'user_email' :self.user.email
            }


    def __str__(self):
        return self.user.username