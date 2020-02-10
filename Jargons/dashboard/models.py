from django.db import models


class UserProfile(models.Model):

    employee_id = models.IntegerField(unique=True, primary_key=True)
    password = models.CharField(max_length=256)

    def __str__(self):
        return str(self.employee_id)

