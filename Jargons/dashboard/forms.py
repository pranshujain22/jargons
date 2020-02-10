from django import forms
from dashboard.models import UserProfile


class UserForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ('employee_id', 'password')

