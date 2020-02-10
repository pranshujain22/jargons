from django.urls import path
from dashboard import views

# TEMPLATE TAGGING
app_name = 'dashboard'

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload, name='upload'),
    path('indices/', views.indices, name='indices'),
    path('predict/', views.get_prediction, name='predict')
]
