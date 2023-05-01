# 최상위 폴더 -> urls.py -> urlpatterns -> 추가
from django.urls import path
from . import views
# path default : path(route, view, kwargs=None, name=None), 
# https://localhost:8000/onememos/ -> views.index -> name='index'
urlpatterns = [
    path('onememos/', views.index, name='index'),
]
