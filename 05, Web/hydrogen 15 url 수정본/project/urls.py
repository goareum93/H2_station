# project > urls.py
from django.urls import path
from . import views

app_name = 'main'

urlpatterns=[
    path('hydrogencar',views.chart,name='car'),
    path('profile',views.empty,name='profile'),
    path('home',views.form,name='home'),
    path('recommend',views.index,name='recommend'),
    path('location',views.tab_panel,name='location'),
    path('news',views.news,name='news'),
    path('seoul',views.ui_elements,name='seoul'),
]