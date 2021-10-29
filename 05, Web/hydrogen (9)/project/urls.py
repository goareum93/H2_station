# project > urls.py
from django.urls import path
from . import views
urlpatterns=[
    path('chart.html',views.chart),
    path('empty.html',views.empty),
    path('home.html',views.form),
    path('',views.form),
    path('index.html',views.index),
    path('tab-panel.html',views.tab_panel),
    path('table.html',views.news),
    path('ui-elements.html',views.ui_elements),
]