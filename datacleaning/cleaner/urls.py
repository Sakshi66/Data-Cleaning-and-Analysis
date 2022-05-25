from django.contrib import admin
from django.urls import path,include
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('upload', views.upload, name="upload"),
    path('getColumn', views.getColumn, name="getColumn"),
    path('targetColumn', views.targetColumn, name="targetColumn"),
    path('describe', views.describe, name="describe"),
    path('nullColRem', views.nullColRem, name="nullColRem"),
    path('nullFillMean', views.nullFillMean, name="nullFillMean"),
    path('nullFillMedian', views.nullFillMedian, name="nullFillMedian"),
    path('nullFillMode', views.nullFillMode, name="nullFillMode"),
    path('encodeOneHot', views.encodeOneHot, name="encodeOneHot"),
    path('fsNorm', views.fsNorm, name="fsNorm"),
    path('fsStand', views.fsStand, name="fsStand"),
    path('csvDownload', views.csvDownload, name="csvDownload"),
    path('fmCoefficientMethod', views.fmCoefficientMethod, name="fmCoefficientMethod"),
    path('fmTreeMethod', views.fmTreeMethod, name="fmTreeMethod"),
    path('fmPCA', views.fmPCA, name="fmPCA"),
    path('getGraph', views.getGraph, name="getGraph"),
]

urlpatterns+=static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)