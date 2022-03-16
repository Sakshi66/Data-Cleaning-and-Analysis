from django.shortcuts import render
from django.http import HttpResponse
from wsgiref.util import FileWrapper
from django.template.response import TemplateResponse
import json

import sys
import os
import pandas as pd
import numpy as nu
import seaborn as sns

import cleaner.libs.DataDescription as dd
import cleaner.libs.NullValues as nv
import cleaner.libs.Encoding as en
import cleaner.libs.FeatureScaling as fs
#import cleaner.libs.Downloading as dw

from . import forms


# Create your views here.
def home(request):
    i = 1
    if i==0:
        return render(request,'default.html')
    elif i==1:
        table_obj = df.to_html()
        return render(request,'clean.html',{'table':table_obj})


def getFile(request):
    pass

def getColumn(request):
    context = {}
    form = InputForm(request.POST or None)
    context['form'] = form
    if request.POST:
        if form.is_valid():
            setColumn = form.cleaned_data.get("colName")
            print(setColumn)
    return render(request, "clean.html", context)

def describe():
    pass

def nullColRem(request):
    df = NullObj.rmNullCol(df,setColumn)
    return render(request, "clean.html")

def nullFillMean():
    df = NullObj.fillMean(df,setColumn)

def nullFillMedian():
    df = NullObj.fillMedian(df,setColumn)

def nullFillMode():
    df = NullObj.fillMode(df,setColumn)

def encodeOneHot():
    df = EncObj.oneHotEncoding(df,setColumn)    

# normalization
def fsNorm():
    df = ScObj.fNormalize(df,setColumn)

# standardization
def fsStand():
    df = ScObj.fStandardize(df,setColumn)

df = pd.read_csv("./cleaner/aa/sample1.csv")
df = df[:10]
setColumn = None

DescObj = dd.DataDescription()  # Data Description Object
NullObj = nv.NullValues()   # NULL Values Handling Object
EncObj = en.Encoding()    # Data Encoding Object
ScObj = fs.FeatureScaling()      # Feature Scaling Object
    
