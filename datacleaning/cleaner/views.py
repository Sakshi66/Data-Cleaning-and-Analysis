from django.shortcuts import render
from django.http import HttpResponse
from django.http import FileResponse
from wsgiref.util import FileWrapper
from django.template.response import TemplateResponse
import json
from django import forms
from django.core.files.storage import FileSystemStorage

import sys
import os
import pandas as pd
import numpy as nu
import seaborn as sns
import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import matplotlib as pl
pl.use('Agg')
import matplotlib.pyplot as plt
import uuid, base64
import io
from io import BytesIO
import plotly.express as px

import cleaner.libs.DataDescription as dd
import cleaner.libs.NullValues as nv
import cleaner.libs.Encoding as en
import cleaner.libs.FeatureScaling as fs
#import cleaner.libs.Downloading as dw

# Create your views here.
def home(request):
    #print("Hello")
    global df, setColumn, i, table_obj
    if i==0:
        return render(request,'default.html')
    elif i==1:
        table_obj = df.to_html()
        print(type(df))
        print(type(table_obj))
        return render(request,'clean.html',{'table':table_obj})
        #{'setColumn':setColumn}



def upload(request):
    global filename, i, table_obj,df
    if request.method == "POST":
        csv_file = request.FILE['file']
        fs = FileSystemStorage()
        name = fs.save(csv_file.name, csv_file)
        filename = {fs.url(name)}
        print(f"file_name = {fs.url(name)}")
        df = pd.read_csv(filename, engine='python', encoding = 'unicode_escape')
        table_obj = df.to_html()
        i=1
    #print(type(df))
    #print(type(table_obj))
    table_obj = df.to_html()
    return render(request,'clean.html',{'table':table_obj})
    #return HttpResponse(filename)


class InputForm(forms.Form):
    colName = forms.CharField(max_length = 200, required = False)

def targetColumn(request):
    global df, target
    if request.method == 'POST':
        form = InputForm(request.POST)
        temp = str(request.POST['target'])
        if temp in df.columns:
            target = temp
        else:
            target = df.columns[-1]
        return HttpResponse(target)

def getColumn(request):
    global df, setColumn
    if request.method == 'POST':
        form = InputForm(request.POST)
        setColumn = str(request.POST['colName'])
        return HttpResponse(setColumn)
        # if form.is_valid():
        #     setColumn = request.POST['colName']
        #    #setColumn = form.cleaned_data.get("colName")
        #    print(setColumn)
        #    return HttpResponse(setColumn)

def nu_encoder(object):
    if isinstance(object, nu.generic):
        return object.item()           

def describe(request):
    global df, setColumn
    setColumn = str(request.POST['colName'])
    tdisc = DescObj.colProperty(df,setColumn)
    print(tdisc)
    #return HttpResponse(json.dumps(tdisc, default=nu_encoder))
    json_stats = json.dumps(tdisc, default=nu_encoder)
    return HttpResponse(json_stats,content_type='application/json')

# Null Values
def nullColRem(request):
    global df, setColumn
    if request.method == 'GET':
        df = NullObj.rmNullCol(df,setColumn)
        #return render(request, "clean.html")
        table_obj = df.to_html()
        #return render(request,'clean.html',{'table':table_obj})
        return HttpResponse(table_obj)

def nullFillMean(request):
    global df, setColumn
    if request.method == 'GET':
        df = NullObj.fillMean(df,setColumn)
        table_obj = df.to_html()
        #return render(request,'clean.html',{'table':table_obj})
        return HttpResponse(table_obj)

def nullFillMedian(request):
    global df, setColumn
    if request.method == 'GET':
        df = NullObj.fillMedian(df,setColumn)
        table_obj = df.to_html()
        #return render(request,'clean.html',{'table':table_obj})
        return HttpResponse(table_obj)

def nullFillMode(request):
    global df, setColumn
    if request.method == 'GET':
        df = NullObj.fillMode(df,setColumn)
        table_obj = df.to_html()
        #return render(request,'clean.html',{'table':table_obj})
        return HttpResponse(table_obj)

# Encoding
def encodeOneHot(request):
    global df, setColumn
    if request.method == 'GET':
        df = EncObj.oneHotEncoding(df,setColumn)
        table_obj = df.to_html()
        #return render(request,'clean.html',{'table':table_obj})
        return HttpResponse(table_obj)    

# normalization
def fsNorm(request):
    global df, setColumn
    if request.method == 'GET':
        df = ScObj.fNormalize(df,setColumn)
        table_obj = df.to_html()
        #return render(request,'clean.html',{'table':table_obj})
        return HttpResponse(table_obj)

# standardization
def fsStand(request):
    global df, setColumn
    if request.method == 'GET':
        df = ScObj.fStandardize(df,setColumn)
        table_obj = df.to_html()
        #return render(request,'clean.html',{'table':table_obj})
        return HttpResponse(table_obj)

def csvDownload(request):
    global df
    #export DataFrame to CSV file
    results = df
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename=filename.csv'
    results.to_csv(path_or_buf=response,sep=',',float_format='%.2f',index=False,decimal=",")
    return response

def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

def getGraph(request):
    global importances

    plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
    plt.title('Feature importances obtained from coefficients', size=20)
    plt.xticks(rotation='vertical')
    #plt.show()

    response = HttpResponse(content_type="image/jpeg")
    plt.savefig(response, format="png")
    return response


def featureData():
    global df, target
    global X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_train_scaled, X_test_scaled

    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.transform(X_test)

def fmCoefficientMethod(request):
    global df, target, importances
    global X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_train_scaled, X_test_scaled

    featureData()

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    importances = pd.DataFrame(data={
        'Attribute': X_train.columns,
        'Importance': model.coef_[0]
    })
    importances = importances.sort_values(by='Importance', ascending=False)

    #plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
    #plt.title('Feature importances obtained from coefficients', size=20)
    #plt.xticks(rotation='vertical')
    #plt.show()

    #plt.switch_backend('AGG')

    imp = importances.to_html()
    #chart = graph.to_html()
    return render(request,'featureImp.html',{'table':imp})


def fmTreeMethod(request):
    global df, target
    global X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_train_scaled, X_test_scaled

    featureData()

    model = XGBClassifier()
    model.fit(X_train_scaled, y_train)
    importances = pd.DataFrame(data={
        'Attribute': X_train.columns,
        'Importance': model.feature_importances_
    })
    importances = importances.sort_values(by='Importance', ascending=False) 

    plt.switch_backend('AGG')
    plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
    plt.title('Feature importances obtained from coefficients', size=20)
    plt.xticks(rotation='vertical')
    #plt.show()
    plt.tight_layout()
    graph = get_graph() 
    imp = importances.to_html()
    return render(request,'featureImp.html',{'table':imp})

def fmPCA(request):
    global df, target
    global X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_train_scaled, X_test_scaled

    pca = PCA().fit(X_train_scaled)

    plt.plot(pca.explained_variance_ratio_.cumsum(), lw=3, color='#087E8B')
    plt.title('Cumulative explained variance by number of principal components', size=20)
    #plt.show()

    loadings = pd.DataFrame(
        data=pca.components_.T * nu.sqrt(pca.explained_variance_), 
        columns=[f'PC{i}' for i in range(1, len(X_train.columns) + 1)],
        index=X_train.columns
    )
    pc1_loadings = loadings.sort_values(by='PC1', ascending=False)[['PC1']]
    pc1_loadings = pc1_loadings.reset_index()
    pc1_loadings.columns = ['Attribute', 'CorrelationWithPC1']
    print(pc1_loadings)
    print(type(pc1_loadings))

    plt.bar(x=pc1_loadings['Attribute'], height=pc1_loadings['CorrelationWithPC1'], color='#087E8B')
    plt.title('PCA loading scores (first principal component)', size=20)
    plt.xticks(rotation='vertical')
    #plt.show()
    
    imp = pc1_loadings.to_html()
    return render(request,'featureImp.html',{'table':imp})

i = 0
filename = "./cleaner/aa/student-marks.csv"
#filename = "./cleaner/aa/IOT-temp.csv"
df = pd.read_csv(filename, sep='\t+', engine='python', encoding = 'unicode_escape')
df = pd.read_csv(filename, engine='python', encoding = 'unicode_escape')
table_obj = df.to_html()
#df = df[:10]
setColumn = None
target = None
importances = None

X = None
y = None
X_train = None
X_test = None
y_train = None
y_test = None

X_train_scaled = None
X_test_scaled = None

DescObj = dd.DataDescription()  # Data Description Object
NullObj = nv.NullValues()   # NULL Values Handling Object
EncObj = en.Encoding()    # Data Encoding Object
ScObj = fs.FeatureScaling()      # Feature Scaling Object

    