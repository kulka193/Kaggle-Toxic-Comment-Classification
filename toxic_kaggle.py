# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 09:40:27 2018

@author: bharg
"""

import numpy as np
import os
#from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense,LSTM, Embedding,Dropout,Flatten
from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot
import pandas as pd
#from zipfile import Zipfile
import urllib.request
import zipfile
from nltk import word_tokenize
from nltk.corpus import stopwords
#import itertools

global STOPWORDS
global topwords             #nunmber of most common occuring words to be selected
topwords=50000  
global num_examples,maxlen          #training examples and max length of each sentence
batch_size=128
epochs=3
maxlen=50       #max lenth of each word sequence or each comment



def SearchAndAppendPath(project_name):
    project_path=''
    for dirpath,dirnames,files in os.walk('C:\\'):
        for folder in dirnames:
            if folder==project_name:
                project_path=os.path.join(dirpath,folder)
                break
            break
    if project_path=='':
        project_path=os.path.join('C:\\Kaggle\\'+project_name)
    return project_path

def getfile_maybeDownload(train_or_test):
    os.chdir(project_path)
    if os.path.exists(project_path+'\\'+train_or_test+'.csv.zip'):
        print("File exists")
        zipfilepath=os.path.join(project_path,train_or_test+'.csv.zip')
    else:
        print('File does not exist. \n Downloading from web...')
        url='https://www.kaggle.com/c/8076/download/train.csv.zip'
        urllib.request.urlretrieve(url,project_path+'\\'+train_or_test+'.csv.zip')
        zipfilepath=project_path+'\\'+train_or_test+'.csv.zip'
    return zipfilepath
 
      
def extract_zip(f):
    with zipfile.ZipFile(f,"r") as zip_ref:
        print("===Extracting data from zip file====")
        zip_ref.extractall(project_path)


def read_file(train_or_test):
    data=pd.read_csv(project_path+train_or_test)
    X=data["comment_text"].iloc[:].values
    y=data.iloc[:,2:8]
    return X,y


def read_and_preprocess(train_or_test):
    data=pd.read_csv(project_path+train_or_test)
    print('=====reading and preprocessing======')
    if train_or_test=='\\train.csv':
        data['comment_text']=data['comment_text'].str.replace('[^a-zA-Z]',' ')
        data['comment_text']=data['comment_text'].str.lower()
        X=data['comment_text'].values.tolist()
        y=data.iloc[:,2:8]
        return X,y
    elif train_or_test=='\\test.csv':
        data['comment_text']=data['comment_text'].str.replace('[^a-zA-Z]',' ')
        data['comment_text']=data['comment_text'].str.lower()
        X=data['comment_text'].values.tolist()
        return X
    else:
        return 0
            
    

    
def remove_stops(X):
    print("======removing stopwords from data============")
    processed=[]
    for i in range(num_examples):
        stops_removed=[]
        words=word_tokenize(X[i])
        for w in words:
            if not w in STOPWORDS:
               stops_removed.append(w)
        processed.append(stops_removed)
    squeezed=[" ".join(processed[i]) for i in range(num_examples)]
    return squeezed



def getvocab(X):
    print('========collected most common words in vocabulary==========')
    BOW=[]
    for i in range(len(X)):
        for word in word_tokenize(X[i]):
            BOW.append(word)
    BOW=set(BOW)
    return BOW


def encode_words(vocab_size,X):
    encoded=[]
    for eachsent in X:
        encoded.append(one_hot(eachsent,vocab_size))
    return encoded


global project_path    
project_path=SearchAndAppendPath('Toxic Comment Challenge')
zipfilepath=getfile_maybeDownload('train')
extract_zip(zipfilepath)
print("=====file extracted=====")
#Xtrain,ytrain=read_file('\\train.csv')

Xtrain,ytrain=read_and_preprocess('\\train.csv')
num_examples=len(Xtrain)
num_classes=ytrain.shape[1]

STOPWORDS=set(stopwords.words("english"))
Xtrain1=remove_stops(Xtrain)
vocabulary=getvocab(Xtrain1)
vocabulary_size=len(vocabulary)    #
processed=encode_words(topwords,Xtrain)
Xtrain_ready=sequence.pad_sequences(processed,maxlen=maxlen,padding='post')
print("=====preprocessing done, Ready to go======")
print('========building the model===========')

model=Sequential()
model.add(Embedding(vocabulary_size, 10, input_length=maxlen))
model.add(LSTM(512,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes,activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(Xtrain_ready,ytrain,
          batch_size=batch_size,
          epochs=epochs)