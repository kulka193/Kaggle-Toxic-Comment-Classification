# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 21:42:14 2018

@author: bharg
"""

import numpy as np
#from gensim.models import Word2Vec
from keras.models import Sequential,model_from_json
from keras.layers import Dense,LSTM, Embedding,Dropout,Flatten
from keras.preprocessing import sequence

#from zipfile import Zipfile

from nltk.corpus import stopwords
from toxic_kaggle import SearchAndAppendPath,getfile_maybeDownload,extract_zip,read_and_preprocess,remove_stops,getvocab,encode_words

maxlen=50
topwords=50000
global project_path    
project_path=SearchAndAppendPath('Toxic Comment Challenge')
zipfilepath=getfile_maybeDownload('test')
extract_zip(zipfilepath)
print("=====file extracted=====")
global num_examples

Xtest=read_and_preprocess('\\test.csv')
num_examples=len(Xtest)

STOPWORDS=set(stopwords.words("english"))
Xtest1=remove_stops(Xtest)
vocabulary=getvocab(Xtest1)
vocabulary_size=len(vocabulary)

processed=encode_words(topwords,Xtest1)
Xtest_ready=sequence.pad_sequences(processed,maxlen=maxlen,padding='post')
print("=====preprocessing done, Ready to go======")
print('========building the model===========')

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

#loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
score = loaded_model.predict(Xtest_ready)
