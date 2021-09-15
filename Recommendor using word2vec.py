# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:35:21 2021

@author: vivek.muraleedharan
"""

import pandas as pd
from gensim.models import word2vec
import re
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords

data=pd.read_excel("movie data_new.xlsx")
data.info()

data.rename(columns={'Unnamed: 0': 'movie_id'}, inplace=True)
#data.drop(columns = [])

#removing blank spaces and line splitter from string
clean_txt = []
for w in range(len(data.Genre)):
    desc = data['Genre'][w] 
    desc = desc.replace(" ","")
    clean_txt.append(re.sub('\n', '', desc))
data['Genre'] = clean_txt

#text preprocessing in Description
clean_sentence = []
for t in range(len(data.Description)):
    text = data['Description'][t]
    clean_sentence.append(remove_stopwords(text))
data["Description"] = clean_sentence
    
    
def get_important_features(data):
    important_features=[]
    for i in range (0,data.shape[0]):     
        important_features.append(data['Movie Name'][i]+
                                  ' ,'+data['Director'][i]+','+data['Genre'][i]+','
                                  +data['Description'][i])
    return important_features


data['important_features']=get_important_features(data)
#data['important_features']= data['important_features'].tolist()
#data.info()


#making list of list to vectorize
def extractDigits(lst):
    res = []
    for el in lst:
        sub = el.split(',')
        res.append(sub)
      
    return(res)
                  
data['important_features']= extractDigits(data['important_features'])


#vectorizing the features
model = word2vec.Word2Vec(data['important_features'], 
                          workers = 1, vector_size = 10,
                          min_count = 1, window = 5, sg = 0)

#model.wv['Jodie Foster']


#saving the model parameters 
word_vectors = model.wv
word_vectors.save("word2vec.wordvectors")
wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')


#function to return the similar movie names 
def get_names(list):
    names =[]
    for i in range(0,len(wv.most_similar(list,topn =100))):
        name = wv.most_similar(list,topn =100)[i][0]
        for i in data['Movie Name']:
            if name == i: 
                names.append(name)      
    return (names[:5])

get_names(['Interstellar'])
