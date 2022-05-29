#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer

def split_into_lemmas(text):
    text = text.lower()
    words = text.split()
    return [wordnet_lemmatizer.lemmatize(word,pos='v') for word in words]
    
def predict_proba(plot):
    
    clf = joblib.load(os.path.dirname(__file__) + '/movies.pkl') 
    vector = CountVectorizer(analyzer=split_into_lemmas,max_features=5000, stop_words="english")
    dataTraining = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTraining.zip', encoding='UTF-8', index_col=0)
    X_dtm = vector.fit_transform(dataTraining['plot'])
    pl=clf.predict_proba(vector.transform([plot]))

    return p1


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add an URL')
        
    else:

        plot = sys.argv[1]

        p1 = predict_proba(plot)
        
        print(plot)
        print('Probability of Phishing: ', p1)