import warnings
warnings.filterwarnings('ignore')

# Importación librerías
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

# Carga de datos de archivo .csv
dataTraining = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTraining.zip', encoding='UTF-8', index_col=0)
dataTesting = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTesting.zip', encoding='UTF-8', index_col=0)

#Se realiza lematización
nltk.download('wordnet')
wordnet_lemmatizer = WordNetLemmatizer()

def split_into_lemmas(text):
    text = text.lower()
    words = text.split()
    return [wordnet_lemmatizer.lemmatize(word,pos='v') for word in words]

vector = CountVectorizer(analyzer=split_into_lemmas,max_features=5000, stop_words="english")
X_dtm = vector.fit_transform(dataTraining['plot'])

# Definición de variable de interés (y)
dataTraining['genres'] = dataTraining['genres'].map(lambda x: eval(x))
le = MultiLabelBinarizer()
y_genres = le.fit_transform(dataTraining['genres'])

# Separación de variables predictoras (X) y variable de interés (y) en set de entrenamiento y test usandola función train_test_split
X_train, X_test, y_train_genres, y_test_genres = train_test_split(X_dtm, y_genres, test_size=0.20, random_state=42)

from xgboost import XGBClassifier

#clf = OneVsRestClassifier(XGBClassifier( learning_rate =0.09, n_estimators=300, max_depth=3, objective= 'binary:logistic'))
clf = OneVsRestClassifier(XGBClassifier( learning_rate =0.1, max_depth=2, min_child_weight=1, n_estimators=270, objective= 'binary:logistic'))
clf.fit(X_train, y_train_genres)

# loading library
import pickle

# create an iterator object with write permission - model.pkl
with open('model_pkl', 'wb') as files:
    pickle.dump(clf, files)
	
import joblib
joblib.dump(clf, 'movies.pkl')