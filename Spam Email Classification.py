
"""
@author: Kianoush 

GitHUb: https://github.com/Kianoush-h
YouTube: https://www.youtube.com/channel/UCvf9_53f6n3YjNEA4NxAkJA
LinkedIn: https://www.linkedin.com/in/kianoush-haratiannejadi/

Email: haratiank2@gmail.com

"""



import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import string
import re
from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score, ConfusionMatrixDisplay, classification_report

import tensorflow as tf
# import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout,Embedding,Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer



df = pd.read_csv('data/combined_data.csv')
head = df.head()

print(f"shape is : {df.shape}")
df.info()


# =============================================================================
# Data Cleaning and EDA
# =============================================================================


sns.countplot(x=df['label'])
plt.show()



def clean_text(text):
    punc = list(punctuation)
    stop = stopwords.words('english')
    bad_tokens = punc + stop
    lemma = WordNetLemmatizer()
    tokens = word_tokenize(text)
    word_tokens = [t for t in tokens if t.isalpha()]
    clean_tokens = [lemma.lemmatize(t.lower()) for t in word_tokens if t not in bad_tokens]
    return ' '.join(clean_tokens)


# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


# data is too large
# Taking less examples as we will be out of memory if we use all 80k examples of tha dataset
df = df.sample(8000)


df['email'] = df['text'].apply(clean_text)
head = df.head()






# =============================================================================
# Model Building
# =============================================================================

pos = df[df['label'] == 1]
neg = df[df['label'] == 0]

# Concat pos and neg label
df = pd.concat([pos,neg],axis=0)
print(f"new shape is: {df.shape}")


X = df['email']
y = df['label']


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# =============================================================================
# PART 1: Logistic Regression
# =============================================================================

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Defining a function to visualize model results
def eval(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    t1 = ConfusionMatrixDisplay(cm)
    print('Classification Report for Logistic Regression: \n')
    print(classification_report(y_test, y_pred))
    t1.plot()
eval('Classification Report', y_test, y_pred)




# =============================================================================
# PART 2: Random Forest
# =============================================================================

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Defining a function to evaluate model results
def eval(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    t1 = ConfusionMatrixDisplay(cm)
    print('Classification Report for Random Forest Classifier: \n')
    print(classification_report(y_test, y_pred))
    t1.plot()
eval('Classification Report', y_test, y_pred)





# =============================================================================
# PART 3: Naive Bayes
# =============================================================================


model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

#Defining a function to evaluate model results
def eval(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    t1 = ConfusionMatrixDisplay(cm)
    print('Classification Report for Naive Bayes Classifier: \n')
    print(classification_report(y_test, y_pred))
    t1.plot()
eval('Classification Report', y_test, y_pred)






















