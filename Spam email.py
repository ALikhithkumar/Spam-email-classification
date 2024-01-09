#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import re
data = pd.read_csv(r'C:\Users\LIKHITH\Downloads\SPAM text message 20170820 - Data.csv',index_col=None)
data
# Check unique values in 'Category' column
print(data['Category'].unique())

# Use countplot for visualization
sns.countplot(x='Category', data=data)
plt.show()
data['Category'] = data['Category'].map({
    'ham' : 0,
    'spam' : 1
    })
def clean_text(Message):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    Message = text.lower()
    Message = re.sub('\[.*?\]', '', Message)
    Message = re.sub('https?://\S+|www\.\S+', '', Message)
    Message = re.sub('<.*?>+', '', text)
    Message = re.sub('[%s]' % re.escape(string.punctuation), '', Message)
    Message = re.sub('\n', '', Message)
    Message = re.sub('\w*\d\w*', '', Message)
    return text
data
x = data["Message"]
y = data["Category"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
def prediction(X_test, model_object):
  
    # Predicton on test with giniIndex
    y_pred = model_object.predict(xv_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred
def cal_accuracy(y_test, y_pred):
      
    print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred))
      
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
model_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 123,max_depth=10, min_samples_leaf=6)
  
# Performing training
model_gini.fit(xv_train, y_train)

# Prediction using gini
y_pred_gini = prediction(xv_test, model_gini)
cal_accuracy(y_test, y_pred_gini)

# Decision tree with entropy
model_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 123,
            max_depth = 10, min_samples_leaf = 6)
  
# Performing training
model_entropy.fit(xv_train, y_train)

# Prediction using entropy
y_pred_entropy = prediction(xv_test, model_entropy)
cal_accuracy(y_test, y_pred_entropy)


# In[ ]:




