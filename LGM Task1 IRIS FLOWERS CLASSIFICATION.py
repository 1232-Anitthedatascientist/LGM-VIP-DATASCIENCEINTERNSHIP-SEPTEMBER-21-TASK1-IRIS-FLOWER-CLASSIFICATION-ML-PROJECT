#!/usr/bin/env python
# coding: utf-8

# # LGM DATA SCIENCE INTERNSHIP
#    Task 1: Iris Flowers Classification
#    

# In[68]:


import pandas as pd
import numpy as np
import os
import seaborn as sn
import sweetviz as sv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# In[7]:


import warnings
warnings.filterwarnings('ignore') 


# # IMPORTING  AND CHECKING THE DATASET 
# 

# In[8]:


df_Iris=pd.read_csv('iris .data',names="Sepal_Length Sepal_Width Petal_Length Petal_Width Species".split())
df_Iris


# In[9]:


df_Iris.shape


# In[18]:


df_Iris.tail()


# In[11]:


df_Iris.info()


# In[14]:


df_Iris['Species'].value_counts()  #Count the number of unique roows of "Species"


# In[15]:


df_Iris.isnull().sum()


# In[20]:


df_Iris.describe()  


# In[19]:


df_Iris.corr()


# # EDA OF DATASET

# In[17]:


sn.countplot(df_Iris['Species'])


# This shows us that we are having a balanced Dataset

# In[21]:


sn.pairplot(df_Iris,hue='Species')
plt.show()


# Here we can observe that "Iris-setosa" can be easily distinguished from the other two species as they are overlapping 

# In[23]:


df_Iris.plot(kind='box',figsize=(14,8))


# It can be observed that "Sepal Width" is having outliers

# In[29]:


cor=df_Iris.corr()
sn.heatmap(cor,cmap='Oranges',annot=True,linewidths=5)
plt.show()


# The heatmap shows that there is high correlation 

# # MACHINE LEARNING MODELS AND THEIR ACCURACY CHECK

# In[36]:


X=df_Iris[['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']]
Y=df_Iris['Species']


# In[57]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=10)


# 1.SUPPPORT VECTOR MACHINE

# In[58]:


from sklearn import svm
from sklearn.metrics import accuracy_score
model_Iris=svm.SVC()
model_Iris.fit(X_train,Y_train)


# In[59]:


Y_Prediction=model_Iris.predict(X_test) #Perform classification on X_test
print("Accuracy of Support vector machine is :", accuracy_score(Y_Prediction,Y_test))


# 2.Decision Tree 

# In[60]:


from sklearn.tree import DecisionTreeClassifier
model_Iris=DecisionTreeClassifier(criterion='entropy',max_depth=7,random_state=18)
model_Iris.fit(X_train,Y_train)


# In[61]:


Y_Prediction=model_Iris.predict(X_test) #Perform classification on X_test
print("Accuracy of Decision Tree Classifier under the criterion of entropy is :", accuracy_score(Y_Prediction,Y_test))


# 3.Random Forest 

# In[62]:


from sklearn.ensemble import RandomForestClassifier
model_Iris=RandomForestClassifier()
model_Iris.fit(X_train,Y_train)


# In[63]:


Y_Prediction=model_Iris.predict(X_test) #Perform classification on X_test
print("Accuracy of Random Forest Classifier is :", accuracy_score(Y_Prediction,Y_test))


# 4.K Nearest Neighbor(KNN)

# In[64]:


from sklearn.neighbors import KNeighborsClassifier
model_Iris=KNeighborsClassifier(n_neighbors=6,metric='euclidean')
model_Iris.fit(X_train,Y_train)


# In[65]:


Y_Prediction=model_Iris.predict(X_test) 
print("Accuracy of KNN is :", accuracy_score(Y_Prediction,Y_test))


# 5.Naive Bayes Classifier

# In[66]:


from sklearn.naive_bayes import GaussianNB
model_Iris=GaussianNB()
model_Iris.fit(X_train,Y_train)


# In[67]:


Y_Prediction=model_Iris.predict(X_test) 
print("Accuracy of Naive Bayes Classifier is :", accuracy_score(Y_Prediction,Y_test))


# # Conclusion

# From all of the above "Machine Learning Models",the accuracy rate of "Naive Bayes"   is highest.

# In[ ]:




