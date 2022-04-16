#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np


# In[2]:


df=pd.read_csv(r'https://raw.githubusercontent.com/tranghth-lux/data-science-complete-tutorial/master/Data/HR_comma_sep.csv.txt')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.notnull().count()


# In[6]:


df.nunique()


# In[7]:


df.dtypes


# In[8]:


df.shape


# In[9]:


df.notnull()


# In[10]:


df.notna()


# In[11]:


df.count()


# In[12]:


#Outlier detection 
sns.set(rc={'figure.figsize':(10,6)})
sns.distplot(df['left'],bins=30)
plt.show()


# In[13]:


# Label Encoder
encoder=LabelEncoder()
df['salary']=encoder.fit_transform(df['salary'])
df.head()


# In[14]:


#Performing Multi Collinearality Test
cm=df.corr().round(2)
#Plot
sns.set(rc={'figure.figsize':(11,8)})
sns.heatmap(cm,annot=True)


# In[15]:


x=df[['salary','satisfaction_level','promotion_last_5years','number_project']]
y=df[['left']]


# In[16]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)


# In[17]:


lin_model=LinearRegression()
lin_model.fit(x_train,y_train)
y_test_pred=lin_model.predict(x_test)
rmse=np.sqrt(mean_squared_error(y_test,y_test_pred))
r2=r2_score(y_test,y_test_pred)

print('RMSE on training data : ',rmse)
print('R2 on training data : ',r2)


# In[18]:


#Logistic regression model
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[19]:


print('Accuracy: ',accuracy_score(y_test,y_pred))


# In[ ]:




