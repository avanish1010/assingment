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


df=pd.read_csv(r'C:\Users\Avanish\Dropbox\PC\Downloads\archive.zip')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.dtypes


# In[6]:


df.info()


# In[7]:


df.shape


# In[8]:


df.nunique()


# In[9]:


df.notna().count()


# In[10]:


df.isnull().count()


# In[11]:


df[['Data_Value_Unit']]


# In[12]:


df[['Data_Value_Footnote']]


# In[13]:


df[['Data_Value_Footnote_Symbol']]


# In[14]:


df.shape


# In[15]:


df.head()


# In[16]:


from pandas_profiling import ProfileReport


# In[17]:


profile = ProfileReport(df)
profile


# In[28]:


y=df['Data_Value']


# In[29]:


x=df[['Low_Confidence_Limit','Data_Value_Alt']]


# In[27]:


df = df.interpolate()


# In[30]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
lin_model=LinearRegression()
lin_model.fit(x_train,y_train)
y_test_pred=lin_model.predict(x_test)
rmse=np.sqrt(mean_squared_error(y_test,y_test_pred))
r2=r2_score(y_test,y_test_pred)

print('RMSE on training data : ',rmse)
print('R2 on training data : ',r2)


# In[34]:


# Using K neighbors
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
rmse_val=[]
for k in range(1,20):
    #k=k+1
    model=KNeighborsRegressor(n_neighbors=k)
    model.fit(x_train,y_train)
    pred=model.predict(x_test)
    error=np.sqrt(mean_squared_error(y_test,pred))
    rmse_val.append(error)
    print('RMSE value for k = ',k,'is:',error)


# In[37]:


#optimum model
model=KNeighborsRegressor(n_neighbors=2)
model.fit(x_train,y_train)
pred=model.predict(x_test)
error=np.sqrt(mean_squared_error(y_test,pred))
print(error)


# In[ ]:




