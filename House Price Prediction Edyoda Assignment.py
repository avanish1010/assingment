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
import numpy as np


# In[2]:


trainning_data=pd.read_csv(r"C:\Users\Avanish\Dropbox\PC\Downloads\train.csv")


# In[3]:


trainning_data.head()


# In[4]:


trainning_data.shape


# In[5]:


trainning_data.describe()


# In[6]:


trainning_data.info()


# In[7]:


trainning_data.isna().count()


# In[8]:


trainning_data.dropna(axis=1,inplace=True)


# In[9]:


trainning_data.isna().count()


# In[10]:


trainning_data.nunique()


# In[11]:


trainning_data.isnull().count()


# In[12]:


trainning_data.notnull().count()


# In[13]:


trainning_data.dtypes


# In[18]:


#Outlier detection 
sns.set(rc={'figure.figsize':(10,6)})
sns.distplot(trainning_data['SalePrice'],bins=30)
plt.show()


# In[19]:


trainning_data.keys()


# In[39]:


sns.barplot(x='MSSubClass',y='SalePrice',data=trainning_data)


# In[60]:


x=trainning_data[['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath','Fireplaces', 'GarageCars']]
y=trainning_data['SalePrice']


# In[61]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
#Building the model
lin_model=LinearRegression()
lin_model.fit(x_train,y_train)

y_test_pred=lin_model.predict(x_test)
rmse=np.sqrt(mean_squared_error(y_test,y_test_pred))
r2=r2_score(y_test,y_test_pred)

print('RMSE on training data : ',rmse)
print('R2 on training data : ',r2)


# In[62]:


#Logistic regression model
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[63]:


print('Accuracy: ',accuracy_score(y_test,y_pred))


# In[ ]:




