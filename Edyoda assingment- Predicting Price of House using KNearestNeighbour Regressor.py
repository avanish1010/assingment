#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv(r'https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt',index_col=0)


# In[3]:


df.head()


# In[4]:


#Use pandas to get some insights into the data
print(df.describe())


# In[5]:


print(df.info())


# In[6]:


print(df.dtypes)


# In[7]:


print(df.shape)


# In[8]:


print(sns.pairplot(df,palette='bright'))


# In[9]:


print(sns.barplot(x='Bathroom',y='Price',data=df))


# In[10]:


print(sns.barplot(x='Bedroom',y='Price',data=df))


# In[11]:


print(sns.barplot(x='Floor',y='Price',data=df))


# In[12]:


print(sns.barplot(x='Living.Room',y='Price',data=df))


# In[13]:


# Manage data for training & testing (20)
# Finding a better value of k (10)


# In[16]:


y=df['Price']
x=df.drop('Price',axis=1)


# In[18]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[21]:


print(x.shape)
print(y.shape)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[22]:


rmse_val=[]
for k in range(1,20):
    model=KNeighborsRegressor(n_neighbors=k)
    model.fit(x_train,y_train)
    pred=model.predict(x_test)
    error=np.sqrt(mean_squared_error(y_test,pred))
    rmse_val.append(error)
    print('RMSE value for k = ',k,'is:',error)


# In[23]:


k_range=range(1,20)
plt.plot(k_range,rmse_val)
plt.xlabel('K')
plt.ylabel('RMSE')
plt.show()


# In[25]:


model=KNeighborsRegressor(n_neighbors=3)
model.fit(x_train,y_train)
pred=model.predict(x_test)
error=np.sqrt(mean_squared_error(y_test,pred))
print(error)


# In[ ]:


# The optimal value of k is 3

