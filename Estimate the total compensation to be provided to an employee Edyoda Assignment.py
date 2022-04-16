#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.neighbors import KNeighborsRegressor


# In[2]:


df=pd.read_csv(r"C:\Users\Avanish\Dropbox\PC\Downloads\train_set.csv")
df.head()


# In[3]:


df.describe()


# In[4]:


df.notnull().count()


# In[5]:


df.shape


# In[6]:


df.nunique()


# In[7]:


df.dtypes


# In[8]:


df.notnull()


# In[9]:


df.count()


# In[10]:


#Outlier detection 
sns.set(rc={'figure.figsize':(10,6)})
sns.distplot(df['Total_Compensation'],bins=30)
plt.show()


# In[11]:


#Performing Multi Collinearality Test
cm=df.corr().round(2)
#Plot
sns.set(rc={'figure.figsize':(11,8)})
sns.heatmap(cm,annot=True)


# In[12]:


x=df[['Salaries','Overtime']]
y=df['Total_Compensation']


# In[13]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)


# In[14]:


#Using Linear Regression
lin_model=LinearRegression()
lin_model.fit(x_train,y_train)
y_test_pred=lin_model.predict(x_test)
rmse=np.sqrt(mean_squared_error(y_test,y_test_pred))
r2=r2_score(y_test,y_test_pred)

print('RMSE on training data : ',rmse)
print('R2 on training data : ',r2)


# In[16]:


# Using K neighbors
import numpy as np
rmse_val=[]
for k in range(30,45):
    #k=k+1
    model=KNeighborsRegressor(n_neighbors=k)
    model.fit(x_train,y_train)
    pred=model.predict(x_test)
    error=np.sqrt(mean_squared_error(y_test,pred))
    rmse_val.append(error)
    print('RMSE value for k = ',k,'is:',error)


# In[18]:


k_range=range(30,45)
plt.plot(k_range,rmse_val)
plt.xlabel('K')
plt.ylabel('RMSE')
plt.show()


# In[19]:


#optimum model
model=KNeighborsRegressor(n_neighbors=31)
model.fit(x_train,y_train)
pred=model.predict(x_test)
error=np.sqrt(mean_squared_error(y_test,pred))
print(error)


# In[ ]:




