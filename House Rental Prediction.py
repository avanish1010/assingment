#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split


# In[2]:


df=pd.read_csv(r'https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt',sep=',',index_col=0)


# In[3]:


df.head()


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.shape


# In[8]:


df.dtypes


# In[9]:


#Outlier detection
sns.set(rc={'figure.figsize':(10,6)})
sns.distplot(df['Price'],bins=30)
plt.show()


# In[10]:


#removing outliers 
df.drop(df.index[list(np.where(df['Price']>150000))],inplace=True)


# In[11]:


#Outlier detection
sns.set(rc={'figure.figsize':(10,6)})
sns.distplot(df['Price'],bins=20)
plt.show()


# In[12]:


df.shape


# In[13]:


sns.barplot(df['Living.Room'],df['Price'])
plt.show()


# In[14]:


sns.barplot(df['Bathroom'],df['Price'])
plt.show()


# In[15]:


sns.jointplot(df['Price'],df['Bedroom'])
plt.show()


# In[16]:


sns.barplot(df['Bedroom'],df['Price'])
plt.show()


# In[17]:


sns.pairplot(df,hue='Bedroom')


# In[18]:


# checking for null values
df['Price'].notnull().count()


# In[19]:


#this means there ar eno null values


# In[20]:


#Performing multi collinearality test
cm=df.corr().round(2)
#Plotting
sns.set(rc={'figure.figsize':(11,8)})
sns.heatmap(cm,annot=True)


# In[21]:


#Bathroom,Bedroom,Sqft have high correlation with price 
x=df[['Bathroom','Bedroom','Sqft']]
y=df['Price']


# In[22]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)


# In[23]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[24]:


lin_model=LinearRegression()
lin_model.fit(x_train,y_train)


# In[25]:


predicted=lin_model.predict(x_test)
rmse=np.sqrt(mean_squared_error(y_test,predicted))
r2=r2_score(y_test,predicted)
print('Rmse score: ',rmse)
print('R2 score : ', r2)


# In[26]:


from sklearn.linear_model import Ridge
rr=Ridge(alpha=10)

#Fitting the model
rr.fit(x_train,y_train)


# In[27]:


rr_trainning_score=rr.score(x_train,y_train)
rr_score=rr.score(x_test,y_test)
print('Trainning Score : ', rr_trainning_score)
print('Testing Score : ', rr_score)


# In[28]:


from sklearn.linear_model import Lasso

lasso=Lasso(alpha=1)
lasso.fit(x_train,y_train)

lasso_train_score=lasso.score(x_train,y_train)
lasso_test_score=lasso.score(x_test,y_test)
coeff_used=np.sum(lasso.coef_!=0)
print('Trainning Score : ', lasso_train_score)
print('Testing Score : ', lasso_test_score)
print('Out of 13 input features , we used : ', coeff_used)


# In[29]:


plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 100$',zorder=7)
plt.plot(rr.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='blue',label=r'Ridge; $\alpha = 0.01$',zorder=7)


# In[30]:


#We can see that ridge has better effect than lasso 


# In[39]:


#polynomial regression
import operator
from sklearn.preprocessing import PolynomialFeatures

polynomial_features=PolynomialFeatures(degree=3)
x_poly=polynomial_features.fit_transform(x_test)

model=LinearRegression()
model.fit(x_poly,y_test)
y_poly_pred=model.predict(x_poly)

rmse=np.sqrt(mean_squared_error(y_test,y_poly_pred))
r2=r2_score(y_test,y_poly_pred)
print('RMSE : ' , rmse)
print('R2 : ', r2)


# In[ ]:


# We can see thet the linear regression has better results than polynomial regression


# In[ ]:




