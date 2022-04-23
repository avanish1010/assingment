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


df=pd.read_csv(r'C:\Users\Avanish\Dropbox\PC\Downloads\airbnb_listing_train.csv.zip')
df.head()


# In[3]:


df.describe()


# In[4]:


df.info()


# In[5]:


df.dtypes


# In[7]:


df.shape


# In[8]:


#dropping name and host_name as we have id and host_id
#dropping neighbourhood_group as entire column has missing value
df.drop(['name','host_name','neighbourhood_group'], axis=1, inplace =True)


# In[9]:


#dropping last_review and reviews_per_month as we have number_of_reviews which is more relevant compare to previous two columns
df.drop(['last_review','reviews_per_month'], axis=1, inplace =True)


# In[10]:


#convert to category dtype
df['neighbourhood'] = df['neighbourhood'].astype('category')


# In[11]:


df.shape


# In[13]:


df.notnull().sum()


# In[14]:


df.isnull().sum()


# In[19]:


#Outlier detection
sns.set(rc={'figure.figsize':(10,6)})
sns.distplot(df['price'],bins=30)
plt.show()


# In[29]:


df.head()


# In[30]:


df.keys()


# In[31]:


x = df[['id', 'host_id','latitude', 'longitude',
       'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count',
       'availability_365', 'price']]
y = df['price']


# In[32]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=5)


# In[33]:


#Performing multi collinearality test
cm=df.corr().round(2)
#Plotting
sns.set(rc={'figure.figsize':(11,8)})
sns.heatmap(cm,annot=True)


# In[34]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[35]:


lin_model=LinearRegression()
lin_model.fit(x_train,y_train)


# In[36]:


predicted=lin_model.predict(x_test)
rmse=np.sqrt(mean_squared_error(y_test,predicted))
r2=r2_score(y_test,predicted)
print('Rmse score: ',rmse)
print('R2 score : ', r2)


# In[37]:


#Ridge
from sklearn.linear_model import Ridge
rr=Ridge(alpha=10)

#Fitting the model
rr.fit(x_train,y_train)


# In[38]:


rr_trainning_score=rr.score(x_train,y_train)
rr_score=rr.score(x_test,y_test)
print('Trainning Score : ', rr_trainning_score)
print('Testing Score : ', rr_score)


# In[39]:


from sklearn.linear_model import Lasso

lasso=Lasso(alpha=1)
lasso.fit(x_train,y_train)

lasso_train_score=lasso.score(x_train,y_train)
lasso_test_score=lasso.score(x_test,y_test)
coeff_used=np.sum(lasso.coef_!=0)
print('Trainning Score : ', lasso_train_score)
print('Testing Score : ', lasso_test_score)
print('Out of 13 input features , we used : ', coeff_used)


# In[40]:


plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 100$',zorder=7)
plt.plot(rr.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='blue',label=r'Ridge; $\alpha = 0.01$',zorder=7)


# In[41]:


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

