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


df=pd.read_csv(r"C:\Users\Avanish\Dropbox\PC\Downloads\archive (1).zip")
df.head()


# In[3]:


del(df['dual_sim'])
del(df['bluetooth'])
#We are removing the two columns as it has only 1 unique value


# In[4]:


df.shape


# In[8]:


df.dtypes


# In[11]:


df.keys()


# In[46]:


df['mobile_price'] = df['mobile_price'].replace({'₹': ''}, regex=True)
df['mobile_price'] = df['mobile_price'].replace({',': ''}, regex=True)
df['mobile_price']=df['mobile_price'].astype(str).astype(int)
df['mobile_price'].dtype


# In[33]:


df[['mp_speed','int_memory','ram','battery_power','mob_width','mob_height','mob_depth','mob_weight']]=df[['mp_speed','int_memory','ram','battery_power','mob_width','mob_height','mob_depth','mob_weight']].astype(str)


# In[34]:


def numericize(df,col):
    return df[col].str.replace('[a-zA-Z]','',regex=True)


# In[35]:


df['mp_speed'] = numericize(df,'mp_speed')

df['int_memory'] = numericize(df,'int_memory')

df['ram'] = numericize(df,'ram')

df['battery_power'] = numericize(df,'battery_power')

df['mob_width'] = numericize(df,'mob_width')

df['mob_height'] = numericize(df,'mob_height')

df['mob_depth'] = numericize(df,'mob_depth')

df['mob_weight'] = numericize(df,'mob_weight')


# In[36]:


df.head()


# In[40]:


df[['mp_speed','int_memory','ram','battery_power','mob_width','mob_height','mob_depth','mob_weight']]=df[['mp_speed','int_memory','ram','battery_power','mob_width','mob_height','mob_depth','mob_weight']].astype(float)


# In[48]:


df.dtypes


# In[49]:


#Outlier detection
sns.set(rc={'figure.figsize':(10,6)})
sns.distplot(df['mobile_price'],bins=30)
plt.show()


# In[50]:


#removing outliers 
df.drop(df.index[list(np.where(df['mobile_price']>40000))],inplace=True)


# In[51]:


#Outlier detection
sns.set(rc={'figure.figsize':(10,6)})
sns.distplot(df['mobile_price'],bins=30)
plt.show()


# In[53]:


df.dtypes


# In[54]:


mask = np.zeros_like(df.corr())
mask[np.triu_indices_from(mask)] = True

sns.heatmap(df.corr(),mask=mask, annot=True)


# In[55]:


#Performing multi collinearality test
cm=df.corr().round(2)
#Plotting
sns.set(rc={'figure.figsize':(11,8)})
sns.heatmap(cm,annot=True)


# In[56]:


def plot_correlation_with_col(df, col):
    sorted_corr = df.corr().sort_values(col)[col][:-1]
    sorted_corr.plot(kind='bar',figsize = (10,5))


# In[58]:


plot_correlation_with_col(df,'mobile_price')


# In[59]:


def replace_regex_numericize(df,col,chars_to_remove):
    processed = df[col]
    for char in chars_to_remove:
        processed = processed.str.replace(char,'',regex=True)
    return pd.to_numeric(processed)


# In[63]:


#edit_numericize
df['disp_size']=df['disp_size'].astype(str)
df['disp_size'] = replace_regex_numericize(df,'disp_size',[' cm .*'])


# In[64]:


df['disp_size']=df['disp_size'].astype(float)


# In[65]:


plot_correlation_with_col(df,'mobile_price')


# In[69]:


def replace_regex(df,col,chars_to_remove):
    processed = df[col]
    for char in chars_to_remove:
        processed = processed.str.replace(char,'',regex=True)
    return processed
df['mobile_name_brand'] = replace_regex(df, 'mobile_name', [' \(.*\)'])


# In[70]:


df.head()


# In[71]:


df['mobile_name_brand_short'] = replace_regex(df, 'mobile_name', [' .*'])


# In[72]:


df.head()


# In[74]:


def convert_words_to_num(df,cat_col):
    return df[cat_col].astype('category').cat.codes
df['mobile_name_brand_new'] = convert_words_to_num(df,'mobile_name_brand')
df['mobile_name_brand_short_new'] = convert_words_to_num(df,'mobile_name_brand')


# In[75]:


df.head()


# In[76]:


#Performing multi collinearality test
cm=df.corr().round(2)
#Plotting
sns.set(rc={'figure.figsize':(11,8)})
sns.heatmap(cm,annot=True)


# In[80]:


x=df[['disp_size','mp_speed','int_memory','ram']]
y=df['mobile_price']


# In[78]:


def plot_grouped_correlation_with_col(df, col):
    sorted_corr = df.corr().sort_values(col)[col][:-1]
    sorted_corr.sort_index().plot(kind='bar',figsize = (10,5))


# In[79]:


plot_grouped_correlation_with_col(df,'mobile_price')


# In[81]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)


# In[82]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[83]:


lin_model=LinearRegression()
lin_model.fit(x_train,y_train)


# In[84]:


predicted=lin_model.predict(x_test)
rmse=np.sqrt(mean_squared_error(y_test,predicted))
r2=r2_score(y_test,predicted)
print('Rmse score: ',rmse)
print('R2 score : ', r2)


# In[85]:


#Ridge
from sklearn.linear_model import Ridge
rr=Ridge(alpha=10)

#Fitting the model
rr.fit(x_train,y_train)


# In[86]:


rr_trainning_score=rr.score(x_train,y_train)
rr_score=rr.score(x_test,y_test)
print('Trainning Score : ', rr_trainning_score)
print('Testing Score : ', rr_score)


# In[87]:


from sklearn.linear_model import Lasso

lasso=Lasso(alpha=1)
lasso.fit(x_train,y_train)

lasso_train_score=lasso.score(x_train,y_train)
lasso_test_score=lasso.score(x_test,y_test)
coeff_used=np.sum(lasso.coef_!=0)
print('Trainning Score : ', lasso_train_score)
print('Testing Score : ', lasso_test_score)
print('Out of 13 input features , we used : ', coeff_used)


# In[88]:


plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 100$',zorder=7)
plt.plot(rr.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='blue',label=r'Ridge; $\alpha = 0.01$',zorder=7)


# In[89]:


#We can see that ridge has better effect than lasso 

