#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[5]:


df=pd.read_csv(r"https://raw.githubusercontent.com/SR1608/Datasets/main/covid-data.csv")
df.head()


# In[ ]:


'''
2. High Level Data Understanding:
a. Find no. of rows & columns in the dataset
b. Data types of columns.
c. Info & describe of data in dataframe.
'''


# In[6]:


print(df.shape)


# In[17]:


print(df.dtypes)


# In[18]:


print(df.info())


# In[19]:


print(df.describe())


# In[ ]:


'''
3. Low Level Data Understanding :
 a. Find count of unique values in location column.
 b. Find which continent has maximum frequency using values 
counts.
 c. Find maximum & mean value in 'total_cases'.
 d. Find 25%,50% & 75% quartile value in 'total_deaths'.
 e. Find which continent has maximum 
'human_development_index'.
 f. Find which continent has minimum 'gdp_per_capita'.
'''


# In[12]:


print(df.nunique())


# In[28]:


print(df['continent'].value_counts())


# In[41]:


print('Maximim value is: ',df['total_cases'].max(),'\n\nMean value is: ',df['total_cases'].mean())


# In[84]:


df['total_deaths'].describe()


# In[56]:


df[df['human_development_index'] == max(df['human_development_index'] )]['continent'].tail(1)


# In[57]:


df[df['gdp_per_capita'] == max(df['gdp_per_capita'] )]['continent'].tail(1)


# In[ ]:


'''
4. Filter the dataframe with only this columns
['continent','location','date','total_cases','total_deaths','gdp_per_ca
pita','
human_development_index'] and update the data frame.
'''


# In[71]:


df=df[['continent','location','date','total_cases','total_deaths','gdp_per_capita','human_development_index']]


# In[ ]:


'''
5. Data Cleaning
 a. Remove all duplicates observations
 b. Find missing values in all columns
 c. Remove all observations where continent column value is 
missing
 Tip : using subset parameter in dropna
 d. Fill all missing values with 0
 '''


# In[87]:


df.drop_duplicates(inplace=True)


# In[78]:


df.isnull().sum()


# In[85]:


df.dropna(subset=['continent'],inplace=True)


# In[98]:


df.fillna(value=0,axis='columns',inplace=True)


# In[ ]:


'''
6. Date time format :
 a. Convert date column in datetime format using 
pandas.to_datetime
 b. Create new column month after extracting month data from 
date
 column.
'''


# In[100]:


df['date']=pd.to_datetime(df['date'])


# In[102]:


df['MONTH']=pd.DatetimeIndex(df['date']).month


# In[ ]:


'''
7)
df.grpby(con).max.reset_index()
7. Data Aggregation:
 a. Find max value in all columns using groupby function on 
'continent'
 column
 Tip: use reset_index() after applying groupby
 b. Store the result in a new dataframe named 'df_groupby'.
 (Use df_groupby dataframe for all further analysis)
'''


# In[113]:


df_groupby=df.groupby('continent').max().reset_index()


# In[114]:


df_groupby


# In[ ]:


'''
8. Feature Engineering :
 a. Create a new feature 'total_deaths_to_total_cases' by ratio of
 'total_deaths' column to 'total_cases'
'''


# In[128]:


df_groupby['total_deaths_to_total_cases'] = df_groupby['total_deaths'] / df_groupby['total_cases']
df_groupby.head()


# In[ ]:


'''
9. Data Visualization :
 a. Perform Univariate analysis on 'gdp_per_capita' column by 
plotting
 histogram using seaborn dist plot.
 b. Plot a scatter plot of 'total_cases' & 'gdp_per_capita'
 c. Plot Pairplot on df_groupby dataset.
 d. Plot a bar plot of 'continent' column with 'total_cases' .
 Tip : using kind='bar' in seaborn catplot
 '''


# In[124]:


sns.displot(data=df_groupby['gdp_per_capita'],kind='hist')


# In[126]:


sns.scatterplot(x='total_cases',y='gdp_per_capita',data=df_groupby)


# In[129]:


sns.pairplot(df_groupby)


# In[132]:


sns.catplot(kind='bar',data=df_groupby,x='continent',y='total_cases')


# In[ ]:


'''
10.Save the df_groupby dataframe in your local drive using 
pandas.to_csv
 function 
'''


# In[136]:


df_groupby.to_csv('df_grouby.csv')


# In[ ]:




