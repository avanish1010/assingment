#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[51]:


users = pd.read_excel(r"C:\Users\Avanish\OneDrive\Documents\Edyoda assingment excel.xlsx",index_col=0)


# In[52]:


users.head(10)


# In[53]:


users.tail(10)


# In[54]:


users.shape[0]


# In[55]:


users.shape[1]


# In[60]:


users.keys()


# In[64]:


users.dtypes


# In[69]:


users.iloc[:,2:3]


# In[71]:


users['occupation'].nunique()


# In[72]:


users.info()


# In[89]:


users['occupation'].value_counts()


# In[90]:


users['age'].mean()


# In[91]:


users['age'].min()


# In[ ]:




