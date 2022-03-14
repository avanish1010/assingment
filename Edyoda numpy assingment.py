#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np


# In[18]:


#1. Create a null vector of size 10 but the fifth value which is 1.
(np.arange(10)==4).astype(int)


# In[19]:


#2. Create a vector with values ranging from 10 to 49.
np.arange(10,50)


# In[20]:


#3. Create a 3x3 matrix with values ranging from 0 to 8.
x=np.arange(0,9).reshape(3,3)
print(x)


# In[21]:


#4. Find indices of non-zero elements from [1,2,0,0,4,0].
x=np.array([1,2,0,0,4,0])
print("indices of non-zero elements are:",np.nonzero(x))


# In[22]:


#5.Create a 10x10 array with random values and find the minimum and maximum values.
arr=np.random.randint(1,100,(10,10))
print(np.min(arr))
print(np.max(arr))


# In[23]:


#6.Create a random vector of size 30 and find the mean value.
np.random.seed(8)
arr=np.random.randint(0,100,30)
print(arr)
print(np.mean(arr,axis=0))


# In[ ]:




