#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Write a Python class to implement pow(x, n)


# In[2]:


class power_class():
    def __init__(self,x,n):
        self.x=x
        self.n=n
    def power(self):
        return self.x**self.n
pow1=power_class(10,2)
pow1.power()


# In[ ]:




