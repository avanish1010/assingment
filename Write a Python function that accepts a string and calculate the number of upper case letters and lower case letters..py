#!/usr/bin/env python
# coding: utf-8

# In[8]:


def upper_lower_count(sentense):
    count_upper=0
    count_lower=0
    for i in sentense:
        if i.isupper():
            count_upper+=1
        elif i.islower():
            count_lower+=1
        else:
            pass
    print( "Number of upper case characters: ",count_upper)
    print("Number of lower case characters: ",count_lower)


# In[9]:


upper_lower_count( 'The quick Brow Fox')


# In[ ]:




