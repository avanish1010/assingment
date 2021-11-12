#!/usr/bin/env python
# coding: utf-8

# In[4]:


mini_dict = {}
for char in range(97,123):
    mini_dict.setdefault(chr(char),char)
print(mini_dict)


# In[ ]:




