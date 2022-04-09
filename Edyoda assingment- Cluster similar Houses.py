#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv(r'https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt',index_col=0)


# In[3]:


df.head()


# In[4]:


mms=StandardScaler()
df[['Sqft','Floor','Bedroom','Living.Room','Bathroom','Price']]=mms.fit_transform(df[['Sqft','Floor','Bedroom','Living.Room','Bathroom','Price']])


# In[5]:


df.head()


# In[6]:


distortions=[]
inertias=[]
mapping1={}
mapping2={}
K=range(1,15)

array1=df['Sqft'].to_numpy()
array2=df['Floor'].to_numpy()
array3=df['Bedroom'].to_numpy()
array4=df['Living.Room'].to_numpy()
array5=df['Bathroom'].to_numpy()

array=np.array(list(zip(array1,array2,array3,array4,array5))).reshape(len(array1),5)

for k in K:
    kmeanmodel=KMeans(n_clusters=k)
    kmeanmodel.fit(array)
    
    distortions.append(sum(np.min(cdist(array,kmeanmodel.cluster_centers_,
                                       'euclidean'),axis=1))/array.shape[0])
    inertias.append(kmeanmodel.inertia_)
    
    mapping1[k]=sum(np.min(cdist(array,kmeanmodel.cluster_centers_,
                                'euclidean'),axis=1))/array.shape[0]
    mapping2[k]=kmeanmodel.inertia_
    


# In[7]:


for key, val in mapping1.items():
    print(str(key)+' : '+str(val))


# In[8]:


plt.plot(K,distortions,'bx-')
plt.xlabel('Values of K')          
plt.ylabel('Distortions')
plt.title('the elbvow method using distortions')
plt.show()


# In[9]:


for key, val in mapping1.items():
    print(str(key)+' : '+str(val))

plt.plot(K,inertias,'bx-')
plt.xlabel('Values of K')          
plt.ylabel('inertias')
plt.title('the elbvow method using distortions')
plt.show()


# In[16]:


data=pd.DataFrame(array,columns=('Sqft','Floor','Bedroom','Living.Room','Bathroom'))
data.head()
# Optimal number of cllusters is 5
kmeans=KMeans(n_clusters=5).fit(data)
centroids= kmeans.cluster_centers_
print(centroids)


# In[17]:


plt.scatter(data['Bedroom'],data['Living.Room'],c=kmeans.labels_.astype(float))
plt.scatter(centroids[:,0],centroids[:,1],c='red')
plt.show()


# In[ ]:




