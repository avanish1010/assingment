#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


df=pd.read_csv(r'C:\Users\Avanish\Dropbox\PC\Downloads\TRAIN.csv (1).zip',na_values=['None','?'])
df.head()


# In[8]:


df.info()


# Calculating the percentage of missing values and dropping the columns with more than about 20% missing¶

# In[9]:


df.isna().sum()


# In[10]:


miss_val=(df.isna().sum()/len(df)*100).sort_values(ascending=False)
miss_val[miss_val>0]


# In[11]:


dropped=[]
for col in df.columns:
    if df[col].isna().sum() > 20000:
        dropped.append(col)


# In[ ]:


df=df.drop(columns=dropped)


# In[12]:


#Showing some insights like number of each unique value

df['readmitted_NO'].value_counts()


# In[13]:


df.gender.value_counts()


# In[14]:


#There is an invalid value to drop

df[df['gender']=='Unknown/Invalid']


# In[15]:


df=df.drop(30506)


# In[16]:


df=df.reset_index(drop=True)


# In[17]:


#There is no clear bias towards any specific value
pd.crosstab(df.age,df.readmitted_NO)


# In[18]:


pd.crosstab(df.diabetesMed,df.insulin)


# In[19]:


pd.crosstab(df.diabetesMed,df.readmitted_NO)


# In[20]:


#There is a lot of 9 diagnoses


# In[21]:


plt.rcParams['figure.figsize'] = (10,6)
sns.countplot(df['number_diagnoses'],palette='twilight_shifted')
plt.show()


# In[22]:


#The majority didn't stay in the hospital for a long


# In[23]:


sns.countplot(df['time_in_hospital'])
plt.show()


# In[24]:


#Caucasian is the most frequant in this data


# In[25]:


sns.countplot(df.race,palette='Spectral')
plt.show()


# In[26]:


#And the ages are about > 50 years and it's predicted as it's for diabetes


# In[27]:


sns.countplot(df.age,palette = 'Dark2_r')
plt.show()


# In[28]:


df.info()


# In[29]:


#Dealing with missing values
df.race=df.race.fillna(df.race.mode())


# In[30]:


#Converting into numeric values so as we could fill the NaN values


# In[31]:


df.diag_1=pd.to_numeric(df.diag_1,errors='coerce')
df.diag_2=pd.to_numeric(df.diag_2,errors='coerce')
df.diag_3=pd.to_numeric(df.diag_3,errors='coerce')


# In[32]:


#Let's see mode or mean !


# In[33]:


sns.set(style = 'whitegrid')
sns.distplot(df.diag_1)
plt.show()


# In[34]:


sns.set(style = 'whitegrid')
sns.distplot(df.diag_2)
plt.show()


# In[35]:


sns.set(style = 'whitegrid')
sns.distplot(df.diag_3)
plt.show()


# In[36]:


#Great, the diag_1 and 2 are better filled with mean, but 3's mode is more suitable for it


# In[37]:


df.diag_1=df.diag_1.fillna(df.diag_1.mean())
df.diag_2=df.diag_2.fillna(df.diag_2.mean())


# In[38]:


df.diag_3.mode()


# In[39]:


df.diag_3=df.diag_3.fillna(250)


# In[40]:


#Visualizing the data


# In[41]:


#The relation between both numbers of medications and lab procedures are about the same for readmitted


# In[42]:


fig,(ax1,ax2)=plt.subplots(1,2)
ax1.scatter(x='num_lab_procedures',y='num_medications',data=df[df['readmitted_NO']==0],color='b')
ax2.scatter(x='num_lab_procedures',y='num_medications',data=df[df['readmitted_NO']==1],color='r')
ax1.set_xlabel('Number of lab procedures')
ax1.set_ylabel('Number of medications')
plt.show()


# In[43]:


#It's obvious that ages more than 50 take more medications but the lab procedures isn't in the same range

sns.stripplot(df['age'], df['num_lab_procedures'], palette = 'Purples', size = 10)
plt.show()


# In[44]:


#And the readmitted doesn't depend on lab procedures as well
sns.boxplot(df.readmitted_NO, df.num_lab_procedures)
plt.show()


# In[45]:


#Preparing the data for the model


# In[46]:


#Encoding categorical columns


# In[47]:


cat=['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide',
     'glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol',
     'troglitazone','tolazamide','examide','citoglipton','insulin','glyburide-metformin',
    'glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone',
     'metformin-pioglitazone','change','diabetesMed','race','gender','age']


# In[50]:


for col in cat:
    df[col]=df[col].astype('category')
    df[col]=df[col].cat.codes

df.dropna(inplace=True,axis=1)
#Using PCA to get the most important features

from sklearn.decomposition import PCA
pca=PCA()

pca.fit(df)


# In[51]:


transformed=pca.transform(df)


# In[52]:


#So we get here that is about just 3 columns and applying this would affect the data badly so we tried another method

features=range(pca.n_components_)
plt.bar(features,pca.explained_variance_)
plt.xticks(features)
plt.show()


# In[53]:


#Let's try SelectKBest for feature selection

from sklearn.feature_selection import SelectKBest   #for feature selection
from sklearn.feature_selection import f_classif

test = SelectKBest(score_func=f_classif, k=20)
fit = test.fit(df.drop(columns=['readmitted_NO']), df['readmitted_NO'])
print(sorted(zip(fit.scores_,df.columns),reverse=True))


# In[54]:


#Picking the most unrelevant features to drop

to_drop=['metformin-pioglitazone','metformin-rosiglitazone','glimepiride-pioglitazone',
       'chlorpropamide','troglitazone','insulin','acetohexamide','glipizide-metformin',
       'tolbutamide','glimepiride','glyburide-metformin','citoglipton','examide','miglitol',
       'diag_1','tolazamide','admission_type_id','rosiglitazone','nateglinide','diag_2',
       'glyburide','acarbose','glipizide']

data=df.drop(columns=to_drop)

#Scaling the data using different types of scaling


# In[58]:


#First let's try standarization

from sklearn.preprocessing import StandardScaler
data_scale=StandardScaler().fit_transform(data)


# In[59]:


data_scale=pd.DataFrame(data_scale,columns=data.columns)


# In[60]:


x_scale=data_scale.drop(columns=['readmitted_NO'])


# In[62]:


x=data.drop(columns=['readmitted_NO'])


# In[63]:


#Our model for clustering would be KMeans

from sklearn.cluster import KMeans

model=KMeans(n_clusters=2)


# In[64]:


#Let's see the results without any scaling first

model.fit(x)
labels=model.predict(x)

pd.crosstab(labels,df['readmitted_NO'])


# In[68]:


#47% , we could do better
from sklearn import metrics
metrics.accuracy_score(labels,df['readmitted_NO'])


# In[69]:


#With standarization

model.fit(x_scale)

labels_scale=model.predict(x_scale)

pd.crosstab(labels_scale,df['readmitted_NO'])


# In[70]:


#We get about 53% with standarization

from sklearn import metrics
metrics.accuracy_score(labels_scale,df['readmitted_NO'])

#Using whiten to scale the data

from scipy.cluster.vq import whiten


# In[71]:


data_whiten=whiten(data)
data_whiten=pd.DataFrame(data_whiten,columns=data.columns)


# In[72]:


x_whiten=data_whiten.drop(columns=['readmitted_NO'])
model.fit(x_whiten)
labels_whiten=model.predict(x_whiten)


# In[73]:


pd.crosstab(labels_whiten,df['readmitted_NO'])


# In[74]:


#And we get 53% again

#Notes : whiten is more stable to get the accurecy than standard scaler


metrics.accuracy_score(labels_whiten,df['readmitted_NO'])


# In[76]:


#Preparing the test set

test=pd.read_csv(r'C:\Users\Avanish\Dropbox\PC\Downloads\TEST.csv.zip',na_values=['None','?'])


# In[77]:


test.info()


# In[78]:


#Dropping the columns and getting the set ready to be clustered :"

test=test.drop(columns=dropped)
test.race=test.race.fillna('Caucasian')
test.diag_1=pd.to_numeric(test.diag_1,errors='coerce')
test.diag_2=pd.to_numeric(test.diag_2,errors='coerce')
test.diag_3=pd.to_numeric(test.diag_3,errors='coerce')
test.diag_1=test.diag_1.fillna(test.diag_1.mean())
test.diag_2=test.diag_2.fillna(test.diag_2.mean())
test.diag_3=test.diag_3.fillna(250)


# In[79]:


cat=['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide',
     'glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol',
     'troglitazone','tolazamide','examide','citoglipton','insulin','glyburide-metformin',
    'glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone',
     'metformin-pioglitazone','change','diabetesMed','race','gender','age']


# In[80]:


for col in cat:
    test[col]=test[col].astype('category')
    test[col]=test[col].cat.codes

to_drop=['metformin-pioglitazone','metformin-rosiglitazone','glimepiride-pioglitazone',
       'chlorpropamide','troglitazone','insulin','acetohexamide','glipizide-metformin',
       'tolbutamide','glimepiride','glyburide-metformin','citoglipton','examide','miglitol',
       'diag_1','tolazamide','admission_type_id','rosiglitazone','nateglinide','diag_2',
       'glyburide','acarbose','glipizide']
test=test.drop(columns=to_drop)

test=test.drop(columns=['index'])

#We'd better use "whiten" to scale


# In[81]:


test_w=whiten(test)
test_w=pd.DataFrame(test_w,columns=test.columns)
target=model.predict(test_w)

#And we finished and after submission we get the same accuracy about 53%

