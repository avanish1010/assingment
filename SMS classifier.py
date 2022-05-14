#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r'C:\Users\Avanish\Dropbox\PC\Downloads\archive (3).zip',encoding='latin-1')
df.head()


# In[3]:


df.drop('Unnamed: 2',axis=1,inplace=True)
df.drop('Unnamed: 3',axis=1,inplace=True)
df.drop('Unnamed: 4',axis=1,inplace=True)


# In[4]:


df.head()


# In[5]:


df.rename(columns={'v1':'target','v2':'text'}, inplace=True)


# In[6]:


from sklearn.preprocessing import LabelEncoder
en = LabelEncoder()
df['target'] = en.fit_transform(df['target'])


# In[7]:


df.sample(5)


# In[8]:


df.isnull().sum()


# In[9]:


df.duplicated().sum()


# In[10]:


# remove duplicates
df.drop_duplicates(keep='first', inplace=True)


# In[11]:


df.shape


# In[12]:


df.target.value_counts()


# In[13]:


sns.countplot(df['target'], orient=True)


# In[14]:


plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f", colors=['gray','k'])
plt.show()


# In[15]:


df['num_characters'] = df['text'].apply(len)


# In[16]:


# count number of words in text
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[17]:


# count number of sentence in text
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
df.head()


# In[18]:


# getting more information of features
df[['num_characters','num_words','num_sentences']].describe()


# In[19]:


# information for (ham) type
df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()


# In[20]:


# information for (spam) type
df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()


# In[21]:


# draw cahracters of different typs upon each other
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')
plt.title('character of spam over ham')

plt.subplot(2,2,2)
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='red')
plt.title('words of spam over ham')

plt.subplot(2,2,3)
sns.histplot(df[df['target'] == 0]['num_sentences'])
sns.histplot(df[df['target'] == 1]['num_sentences'],color='red')
plt.title('sentences of spam over ham')
plt.show()


# In[22]:


# as we can see there is slightly collinearity in data
g = sns.heatmap(df.corr(),annot=True)
g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()


# In[23]:


import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[24]:


def transform_text(text):
    text = text.lower() # lower casing
    text = nltk.word_tokenize(text) # tokenization
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear() # remove special characters
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation: # stopword & punctuation remove
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i)) # stemming
            
    return " ".join(y)


# In[25]:


df['transformed_text'] = df['text'].apply(transform_text)


# In[26]:


df.head()


# In[27]:


# nltk.FreqDist() will find frequent words 
freq= nltk.FreqDist(df['transformed_text'])
sorted_freq = sorted(freq , key = freq.__getitem__, reverse = True)

# plot the graph for most frequent occuring words
for key,val in freq.items():
    print(str(key)+ ' : '+str(val))
freq.plot(20, cumulative=False)


# In[28]:


# Trying different models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[29]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)


# In[30]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc
}


# In[31]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer() # i will use the Bag of Word technique to converting the text into numbers
tfidf = TfidfVectorizer(max_features=3000)


# In[32]:


X = cv.fit_transform(df['transformed_text']).toarray()
y = df['target'].values


# In[33]:


# train test split the text
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=2)


# In[34]:


from sklearn.metrics import accuracy_score,precision_score
def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[35]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[37]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[38]:


performance_df


# In[39]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")
performance_df1.head()


# In[40]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


# From the above we can se that the Random Forest Classifier is the best classifier for the model

