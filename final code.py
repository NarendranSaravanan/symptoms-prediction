#!/usr/bin/env python
# coding: utf-8

# In[17]:


import csv
import pandas as pd 
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


df = pd.read_excel('N:/sem4/machine learning/project/data_set.xlsx')
print(df)


# In[25]:


df.head()


# In[27]:


data = df.fillna(method='ffill')


# In[28]:


data.head()


# In[33]:


def process_data(data):
    data_list = []
    data_name = data.replace('^', '_').split('_')
    n=1
    for names in data_name:
        if (n%2 == 0):
            data_list.append(names)
        n+=1
    return data_list


# In[36]:


disease_list = []
disease_symptom_dict = defaultdict(list)
disease_symptom_count = {}
count = 0

for idx, row in data.iterrows():
    
    # Get the Disease Names
    if (row['Disease'] !="\xc2\xa0") and (row['Disease'] != ""):
        disease = row['Disease']
        disease_list = process_data(data=disease)
        count = row['Count of Disease Occurrence']

    # Get the Symptoms Corresponding to Diseases
    if (row['Symptom'] !="\xc2\xa0") and (row['Symptom'] != ""):
        symptom = row['Symptom']
        symptom_list = process_data(data=symptom)
        for d in disease_list:
            for s in symptom_list:
                disease_symptom_dict[d].append(s)
            disease_symptom_count[d] = count
            


# In[43]:


# Save cleaned data as CSV
f = open('N:/sem4/machine learning/project/cleaned_data.csv', 'w')

with f:
    writer = csv.writer(f)
    for key, val in disease_symptom_dict.items():
        for i in range(len(val)):
            writer.writerow([key, val[i], disease_symptom_count[key]])
# Read Cleaned Data as DF
df = pd.read_csv('N:/sem4/machine learning/project/cleaned_data.csv')
df.columns = ['disease', 'symptom', 'occurence_count']
df.head()


# In[44]:


df.replace(float('nan'), np.nan, inplace=True)
df.dropna(inplace=True)


# In[45]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df['symptom'])
print(integer_encoded)


# In[46]:


onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)


# In[47]:


cols = np.asarray(df['symptom'].unique())


# In[48]:


df_ohe = pd.DataFrame(columns = cols)


# In[49]:


for i in range(len(onehot_encoded)):
    df_ohe.loc[i] = onehot_encoded[i]


# In[50]:


df_disease = df['disease']


# In[51]:


df_concat = pd.concat([df_disease,df_ohe], axis=1)


# In[52]:


df_concat.drop_duplicates(keep='first',inplace=True)


# In[53]:


cols = df_concat.columns


# In[54]:


cols = cols[1:]
# Since, every disease has multiple symptoms, combine all symptoms per disease per row
df_concat = df_concat.groupby('disease').sum()
df_concat = df_concat.reset_index()


# In[55]:


df_concat.to_csv("N:/sem4/machine learning/project/training_dataset.csv", index=False)
# One Hot Encoded Features
X = df_concat[cols]

# Labels
y = df_concat['disease']


# In[63]:


X


# In[64]:


y


# In[60]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
dt = DecisionTreeClassifier()
clf_dt=dt.fit(X, y)


# In[78]:





# In[79]:


disease_pred = clf_dt.predict(X)
disease_real = y.values
for i in range(0, len(disease_real)):
    if disease_pred[i]!=disease_real[i]:
        print(i)
        print ('Pred: {0}\nActual: {1}\n'.format(disease_pred[i], disease_real[i]))


# In[76]:


len(clf_dt.predict(X))


# In[75]:


len(y.values)


# In[80]:


disease_pred


# In[81]:


disease_real


# In[ ]:




