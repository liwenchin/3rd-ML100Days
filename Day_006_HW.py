#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd


# In[3]:


dir_data = 'C:/Users/a0910/Documents/GitHub/3rd-ML100Days/Part01'
f_app_train = os.path.join(dir_data, 'application_train.csv')
f_app_test = os.path.join(dir_data, 'application_test.csv')

app_train = pd.read_csv(f_app_train)
app_test = pd.read_csv(f_app_test)


# In[4]:



app_train.dtypes.value_counts()


# In[6]:


app_train.select_dtypes(include=["object"]).apply(pd.Series.nunique, axis = 0)


# In[7]:


from sklearn.preprocessing import LabelEncoder


# In[8]:


# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in app_train:
    if app_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(app_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(app_train[col])
            # Transform both training and testing data
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)


# In[9]:


app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print(app_train['CODE_GENDER_F'].head())
print(app_train['CODE_GENDER_M'].head())
print(app_train['NAME_EDUCATION_TYPE_Academic degree'].head())


# In[10]:


app_train = pd.read_csv(f_app_train)
sub_train = pd.DataFrame(app_train['WEEKDAY_APPR_PROCESS_START'])
print(sub_train.shape)
sub_train.head()


# In[3]:


import os
import numpy as np
import pandas as pd

dir_data = 'C:/Users/a0910/Documents/GitHub/3rd-ML100Days/Part01'
f_app_train = os.path.join(dir_data, 'application_train.csv')
f_app_test = os.path.join(dir_data, 'application_test.csv')

app_train = pd.read_csv(f_app_train)
app_test = pd.read_csv(f_app_test)

app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)
print(app_train['WEEKDAY_APPR_PROCESS_START'].head())


# In[ ]:




