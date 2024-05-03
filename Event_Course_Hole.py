#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[5]:


df = pd.read_csv('/Users/danielbrown/Desktop/combined_df_all_clusters.csv')
df


# In[6]:


df_event_course = pd.read_csv('/Users/danielbrown/Desktop/Event_Name_Courses - Sheet1.csv')
#df_event_course


# In[7]:


# Splitting courses with delimiter '/' and then exploding the dataframe
#df_event_course['Course'] = df_event_course['Course'].str.split('/')
#df_event_course = df_event_course.explode('Course')

# Displaying the modified dataframe
#print(df_event_course)


# In[8]:


#!pip install fuzzywuzzy


# In[9]:


from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Function to find best match
def find_best_match(row, choices):
    match, score, idx = process.extractOne(row['Course'], choices, scorer=fuzz.partial_ratio)
    if score >= 70:  # Threshold for considering a match
        return match
    else:
        return None

# Apply fuzzy matching to find best matches
df['Course'] = df.apply(lambda row: find_best_match(row, df_event_course['Course']), axis=1)

# Left join on the best matches
merged_df = pd.merge(df_event_course, df, how='left', left_on='Course', right_on='Course')

print(merged_df)


# In[10]:


merged_df.to_csv(r"/Users/danielbrown/Desktop/combined_df_clusters_.csv")


# In[ ]:




