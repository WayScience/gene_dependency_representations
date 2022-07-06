#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pathlib
import numpy as np
import pandas as pd
sys.path.insert(0, "./0.data-download/scripts/")
from data_loader import load_data
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.decomposition import PCA
from tensorflow import keras

from tensorflow.keras.models import Model, Sequential
import seaborn
import random as python_random
import tensorflow as tf


# In[2]:


# load the data 
data_directory = "./0.data-download/data/"
dfs = load_data(data_directory, adult_or_pediatric = "all")
dependency_df = dfs[1]
sample_df = dfs[0]


# In[24]:


# searching for nulls
nan_rows  = dependency_df[dependency_df.isna().any(axis=1)]
nan_rows


# In[4]:


groups = sample_df.groupby("age_categories")
df_list = []
for name, df in groups:
    
    # only looking for adult pediatric
    if name == "Adult" or name == "Pediatric":
        df_list.append(df)
        
# merge dataframes through concatentation 
new_df = pd.concat(df_list, axis=0)
new_df


# In[5]:


ref_df = new_df[["DepMap_ID", "sex", "age_categories"]]


# In[6]:


ref_df.loc[ref_df["age_categories"] == "Pediatric"].reset_index(drop=True)


# In[7]:


# data frame containing ALL the PEDIATRIC samples
bulk_pediatric_training_df = ref_df.loc[ref_df["age_categories"] == "Pediatric"].reset_index(drop=True)
print(bulk_pediatric_training_df.shape)
bulk_pediatric_training_df.head(3)


# In[8]:


# data frame containing ALL the ADULT samples
bulk_adult_training_df = ref_df.loc[ref_df["age_categories"] == "Adult"].reset_index(drop=True)
print(bulk_adult_training_df.shape)
bulk_adult_training_df.head(3)


# In[9]:


# sorting out 103 rows (85% of the PEDIATRIC samples) for the TRAINING data frame

pre_merge_pediatric_training_df = bulk_pediatric_training_df[0:103].reset_index(drop=True)
print(pre_merge_pediatric_training_df.shape)
pre_merge_pediatric_training_df.head(3)


# In[10]:


# sorting out 18 rows (15% of the PEDIATRIC samples) for the TESTING data frame
pre_merge_pediatric_testing_df = bulk_pediatric_training_df[103:].reset_index(drop=True)
print(pre_merge_pediatric_testing_df.shape)
pre_merge_pediatric_testing_df.head(3)


# In[11]:


# sorting out 649 rows (85% of the ADULT samples) for the TRAINING data frame
pre_merge_adult_training_df = bulk_adult_training_df[0:649]
print(pre_merge_adult_training_df.shape)
pre_merge_adult_training_df.head(3)


# In[12]:


# sorting out 114 rows (15% of the ADULT samples) for the TESTING data frame
pre_merge_adult_testing_df = bulk_adult_training_df[649:].reset_index(drop=True)
print(pre_merge_adult_testing_df.shape)
pre_merge_adult_testing_df.head(3)


# In[13]:


# merging the TRAINING data frames 
training_merge_frames = [pre_merge_adult_training_df, pre_merge_pediatric_training_df]
training_df_IDs = pd.concat(training_merge_frames).reset_index(drop=True)
training_df_IDs


# In[14]:


# merging the TRAINING data frames 
testing_merge_frames = [pre_merge_adult_testing_df, pre_merge_pediatric_testing_df]
testing_df_IDs = pd.concat(testing_merge_frames).reset_index(drop=True)
testing_df_IDs


# In[15]:


print(dependency_df.shape)
dependency_df.head()


# In[16]:


# searching for similar IDs FROM the training_df_IDs IN the dependency_df
training_df_IDs = training_df_IDs["DepMap_ID"].tolist()
training_df_IDs = set(training_df_IDs) & set(dependency_df["DepMap_ID"].tolist())


# In[17]:


training_df = dependency_df.loc[dependency_df["DepMap_ID"].isin(training_df_IDs)].reset_index(drop=True)
print(training_df.shape)
training_df.head(3)


# In[18]:


# searching for similar IDs FROM the testing_df_IDs IN the dependency_df
testing_df_IDs = testing_df_IDs["DepMap_ID"].tolist()
testing_df_IDs = set(testing_df_IDs) & set(dependency_df["DepMap_ID"].tolist())


# In[19]:


testing_df = dependency_df.loc[dependency_df["DepMap_ID"].isin(testing_df_IDs)].reset_index(drop=True)
print(testing_df.shape)
testing_df

