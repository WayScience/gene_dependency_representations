#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, "../0.data-download/scripts/")
from data_loader import load_data
from sklearn.model_selection import train_test_split
import random


# In[2]:


random.seed(18)
print(random.random())


# In[3]:


# load all of the data
data_directory = "../0.data-download/data/"
model_df, dependency_df = load_data(data_directory, adult_or_pediatric="all")


# In[4]:


# verifying that the ModelIDs in model_df and dependency_df are alligned
model_df["ID_allignment_verify"] = np.where(
    dependency_df["ModelID"] == model_df["ModelID"], "True", "False"
)
verrify = len(model_df["ID_allignment_verify"].unique())
print(model_df["ID_allignment_verify"])
print(
    f"There is {verrify} output object contained in the ID_allignment_verify column \n"
)


# In[5]:


# assign 'AgeCategory' and 'Sex' columns to the dependency dataframe as a single column
presplit_dependency_df = dependency_df.assign(
    age_and_sex=model_df.AgeCategory.astype(str) + "_" + model_df.Sex.astype(str)
)
presplit_dependency_df


# In[6]:


groups = model_df.groupby("AgeCategory")
df_list = []
for name, df in groups:

    # only looking for samples that contain Adult or Pediatric information
    if name == "Adult" or name == "Pediatric":
        df_list.append(df)

# merge sample dataframes through concatentation and reorganize so that ModelIDs are in alphabetical order
new_df = pd.concat(df_list, axis=0)
new_df = new_df.set_index("ModelID")
new_df = new_df.sort_index(ascending=True)
new_df = new_df.reset_index()


# In[7]:


# creating a list of ModelIDs that correlate to pediatric and adult samples
PA_dependency_IDs = new_df["ModelID"].tolist()

PA_IDs = set(PA_dependency_IDs) & set(presplit_dependency_df["ModelID"].tolist())

# creating a new gene dependency data frame containing correlating ModelIDs to the filtered sample info IDs
PA_dependency_df = presplit_dependency_df.loc[
    presplit_dependency_df["ModelID"].isin(PA_IDs)
].reset_index(drop=True)


# In[8]:


# split the data based on age category and sex
train_df, test_df = train_test_split(
    PA_dependency_df, test_size=0.15, stratify=PA_dependency_df.age_and_sex
)


# In[9]:


# save the TESTING dataframe
test_df = test_df.reset_index(drop=True)
testing_df_output = pathlib.Path("../0.data-download/data/VAE_test_df.csv")
test_df.to_csv(testing_df_output, index=False)
print(test_df.shape)
test_df.head(3)


# In[10]:


# save the TRAINING dataframe
train_df = train_df.reset_index(drop=True)
training_df_output = pathlib.Path("../0.data-download/data/VAE_train_df.csv")
train_df.to_csv(training_df_output, index=False)
print(train_df.shape)
train_df.head(3)

