#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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
model_df, effect_df = load_data(data_directory, adult_or_pediatric="all")


# In[4]:


# verifying that the ModelIDs in model_df and effect_df are alligned
model_df["ID_allignment_verify"] = np.where(
    effect_df["ModelID"] == model_df["ModelID"], "True", "False"
)
verrify = len(model_df["ID_allignment_verify"].unique())
print(model_df["ID_allignment_verify"])
print(
    f"There is {verrify} output object contained in the ID_allignment_verify column \n"
)


# In[5]:


# assign 'AgeCategory' and 'Sex' columns to the effect dataframe as a single column
presplit_effect_df = effect_df.assign(
    age_and_sex=model_df.AgeCategory.astype(str) + "_" + model_df.Sex.astype(str)
)
presplit_effect_df


# In[6]:


# preparing presplit dataframe to be scaled
col_num = len(presplit_effect_df.columns.to_list())
presplit_effect_scaled_df = presplit_effect_df.iloc[:, 1:col_num-1]

# scaling gene effect data to 0-1 range
scaler = MinMaxScaler(feature_range=(0,1))
presplit_effect_scaled_df = scaler.fit_transform(presplit_effect_scaled_df)

# adding id column and age and sex column back
presplit_effect_scaled_df = pd.DataFrame(presplit_effect_scaled_df)
presplit_effect_scaled_df.insert(0, presplit_effect_df.columns[0], presplit_effect_df[presplit_effect_df.columns[0]])
presplit_effect_scaled_df.insert(col_num-1, presplit_effect_df.columns[col_num-1], presplit_effect_df[presplit_effect_df.columns[col_num-1]])
presplit_effect_scaled_df.columns = presplit_effect_df.columns
presplit_effect_scaled_df


# In[7]:


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


# In[8]:


# creating a list of ModelIDs that correlate to pediatric and adult samples
PA_effect_IDs = new_df["ModelID"].tolist()

PA_IDs = set(PA_effect_IDs) & set(presplit_effect_scaled_df["ModelID"].tolist())

# creating a new gene effect data frame containing correlating ModelIDs to the filtered sample info IDs
PA_effect_df = presplit_effect_scaled_df.loc[
    presplit_effect_scaled_df["ModelID"].isin(PA_IDs)
].reset_index(drop=True)


# In[9]:


# split the data based on age category and sex
train_df, test_df = train_test_split(
    PA_effect_df, test_size=0.15, stratify=PA_effect_df.age_and_sex
)


# In[10]:


# save the TESTING dataframe
test_df = test_df.reset_index(drop=True)
testing_df_output = pathlib.Path("../0.data-download/data/VAE_test_df.csv")
test_df.to_csv(testing_df_output, index=False)
print(test_df.shape)
test_df.head(3)


# In[11]:


# save the TRAINING dataframe
train_df = train_df.reset_index(drop=True)
training_df_output = pathlib.Path("../0.data-download/data/VAE_train_df.csv")
train_df.to_csv(training_df_output, index=False)
print(train_df.shape)
train_df.head(3)

