#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages that will be utilized
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import pathlib
import seaborn as sns

# assign the desired file a variable using pathlib.Path command
input_file = pathlib.Path("../0.data-download/data/sample_info.csv")

# set the data frame to be the desired .csv file that is read by pandas(pd) using the pd.read_csv(desired file read as a previously defined variable)
df = pd.read_csv(input_file)

# print the parameters of the read file
print(df.shape)


# In[2]:


# how many samples?
n_samples = len(df["DepMap_ID"].unique())
print(f"Number of Samples: {n_samples} \n")

# how many different ages were sampled from? 
all_ages = df["age"].unique()
print(f"Ages sampled from: \n {all_ages} \n")

# how many different types of cancer?
all_cancers = df["primary_disease"].unique()
print(f"All Cancer Types: \n {all_cancers} \n")


# In[3]:


# "Adult" was set to 18
# "Pediatric" was set to 0
# "Fetus" was set to 0

# create a new data frame that excludes blank cells in the data sets "Age" column
no_nan_age_df = df.loc[df["age"].notnull()]

# create a new data frame that will also assign the integer value of 18 to cells containing "Adult"
no_nan_age_df.loc[(df["age"] == "Adult")] = 18

# create a new data frame that will also assign the integer value of 0 to cells containing "Pediatric"
no_nan_age_df.loc[(df["age"] == "Pediatric")] = 0

# create a new data frame that will also assign the integer value of -1 to cells containg "Fetus"
no_nan_age_df.loc[(df["age"] == "Fetus")] = -1



adult_df = no_nan_age_df.loc[no_nan_age_df["age"].astype(int) >= 18]
adult_df = adult_df.astype({"age": int})
pediatric_df = no_nan_age_df.loc[(no_nan_age_df["age"].astype(int) < 18) & (no_nan_age_df["age"].astype(int) != -1)]
pediatric_df = pediatric_df.astype({"age": int})


# In[4]:


ig, axs = plt.subplots(1, 2, dpi=150)

# plot adult age distribution 
sns.histplot(x="age", data=adult_df, stat="count", ax=axs[0])

# plot pediatric age distribution
sns.histplot(x="age", data=pediatric_df, stat="count", ax=axs[1])

# visualizes the general distribution of ages sampled from


# In[5]:


pediatric_df.loc[pediatric_df["age"] == 1]

