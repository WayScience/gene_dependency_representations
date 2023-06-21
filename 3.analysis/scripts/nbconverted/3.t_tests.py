#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
sys.path.insert(0, ".././0.data-download/scripts/")
from data_loader import load_data
from scipy.stats import ttest_ind


# In[2]:


latent_df = pd.read_csv("../2.train-VAE/results/latent_df.csv")
metadata_df = pd.read_csv(".././0.data-download/data/metadata_df.csv")
data_dir = "../0.data-download/data/"
model_df, dependency_df = load_data(data_dir, adult_or_pediatric="all")


# In[3]:


# Creating categorized lists of sample IDs used in BVAE training
# note that 10 of the 912 used samples have Unknown Sex

ped_ids = []
adult_ids = []
male_ids = []
female_ids = []
ped_male_ids = []
adult_male_ids = []
ped_female_ids = []
adult_female_ids = []

for index, row in metadata_df.iterrows():
    if row['AgeCategory'] == 'Pediatric':
        ped_ids.append(row['ModelID'])
    
    if row['AgeCategory'] == 'Adult':
        adult_ids.append(row['ModelID'])
    
    if row['Sex'] == 'Male':
        male_ids.append(row['ModelID'])

    if row['Sex'] == 'Female':
        female_ids.append(row['ModelID'])
    
    if row['AgeCategory'] == 'Pediatric' and row['Sex'] == 'Male':
        ped_male_ids.append(row['ModelID'])

    if row['AgeCategory'] == 'Adult' and row['Sex'] == 'Male':
        adult_male_ids.append(row['ModelID'])
    
    if row['AgeCategory'] == 'Pediatric' and row['Sex'] == 'Female':
        ped_female_ids.append(row['ModelID'])

    if row['AgeCategory'] == 'Adult' and row['Sex'] == 'Female':
        adult_female_ids.append(row['ModelID'])


# In[4]:


# Generating latent dataframes for each category and dropping the id column to prep for t tests

adult_latent_df = latent_df.copy()
for index, row in adult_latent_df.iterrows():
    if row['ModelID'] not in adult_ids:
        adult_latent_df.drop(index, inplace=True)
adult_latent_df_float = adult_latent_df.drop(columns=["ModelID"])
adult_latent_df_float.reset_index(drop=True, inplace=True)

ped_latent_df = latent_df.copy()
for index, row in ped_latent_df.iterrows():
    if row['ModelID'] not in ped_ids:
        ped_latent_df.drop(index, inplace=True)
ped_latent_df_float = ped_latent_df.drop(columns=["ModelID"])
ped_latent_df_float.reset_index(drop=True, inplace=True)

male_latent_df = latent_df.copy()
for index, row in male_latent_df.iterrows():
    if row['ModelID'] not in male_ids:
        male_latent_df.drop(index, inplace=True)
male_latent_df_float = male_latent_df.drop(columns=["ModelID"])
male_latent_df_float.reset_index(drop=True, inplace=True)

female_latent_df = latent_df.copy()
for index, row in female_latent_df.iterrows():
    if row['ModelID'] not in female_ids:
        female_latent_df.drop(index, inplace=True)
female_latent_df_float = female_latent_df.drop(columns=["ModelID"])
female_latent_df_float.reset_index(drop=True, inplace=True)

ped_male_latent_df = latent_df.copy()
for index, row in ped_male_latent_df.iterrows():
    if row['ModelID'] not in ped_male_ids:
        ped_male_latent_df.drop(index, inplace=True)
ped_male_latent_df_float = ped_male_latent_df.drop(columns=["ModelID"])
ped_male_latent_df_float.reset_index(drop=True, inplace=True)

adult_male_latent_df = latent_df.copy()
for index, row in adult_male_latent_df.iterrows():
    if row['ModelID'] not in adult_male_ids:
        adult_male_latent_df.drop(index, inplace=True)
adult_male_latent_df_float = adult_male_latent_df.drop(columns=["ModelID"])
adult_male_latent_df_float.reset_index(drop=True, inplace=True)

ped_female_latent_df = latent_df.copy()
for index, row in ped_female_latent_df.iterrows():
    if row['ModelID'] not in ped_female_ids:
        ped_female_latent_df.drop(index, inplace=True)
ped_female_latent_df_float = ped_female_latent_df.drop(columns=["ModelID"])
ped_female_latent_df_float.reset_index(drop=True, inplace=True)

adult_female_latent_df = latent_df.copy()
for index, row in adult_female_latent_df.iterrows():
    if row['ModelID'] not in adult_female_ids:
        adult_female_latent_df.drop(index, inplace=True)
adult_female_latent_df_float = adult_female_latent_df.drop(columns=["ModelID"])
adult_female_latent_df_float.reset_index(drop=True, inplace=True)


# In[5]:


# t tests comparing adult vs ped for each latent dimension

t_test_adult_vs_ped = ttest_ind(adult_latent_df_float, ped_latent_df_float)
t_test_adult_vs_ped = pd.DataFrame(t_test_adult_vs_ped).T
t_test_adult_vs_ped.columns = ["t_stat", "p_value"]
print(t_test_adult_vs_ped.shape)

# Sort to show latent dimensions with most significant p values
t_test_adult_vs_ped.sort_values(by='p_value', ascending = True)


# In[6]:


# t tests comparing male vs female for each latent dimension

t_test_male_vs_female = ttest_ind(male_latent_df_float, female_latent_df_float)
t_test_male_vs_female = pd.DataFrame(t_test_male_vs_female).T
t_test_male_vs_female.columns = ["t_stat", "p_value"]
print(t_test_male_vs_female.shape)

# Sort to show latent dimensions with most significant p values
t_test_male_vs_female.sort_values(by='p_value', ascending = True)


# In[7]:


# t tests comparing adult male vs ped male for each latent dimension

t_test_adult_male_vs_ped_male = ttest_ind(ped_male_latent_df_float, adult_male_latent_df_float)
t_test_adult_male_vs_ped_male = pd.DataFrame(t_test_adult_male_vs_ped_male).T
t_test_adult_male_vs_ped_male.columns = ["t_stat", "p_value"]
print(t_test_adult_male_vs_ped_male.shape)

# Sort to show latent dimensions with most significant p values
t_test_adult_male_vs_ped_male.sort_values(by='p_value', ascending = True)


# In[8]:


# t tests comparing adult female vs ped female for each latent dimension

t_test_adult_female_vs_ped_female = ttest_ind(ped_female_latent_df_float, adult_female_latent_df_float)
t_test_adult_female_vs_ped_female = pd.DataFrame(t_test_adult_female_vs_ped_female).T
t_test_adult_female_vs_ped_female.columns = ["t_stat", "p_value"]
print(t_test_adult_female_vs_ped_female.shape)

# Sort to show latent dimensions with most significant p values
t_test_adult_female_vs_ped_female.sort_values(by='p_value', ascending = True)


# In[9]:


# t tests comparing ped male vs ped female for each latent dimension

t_test_ped_male_vs_ped_female = ttest_ind(ped_female_latent_df_float, ped_male_latent_df_float)
t_test_ped_male_vs_ped_female = pd.DataFrame(t_test_ped_male_vs_ped_female).T
t_test_ped_male_vs_ped_female.columns = ["t_stat", "p_value"]
print(t_test_ped_male_vs_ped_female.shape)

# Sort to show latent dimensions with most significant p values
t_test_ped_male_vs_ped_female.sort_values(by='p_value', ascending = True)


# In[10]:


# t tests comparing adult male vs adult female for each latent dimension

t_test_adult_male_vs_adult_female = ttest_ind(adult_female_latent_df_float, adult_male_latent_df_float)
t_test_adult_male_vs_adult_female = pd.DataFrame(t_test_adult_male_vs_adult_female).T
t_test_adult_male_vs_adult_female.columns = ["t_stat", "p_value"]
print(t_test_adult_male_vs_adult_female.shape)

# Sort to show latent dimensions with most significant p values
t_test_adult_male_vs_adult_female.sort_values(by='p_value', ascending = True)

