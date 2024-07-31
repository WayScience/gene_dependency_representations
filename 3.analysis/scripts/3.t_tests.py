#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
sys.path.insert(0, "../utils/")
from data_loader import load_data
from scipy.stats import ttest_ind 
from scipy.stats import f_oneway
import pathlib


# In[2]:


latent_df = pd.read_parquet("../2.train-VAE/results/latent_df.parquet")
metadata_df = pd.read_parquet(".././0.data-download/data/metadata_df.parquet")
data_dir = "../0.data-download/data/"
model_df, dependency_df = load_data(data_dir, adult_or_pediatric="all")


# In[3]:


# Creating categorized lists of sample IDs used in BVAE training
# note that 10 of the 912 used samples have Unknown Sex

ped_ids = metadata_df.query("AgeCategory == 'Pediatric'").ModelID.tolist()
adult_ids = metadata_df.query("AgeCategory == 'Adult'").ModelID.tolist()
male_ids = metadata_df.query("Sex == 'Male'").ModelID.tolist()
female_ids = metadata_df.query("Sex == 'Female'").ModelID.tolist()
ped_male_ids = metadata_df.query("AgeCategory == 'Pediatric'").query("Sex == 'Male'").ModelID.tolist()
adult_male_ids = metadata_df.query("AgeCategory == 'Adult'").query("Sex == 'Male'").ModelID.tolist()
ped_female_ids = metadata_df.query("AgeCategory == 'Pediatric'").query("Sex == 'Female'").ModelID.tolist()
adult_female_ids = metadata_df.query("AgeCategory == 'Adult'").query("Sex == 'Female'").ModelID.tolist()


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
t_test_adult_vs_ped['comparison'] = 'Adult vs Pediatric'
t_test_adult_vs_ped['latent_feature'] = t_test_adult_vs_ped.index + 1
print(t_test_adult_vs_ped.shape)

t_test_adult_vs_ped.head(5)


# In[6]:


# t tests comparing male vs female for each latent dimension

t_test_male_vs_female = ttest_ind(male_latent_df_float, female_latent_df_float)
t_test_male_vs_female = pd.DataFrame(t_test_male_vs_female).T
t_test_male_vs_female.columns = ["t_stat", "p_value"]
t_test_male_vs_female['comparison'] = 'Male vs Female'
t_test_male_vs_female['latent_feature'] = t_test_male_vs_female.index + 1
print(t_test_male_vs_female.shape)

t_test_male_vs_female.head(5)


# In[7]:


# t tests comparing adult male vs ped male for each latent dimension

t_test_adult_male_vs_ped_male = ttest_ind(ped_male_latent_df_float, adult_male_latent_df_float)
t_test_adult_male_vs_ped_male = pd.DataFrame(t_test_adult_male_vs_ped_male).T
t_test_adult_male_vs_ped_male.columns = ["t_stat", "p_value"]
t_test_adult_male_vs_ped_male['comparison'] = 'Adult Male vs Pediatric Male'
t_test_adult_male_vs_ped_male['latent_feature'] = t_test_adult_male_vs_ped_male.index + 1
print(t_test_adult_male_vs_ped_male.shape)

t_test_adult_male_vs_ped_male.head(5)


# In[8]:


# t tests comparing adult female vs ped female for each latent dimension

t_test_adult_female_vs_ped_female = ttest_ind(ped_female_latent_df_float, adult_female_latent_df_float)
t_test_adult_female_vs_ped_female = pd.DataFrame(t_test_adult_female_vs_ped_female).T
t_test_adult_female_vs_ped_female.columns = ["t_stat", "p_value"]
t_test_adult_female_vs_ped_female['comparison'] = 'Adult Female vs Pediatric Female'
t_test_adult_female_vs_ped_female['latent_feature'] = t_test_adult_female_vs_ped_female.index + 1
print(t_test_adult_female_vs_ped_female.shape)

t_test_adult_female_vs_ped_female.head(5)


# In[9]:


# t tests comparing ped male vs ped female for each latent dimension

t_test_ped_male_vs_ped_female = ttest_ind(ped_female_latent_df_float, ped_male_latent_df_float)
t_test_ped_male_vs_ped_female = pd.DataFrame(t_test_ped_male_vs_ped_female).T
t_test_ped_male_vs_ped_female.columns = ["t_stat", "p_value"]
t_test_ped_male_vs_ped_female['comparison'] = 'Pediatric Male vs Pediatric Female'
t_test_ped_male_vs_ped_female['latent_feature'] = t_test_ped_male_vs_ped_female.index + 1
print(t_test_ped_male_vs_ped_female.shape)

t_test_ped_male_vs_ped_female.head(5)


# In[10]:


# t tests comparing adult male vs adult female for each latent dimension

t_test_adult_male_vs_adult_female = ttest_ind(adult_female_latent_df_float, adult_male_latent_df_float)
t_test_adult_male_vs_adult_female = pd.DataFrame(t_test_adult_male_vs_adult_female).T
t_test_adult_male_vs_adult_female.columns = ["t_stat", "p_value"]
t_test_adult_male_vs_adult_female['comparison'] = 'Adult Male vs Adult Female'
t_test_adult_male_vs_adult_female['latent_feature'] = t_test_adult_male_vs_adult_female.index + 1
print(t_test_adult_male_vs_adult_female.shape)

t_test_adult_male_vs_adult_female.head(5)


# In[11]:


# Combining and saving t test results
t_test_results_df = pd.concat([
    t_test_adult_vs_ped, 
    t_test_male_vs_female, 
    t_test_adult_male_vs_ped_male, 
    t_test_adult_female_vs_ped_female, 
    t_test_ped_male_vs_ped_female, 
    t_test_adult_male_vs_adult_female
]).reset_index(drop=True)
t_test_results_dir = pathlib.Path("./results/t_test_results.tsv")
t_test_results_df.to_parquet(t_test_results_dir)

# sort to show most significant p-values
t_test_results_df.sort_values(by='p_value', ascending = True)


# In[12]:


# ANOVA Testing
f_statistic, p_value = f_oneway(adult_male_latent_df_float, ped_male_latent_df_float, adult_female_latent_df_float, ped_female_latent_df_float)
anova_df = pd.DataFrame({'f_stat': f_statistic.tolist(), 'p_value': p_value.tolist()})
anova_df['latent_feature'] = anova_df.index + 1
anova_df

