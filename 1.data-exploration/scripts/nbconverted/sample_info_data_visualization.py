#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages that will be utilized
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import pathlib
import seaborn as sns
import plotnine as gg
from plotnine import *
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# assign the desired file a variable using pathlib.Path command
input_file = pathlib.Path("../0.data-download/data/sample_info.csv")

# set the data frame to be the desired .csv file that is read by pandas(pd) using the pd.read_csv(desired file read as a previously defined variable)
df = pd.read_csv(input_file)

# assign the desired file a variable using pathlib.Path command
input_file2 = pathlib.Path("../0.data-download/data/CRISPR_gene_dependency.csv")

# set the data frame to be the desired .csv file that is read by pandas(pd) using the pd.read_csv(desired file read as a previously defined variable)
df2 = pd.read_csv(input_file2)


# In[3]:


# print the parameters of the read file
print(df.shape)
df.head(1)


# In[4]:


# print the parameters of the read file
print(df2.shape)
df2.head(1)


# In[5]:


# sample_info.csv visualization
# how many samples from sample_info.csv?
n_samples = len(df["DepMap_ID"].unique())
print(f"Number of Samples Documented in sample_info.csv: {n_samples} \n")

# sample_info.csv visualization
# how many samples from CRISPR_gene_dependency.csv?
n_samples2 = len(df2["DepMap_ID"].unique())
print(f"Number of Samples Included in CRISPR_gene_dependency.csv: {n_samples2} \n")

# how many different ages were sampled from? 
all_ages = df["age"].unique()
print(f"Ages sampled from: \n {all_ages} \n")


# how many different types of cancer?
all_cancers = df["primary_disease"].unique()
print(f"All Cancer Types: \n {all_cancers} \n")

# create a bar chart that shows the number of types of cancer sampled 
data = df
cancer_types_bar = (
    gg.ggplot(data, gg.aes(x="primary_disease")) + gg.geom_bar() + gg.theme(axis_text_x =element_text(angle = 90))
    )
print(cancer_types_bar)

# identify which samples are included in both sample_info.csv and CRISPR_gene_dependency.csv
similar_samples = list(set(df["DepMap_ID"]) & set(df2["DepMap_ID"]))

# count the number of samples that overlap in both data sets 
sample_overlap = len(similar_samples)
print(f"number of sample overlaps between sample_info.csv and CRISPR_gene_dependency.csv: {sample_overlap} \n")


# In[6]:


age_vector_to_clean = df.loc[:, "age"].tolist()

age_categories = []
age_distribution = []

adult_threshold = 18

# Loop through each age entry to clean it
for age_entry in age_vector_to_clean:
    try:
        # If the age is an integer, apply appropriate category
        if int(age_entry) >= adult_threshold:
            age_categories.append("Adult")
        else:
            age_categories.append("Pediatric")
        
        # If the age is an integer, apply appropriate continuous measure
        age_distribution.append(int(age_entry))

    except ValueError:
        # If conversion fails, categorize appropriately
        if pd.notnull(age_entry):
            age_categories.append(age_entry)
        else:
            age_categories.append("Missing")
        
        age_distribution.append(np.nan)


# In[7]:


# New dataframe containing two new columns age_categories & age_distribution
df_age_visual = (
    df.assign(
        age_categories=age_categories,
        age_distribution=age_distribution
    )
)

df_age_visual.head()


# In[8]:


(
    gg.ggplot(df_age_visual, gg.aes(x="age_categories"))
    + gg.geom_bar()
)


# In[9]:


(
    gg.ggplot(df_age_visual, gg.aes(x="age_distribution"))
    + gg.geom_density()
    + gg.geom_vline(xintercept=adult_threshold, linetype="dashed", color="red")
)


# In[10]:


pd.DataFrame(age_categories).loc[:, 0].value_counts()


# In[11]:


gendersamp = df
gendersamp_plot = (
    gg.ggplot(gendersamp, gg.aes(x="sex")) + gg.geom_bar()
)
print(gendersamp_plot)

