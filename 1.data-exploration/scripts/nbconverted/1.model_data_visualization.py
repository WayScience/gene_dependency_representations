#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import seaborn as sns
from plotnine import *

warnings.filterwarnings("ignore")


# In[2]:


# Set constants
liquid_tumors = ["Leukemia", "Lymphoma"]


# In[3]:


# Set i/o paths and files
data_dir = pathlib.Path("../0.data-download/data")
fig_dir = pathlib.Path("figures")
fig_dir.mkdir(exist_ok=True)

model_input_file = pathlib.Path(f"{data_dir}/Model.csv")
crispr_input_file = pathlib.Path(f"{data_dir}/CRISPRGeneDependency.csv")

model_output_age_cleaned_file = pathlib.Path(f"{data_dir}/Model_age_column_cleaned.csv")
ped_model_output_age_cleaned_file = pathlib.Path(f"{data_dir}/Pediatric_model_age_column_cleaned.csv")
adult_model_output_age_cleaned_file = pathlib.Path(f"{data_dir}/Adult_model_age_column_cleaned.csv")


cancer_type_output_figure = pathlib.Path(f"{fig_dir}/sample_cancer_types_bar_chart.png")
age_category_output_figure = pathlib.Path(f"{fig_dir}/age_categories_bar_chart.png")
age_distribution_output_figure = pathlib.Path(
    f"{fig_dir}/sample_age_distribution_plot.png"
)
sex_output_figure = pathlib.Path(f"{fig_dir}/sample_gender_bar_chart.png")
pediatric_cancer_type_output_figure = pathlib.Path(
    f"{fig_dir}/pediatric_sample_cancer_types_bar_chart.png"
)
adult_cancer_type_output_figure = pathlib.Path(
    f"{fig_dir}/adult_sample_cancer_types_bar_chart.png"
)


# In[4]:


# Load model data
model_df = pd.read_csv(model_input_file)

print(model_df.shape)
model_df.head(3)


# In[5]:


# Load dependency data
gene_dependency_df = pd.read_csv(crispr_input_file)

print(gene_dependency_df.shape)
gene_dependency_df.head(3)


# ## Describe input data

# In[6]:


# Model.csv visualization
# How many samples from Model.csv?
n_samples_model = len(model_df["ModelID"].unique())
print(f"Number of samples documented in Model.csv: {n_samples_model} \n")

# How many samples from CRISPRGeneDependency.csv?
n_samples_gene = len(gene_dependency_df["ModelID"].unique())
print(f"Number of samples measured in CRISPRGeneDependency.csv: {n_samples_gene} \n")

# Identify which samples are included in both Model.csv and CRISPRGeneDependency.csv
sample_overlap = list(set(model_df["ModelID"]) & set(gene_dependency_df["ModelID"]))

# count the number of samples that overlap in both data sets
print(f"Samples measured in both: {len(sample_overlap)} \n")

# How many different types of cancer?
n_cancer_types = model_df.query("ModelID in @sample_overlap")[
    "OncotreePrimaryDisease"
].nunique()
print(f"Number of Cancer Types: \n {n_cancer_types} \n")


# In[7]:


# Visualize cancer type distribution
cancer_types_bar = (
    gg.ggplot(model_df, gg.aes(x="OncotreePrimaryDisease"))
    + gg.geom_bar()
    + gg.coord_flip()
    + gg.ggtitle("Distribution of cancer types")
    + gg.theme_bw()
)

cancer_types_bar.save(cancer_type_output_figure, dpi=500)

cancer_types_bar


# ## Clean age variable

# In[8]:


age_distribution = []

# Loop through each age entry to clean it
for age_entry in model_df.Age.tolist():
    try: 
        # If the age is an integer or float, apply appropriate continuous measure
        age_distribution.append(int(age_entry))
    except:
        # If conversion fails, categorize appropriately
        age_distribution.append(np.nan)


# In[9]:


# Add columns age_categories & age_distribution
model_df = model_df.assign(age_distribution=age_distribution)

# Output file
model_df.to_csv(model_output_age_cleaned_file, index=False)

print(model_df.shape)
model_df.head(3)


# ## Visualize age categories and distribution

# In[10]:


age_categories_bar = (
    gg.ggplot(model_df, gg.aes(x="AgeCategory"))
    + gg.geom_bar()
    + gg.ggtitle(
        f"Age categories of derived cell lines"
    )
    + gg.theme_bw()
)

age_categories_bar.save(age_category_output_figure, dpi=500)

age_categories_bar


# In[11]:


age_distribution_plot = (
    gg.ggplot(model_df, gg.aes(x="age_distribution"))
    + gg.geom_density()
    + gg.geom_vline(xintercept=18, linetype="dashed", color="red")
    + gg.ggtitle(
        f"Age distribution of derived cell lines"
    )
    + gg.theme_bw()
)

age_distribution_plot.save(age_distribution_output_figure, dpi=500)

age_distribution_plot


# In[12]:


model_df['AgeCategory'].value_counts()


# In[13]:


gendersamp_plot = (
    gg.ggplot(model_df, gg.aes(x="Sex"))
    + gg.geom_bar()
    + gg.ggtitle(f"Sex categories of derived cell lines")
    + gg.theme_bw()
)

gendersamp_plot.save(sex_output_figure)

gendersamp_plot


# ## What cell lines are pediatric cancer?

# In[14]:


pediatric_model_df = (
    model_df.query("AgeCategory == 'Pediatric'")
    .query("ModelID in @sample_overlap")
    .reset_index(drop=True)
)

pediatric_model_df.to_csv(ped_model_output_age_cleaned_file, index=False)

print(pediatric_model_df.shape)
pediatric_model_df.head(3)


# In[15]:


# What are the neuroblastoma models?
pediatric_model_df.query(
    "OncotreeSubtype == 'Neuroblastoma'"
).StrippedCellLineName


# In[16]:


# What is the distribution of pediatric tumor types
pediatric_cancer_counts = pediatric_model_df.OncotreePrimaryDisease.value_counts()
pediatric_cancer_counts


# In[17]:


pediatric_cancer_counts.reset_index()


# In[18]:


# Visualize pediatric cancer type distribution
ped_cancer_types_bar = (
    gg.ggplot(
        pediatric_cancer_counts.reset_index(), gg.aes(x="index", y="OncotreePrimaryDisease")
    )
    + gg.geom_bar(stat="identity")
    + gg.coord_flip()
    + gg.ggtitle("Distribution of pediatric cancer types")
    + gg.ylab("count")
    + gg.xlab("cancer type")
    + gg.theme_bw()
)

ped_cancer_types_bar.save(pediatric_cancer_type_output_figure, dpi=500)

ped_cancer_types_bar


# In[19]:


# Pediatric solid vs liquid tumors
cancer_types = pediatric_model_df['OncotreePrimaryDisease'].tolist()

liquid = []
non_liquid = []

for type in cancer_types:
    if liquid_tumors[0] in type or liquid_tumors[1] in type:
        liquid.append(type)
    else:
        non_liquid.append(type)

print("The number of pediatric solid tumors:")
print(len(non_liquid))

print("The number of pediatric liquid tumors:")
print(len(liquid))


# ## What cell lines are adult cancer?

# In[20]:


adult_model_df = (
    model_df.query("AgeCategory == 'Adult'")
    .query("ModelID in @sample_overlap")
    .reset_index(drop=True)
)

adult_model_df.to_csv(adult_model_output_age_cleaned_file, index=False)

print(adult_model_df.shape)
adult_model_df.head(3)


# In[21]:


# What is the distribution of adult tumor types
adult_cancer_counts = adult_model_df.OncotreePrimaryDisease.value_counts()
adult_cancer_counts


# In[22]:


adult_cancer_counts.reset_index()


# In[23]:


# Visualize adult cancer type distribution
adult_cancer_types_bar = (
    gg.ggplot(
        adult_cancer_counts.reset_index(), gg.aes(x="index", y="OncotreePrimaryDisease")
    )
    + gg.geom_bar(stat="identity")
    + gg.coord_flip()
    + gg.ggtitle("Distribution of adult cancer types")
    + gg.ylab("count")
    + gg.xlab("cancer type")
    + gg.theme_bw()
)

adult_cancer_types_bar.save(adult_cancer_type_output_figure, dpi=500)

adult_cancer_types_bar


# In[24]:


# Adult solid vs liquid tumors
cancer_types = adult_model_df['OncotreePrimaryDisease'].tolist()

liquid = []
non_liquid = []

for type in cancer_types:
    if liquid_tumors[0] in type or liquid_tumors[1] in type:
        liquid.append(type)
    else:
        non_liquid.append(type)

print("The number of adult solid tumors:")
print(len(non_liquid))

print("The number of adult liquid tumors:")
print(len(liquid))

