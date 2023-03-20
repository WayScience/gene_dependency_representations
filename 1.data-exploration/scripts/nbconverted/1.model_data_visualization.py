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
adult_threshold = 15
pediatric_liquid_tumors = ["Leukemia", "Lymphoma"]


# In[3]:


# Set i/o paths and files
data_dir = pathlib.Path("../0.data-download/data")
fig_dir = pathlib.Path("figures")
fig_dir.mkdir(exist_ok=True)

model_input_file = pathlib.Path(f"{data_dir}/Model.csv")
crispr_input_file = pathlib.Path(f"{data_dir}/CRISPRGeneDependency.csv")

model_ouput_age_cleaned_file = pathlib.Path(f"{data_dir}/Model_age_column_cleaned.csv")
cancer_type_output_figure = pathlib.Path(f"{fig_dir}/sample_cancer_types_bar_chart.png")
age_category_output_figure = pathlib.Path(f"{fig_dir}/age_categories_bar_chart.png")
age_distribution_output_figure = pathlib.Path(
    f"{fig_dir}/sample_age_distribution_plot.png"
)
sex_output_figure = pathlib.Path(f"{fig_dir}/sample_gender_bar_chart.png")
pediatric_cancer_type_output_figure = pathlib.Path(
    f"{fig_dir}/pediatric_sample_cancer_types_bar_chart.png"
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
n_samples_model = len(model_df["DepMap_ID"].unique())
print(f"Number of samples documented in Model.csv: {n_samples_model} \n")

# How many samples from CRISPRGeneDependency.csv?
n_samples_gene = len(gene_dependency_df["DepMap_ID"].unique())
print(f"Number of samples measured in CRISPRGeneDependency.csv: {n_samples_gene} \n")

# Identify which samples are included in both Model.csv and CRISPRGeneDependency.csv
sample_overlap = list(set(model_df["DepMap_ID"]) & set(gene_dependency_df["DepMap_ID"]))

# count the number of samples that overlap in both data sets
print(f"Samples measured in both: {len(sample_overlap)} \n")

# How many different types of cancer?
n_cancer_types = model_df.query("DepMap_ID in @sample_overlap")[
    "primary_disease"
].nunique()
print(f"Number of Cancer Types: \n {n_cancer_types} \n")


# In[7]:


# Visualize cancer type distribution
cancer_types_bar = (
    gg.ggplot(model_df, gg.aes(x="primary_disease"))
    + gg.geom_bar()
    + gg.coord_flip()
    + gg.ggtitle("Distribution of cancer types")
    + gg.theme_bw()
)

cancer_types_bar.save(cancer_type_output_figure, dpi=500)

cancer_types_bar


# ## Clean age variable

# In[8]:


age_categories = []
age_distribution = []

# Loop through each age entry to clean it
for age_entry in model_df.age.tolist():
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


# In[9]:


# Add columns age_categories & age_distribution
model_df = model_df.assign(
    age_categories=age_categories, age_distribution=age_distribution
)

# Output file
model_df.to_csv(model_ouput_age_cleaned_file, index=False)

print(model_df.shape)
model_df.head(3)


# ## Visualize age categories and distribution

# In[10]:


age_categories_bar = (
    gg.ggplot(model_df, gg.aes(x="age_categories"))
    + gg.geom_bar()
    + gg.ggtitle(
        f"Age categories of derived cell lines (Pediatric =< {adult_threshold})"
    )
    + gg.theme_bw()
)

age_categories_bar.save(age_category_output_figure, dpi=500)

age_categories_bar


# In[11]:


age_distribution_plot = (
    gg.ggplot(model_df, gg.aes(x="age_distribution"))
    + gg.geom_density()
    + gg.geom_vline(xintercept=adult_threshold, linetype="dashed", color="red")
    + gg.ggtitle(
        f"Age distribution of derived cell lines (Pediatric =< {adult_threshold})"
    )
    + gg.theme_bw()
)

age_distribution_plot.save(age_distribution_output_figure, dpi=500)

age_distribution_plot


# In[12]:


pd.DataFrame(age_categories).loc[:, 0].value_counts()


# In[13]:


gendersamp_plot = (
    gg.ggplot(model_df, gg.aes(x="sex"))
    + gg.geom_bar()
    + gg.ggtitle(f"Sex categories of derived cell lines")
    + gg.theme_bw()
)

gendersamp_plot.save(sex_output_figure)

gendersamp_plot


# ## What cell lines are pediatric cancer?

# In[14]:


pediatric_model_df = (
    model_df.query("age_categories == 'Pediatric'")
    .query("DepMap_ID in @sample_overlap")
    .reset_index(drop=True)
)

print(pediatric_model_df.shape)
pediatric_model_df.head(3)


# In[15]:


# What are the neuroblastoma models?
pediatric_model_df.query(
    "Cellosaurus_NCIt_disease == 'Neuroblastoma'"
).stripped_cell_line_name


# In[16]:


# What is the distribution of pediatric tumor types
pediatric_cancer_counts = pediatric_model_df.primary_disease.value_counts()
pediatric_cancer_counts


# In[17]:


pediatric_cancer_counts.reset_index()


# In[18]:


# Visualize cancer type distribution
ped_cancer_types_bar = (
    gg.ggplot(
        pediatric_cancer_counts.reset_index(), gg.aes(x="index", y="primary_disease")
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
print("The number of pediatric solid tumors:")
print(
    pediatric_model_df.query("primary_disease not in @pediatric_liquid_tumors")
    .primary_disease.value_counts()
    .sum()
)
print("The number of pediatric liquid tumors:")
print(
    pediatric_model_df.query("primary_disease in @pediatric_liquid_tumors")
    .primary_disease.value_counts()
    .sum()
)

