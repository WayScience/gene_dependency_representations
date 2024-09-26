#!/usr/bin/env python
# coding: utf-8

# ## Describe PRISM data

# In[1]:


import pathlib
import sys

import pandas as pd

sys.path.append("../")
from utils import load_utils


# In[5]:


# Load PRISM data
top_dir = "../5.drug-dependency"
data_dir = "data"

prism_df, prism_cell_df, prism_trt_df = load_utils.load_prism(
    top_dir=top_dir,
    data_dir=data_dir,
    secondary_screen=False,
    load_cell_info=True,
    load_treatment_info=True,
)

print(prism_df.shape)
prism_df.head(3)


# In[6]:


print(prism_cell_df.shape)
prism_cell_df.head(3)


# In[7]:


print(prism_trt_df.shape)
prism_trt_df.head(3)


# In[8]:


# How many tissues
prism_cell_df.loc[:, "tissue"].value_counts()


# In[9]:


# How many unique treatments
print(prism_trt_df.name.nunique())

# How many MOAs
print(prism_trt_df.moa.nunique())
prism_trt_df.moa.value_counts().head(20)


# ## Secondary screen data

# In[10]:


prism_df, prism_cell_df, prism_trt_df = load_utils.load_prism(
    top_dir=top_dir,
    data_dir=data_dir,
    secondary_screen=True,
    load_cell_info=True,
    load_treatment_info=True,
)

print(prism_df.shape)
prism_df.head(3)


# In[11]:


print(prism_cell_df.shape)
prism_cell_df.head(3)


# In[12]:


print(prism_trt_df.shape)
prism_trt_df.head(3)


# In[13]:


# How many tissues in secondary screen
prism_cell_df.loc[:, "tissue"].value_counts()


# In[14]:


# How many unique treatments
print(prism_trt_df.name.nunique())

# How many MOAs
print(prism_trt_df.moa.nunique())
prism_trt_df.moa.value_counts().head(20)


# In[15]:


# How many doses per compound
prism_trt_df.groupby("name").dose.count().value_counts()

