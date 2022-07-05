#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sys
import pathlib
import pandas as pd 
import numpy as np

sys.path.insert(0, "../0.data-download/scripts")
from data_loader import load_data

data_directory = "../0.data-download/data/"
dfs = load_data(data_directory, adult_or_pediatric = "all")


# In[8]:


dfs[0] 


# In[9]:


dfs[1]

