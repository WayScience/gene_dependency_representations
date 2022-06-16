#!/usr/bin/env python
# coding: utf-8

# In[8]:


import urllib.request
CRISPR_gene_dependency = "https://ndownloader.figshare.com/files/34990033"
sample_info = "https://ndownloader.figshare.com/files/35020903"
urllib.request.urlretrieve(CRISPR_gene_dependency, '/home/markw/gene_dependency_representations/0.data-download/data/CRISPR_gene_dependency.csv')
urllib.request.urlretrieve(sample_info, '/home/markw/gene_dependency_representations/0.data-download/data/sample_info.csv')

