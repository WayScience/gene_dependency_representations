#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import random

sns.set_theme(color_codes=True)
sys.path.insert(0, ".././0.data-download/scripts/")

from matplotlib.pyplot import figure, gcf
from sklearn.decomposition import PCA
from tensorflow import keras

import blitzgsea as blitz
import urllib.request

# Download the gene set library here: https://github.com/MaayanLab/blitzgsea


# In[2]:


random.seed(18)
print(random.random())


# In[3]:


# list available gene set libraries in Enrichr
blitz.enrichr.print_libraries()


# In[4]:


# use enrichr submodule to retrieve gene set library
# these libraries are finicky to work with--they usually work the first time but then may stop working. You may need to remove the library from your computer and trying to reimport it to work again.
library = blitz.enrichr.get_library("GO_Biological_Process_2017")


# In[5]:


# load the weight matrix 
gene_weight_dir = pathlib.Path("../2.train-VAE//results/weight_matrix_gsea.csv")
signature = pd.read_csv(gene_weight_dir)
print(signature.shape)
signature.head()


# In[6]:


all_GSEA_results = []
all_signatures = []
#negative_control = []
for col in signature.iloc[:,1:50].columns:
    df = signature.iloc[:,[0,int(col)]]
    result = blitz.gsea(df, library)
    all_GSEA_results.append(result.assign(z_dim=f"z_{col}"))
    all_signatures.append(df)
    #add line here to create a negative control--this will be a random scramble of the gene scores
all_GSEA_results


# In[7]:


# stack up all of the results to be analyzed
all_GSEA_results= pd.concat(all_GSEA_results)


# In[8]:


# sort by what you want to evaluate
all_GSEA_results.sort_values(by='es', ascending=False)


# In[9]:


plt.scatter(x=all_GSEA_results['es'],y=all_GSEA_results['pval'].apply(lambda x:-np.log10(x)),s=1)


# In[10]:


# plot the enrichment results and save to png--this code needs some work done to it in order to work!
fig = blitz.plot.running_sum(signature, "DIABETIC CARDIOMYOPATHY", library, result=result, compact=False)
fig.savefig("running_sum.png", bbox_inches='tight')

fig_compact = blitz.plot.running_sum(signature, "PATHOGENIC ESCHERICHIA COLI INFECTION", library, result=result, compact=True)
fig_compact.savefig("running_sum_compact.png", bbox_inches='tight')

fig_table = blitz.plot.top_table(signature, library, result, n=15)
fig_table.savefig("top_table.png", bbox_inches='tight')

