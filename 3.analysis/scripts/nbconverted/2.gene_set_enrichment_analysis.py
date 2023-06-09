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
neg_GSEA_results = []
negative_control = []

for col in signature.iloc[:,1:101].columns:
    df = signature.iloc[:,[0,int(col)]]
    result = blitz.gsea(df, library)
    all_GSEA_results.append(result.assign(z_dim=f"z_{col}"))
    all_signatures.append(df)
    neg_df = df.copy()
    neg_df[col] = neg_df[col].sample(frac=1).reset_index(drop=True)
    neg_result = blitz.gsea(neg_df, library)
    neg_GSEA_results.append(neg_result.assign(z_dim=f"z_{col}"))
    negative_control.append(neg_df)

all_GSEA_results
neg_GSEA_results


# In[7]:


# stack up all of the results to be analyzed
all_GSEA_results= pd.concat(all_GSEA_results)
neg_GSEA_results = pd.concat(neg_GSEA_results)


# In[8]:


# sort by what you want to evaluate
all_GSEA_results['rank'] = (-np.log10(all_GSEA_results.pval))*(all_GSEA_results.es)
all_GSEA_results.sort_values(by='pval', ascending = True)

neg_GSEA_results['rank'] = (-np.log10(neg_GSEA_results.pval))*(neg_GSEA_results.es)
neg_GSEA_results.sort_values(by='es', ascending = False)


# In[9]:


plt.figure()
plt.scatter(x=all_GSEA_results['es'],y=all_GSEA_results['pval'].apply(lambda x:-np.log10(x)),s=10)
plt.xlabel('log2 Fold Change (ES)')
plt.ylabel('-log10(pvalue)')
plt.title('Gene Set Enrichment Analysis')

plt.figure()
plt.scatter(x=neg_GSEA_results['es'],y=neg_GSEA_results['pval'].apply(lambda x:-np.log10(x)), s=10)
plt.xlabel('log2 Fold Change (ES)')
plt.ylabel('-log10(pvalue)')
plt.title('Control Gene Set Enrichment Analysis')


# In[10]:


# Using VAE generated data

z = 0
for df in all_signatures:
    
    z = z+1

    fig = blitz.plot.running_sum(df, "regulation of transcription from RNA polymerase II promoter in response to hypoxia (GO:0061418)", library, result=result, compact=False)
    fig.savefig("running_sum_z_" + str(z) + ".png", bbox_inches='tight')

    fig_compact = blitz.plot.running_sum(df, "regulation of transcription from RNA polymerase II promoter in response to hypoxia (GO:0061418)", library, result=result, compact=True)
    fig_compact.savefig("running_sum_compact_z_" + str(z) + ".png", bbox_inches='tight')

    fig_table = blitz.plot.top_table(df, library, result, n=15)
    fig_table.savefig("top_table_z_" + str(z) + ".png", bbox_inches='tight')

# Using negative control

z = 0
for df in negative_control:
    
    z = z+1

    fig = blitz.plot.running_sum(df, "regulation of transcription from RNA polymerase II promoter in response to hypoxia (GO:0061418)", library, result=result, compact=False)
    fig.savefig("running_sum_neg_z_" + str(z) + ".png", bbox_inches='tight')

    fig_compact = blitz.plot.running_sum(df, "regulation of transcription from RNA polymerase II promoter in response to hypoxia (GO:0061418)", library, result=result, compact=True)
    fig_compact.savefig("running_sum_compact_neg_z_" + str(z) + ".png", bbox_inches='tight')

    fig_table = blitz.plot.top_table(df, library, result, n=15)
    fig_table.savefig("top_table_neg_z_" + str(z) + ".png", bbox_inches='tigh t')

