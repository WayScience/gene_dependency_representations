#!/usr/bin/env python
# coding: utf-8

# # Download TARGET Pan Cancer Data for Compression
# The notebook downloads gene expression and clinical data from the TARGET project. The data is downloaded from UCSC Xena.
# The data is in log2(FPKM) RSEM transformed

# In[1]:


import os
from urllib.request import urlretrieve
import random
import pandas as pd
from sklearn.model_selection import train_test_split


# In[2]:


url = 'https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/'
name = 'target_RSEM_gene_fpkm.gz'

path = os.path.join('data', name)


# In[3]:


urlretrieve('{}{}'.format(url, name), path)


# In[4]:


get_ipython().system(" sha256sum 'data/target_RSEM_gene_fpkm.gz'")


# In[5]:


name = 'gencode.v23.annotation.gene.probemap'

path = os.path.join('data', name)


# In[6]:


urlretrieve('{}probeMap/{}'.format(url, name), path)


# In[7]:


get_ipython().system(" sha256sum 'data/gencode.v23.annotation.gene.probemap'")


# In[8]:


name = 'TARGET_phenotype.gz'

path = os.path.join('data', name)


# In[9]:


urlretrieve('{}{}'.format(url, name), path)


# In[10]:


get_ipython().system(" sha256sum 'data/TARGET_phenotype.gz'")


# # Process TARGET PanCancer Data
# Retrive the downloaded expression data, update gene identifiers to entrez, and curate sample IDs. The script will also identify a balanced hold-out test set to compare projection performance into learned latent spaces across algorithms. 

# In[11]:


random.seed(1234)


# # Read Phenotype Information

# In[12]:


path = os.path.join('data', 'TARGET_phenotype.gz')
pheno_df = pd.read_table(path)

print(pheno_df.shape)
pheno_df.head(3)


# # Read Entrez ID Curation Information
# Load curated gene names from versioned resource. See https://github.com/cognoma/genes for more details

# In[13]:


# Commit from https://github.com/cognoma/genes
genes_commit = 'ad9631bb4e77e2cdc5413b0d77cb8f7e93fc5bee'


# In[14]:


url = 'https://raw.githubusercontent.com/cognoma/genes/{}/data/genes.tsv'.format(genes_commit)
gene_df = pd.read_table(url)

# Only consider protein-coding genes
gene_df = (
   gene_df.query("gene_type == 'protein-coding'")
)


print(gene_df.shape)
gene_df.head(2)


# In[15]:


# Load gene updater - old to new Entrez gene identifiers
url = 'https://raw.githubusercontent.com/cognoma/genes/{}/data/updater.tsv'.format(genes_commit)
updater_df = pd.read_table(url)
old_to_new_entrez = dict(zip(updater_df.old_entrez_gene_id,
                            updater_df.new_entrez_gene_id))


# # Read Probe Mapping Info 

# In[16]:


path = os.path.join('data', 'gencode.v23.annotation.gene.probemap')
probe_map_df = pd.read_table(path)


# Inner merge gene df to get ensembl to entrez mapping
probe_map_df = probe_map_df.merge(gene_df, how='inner', left_on='gene', right_on='symbol')


# Mapping to rename gene expression index
ensembl_to_entrez = dict(zip(probe_map_df.id, probe_map_df.entrez_gene_id))


print(probe_map_df.shape)
probe_map_df.head(3)


# # Read Gene Expression Data

# In[17]:


file = os.path.join('data', 'target_RSEM_gene_fpkm.gz')
expr_df = pd.read_table(file, index_col=0)


print(expr_df.shape)
expr_df.head(2)


# # Process gene expression matrix 

# In[18]:


expr_df = (expr_df
   .dropna(axis='rows')
   .reindex(probe_map_df.id)
   .rename(index=ensembl_to_entrez)
   .rename(index=old_to_new_entrez)
   .groupby(level=0).mean()
   .transpose()
   .sort_index(axis='rows')
   .sort_index(axis='columns')
)


expr_df.index.rename('sample_id', inplace=True)


print(expr_df.shape)
expr_df.head(2)


# # Stratify Balanced Training and Testing Sets in TARGET Gene Expression
# Output training and testing gene expression datasets 

# In[19]:


strat = pheno_df.set_index('sample_id').reindex(expr_df.index).primary_disease_code


# In[20]:


cancertype_count_df = (
   pd.DataFrame(strat.value_counts())
   .reset_index()
   .rename({'index': 'cancertype', 'primary_disease_code': 'n ='}, axis='columns')
)


file = os.path.join('data', 'target_sample_counts.tsv')
cancertype_count_df.to_csv(file, sep='\t', index=False)


cancertype_count_df


# In[21]:


train_df, test_df = train_test_split(expr_df,
                                    test_size=0.1,
                                    random_state=123,
                                    stratify=strat)


# In[22]:


print(train_df.shape)
test_df.shape


# In[23]:


train_file = os.path.join('data', 'train_target_expression_matrix_processed.tsv.gz')
train_df.to_csv(train_file, sep='\t', compression='gzip', float_format='%.3g')


# In[24]:


test_file = os.path.join('data', 'test_target_expression_matrix_processed.tsv.gz')
test_df.to_csv(test_file, sep='\t', compression='gzip', float_format='%.3g')


# # Sort genes based on median absolute deviation and output to file

# In[25]:


# Determine most variably expressed genes and subset
mad_genes_df = pd.DataFrame(train_df.mad(axis=0).sort_values(ascending=False)).reset_index()
mad_genes_df.columns = ['gene_id', 'median_absolute_deviation']


file = os.path.join('data', 'target_mad_genes.tsv')
mad_genes_df.to_csv(file, sep='\t', index=False)

