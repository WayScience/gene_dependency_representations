#!/usr/bin/env python
# coding: utf-8

# # Download TARGET Pan Cancer Data for Compression
# The notebook downloads gene expression and clinical data from the TARGET project. The data is downloaded from UCSC Xena.
# The data is in log2(FPKM) RSEM transformed

# In[1]:


import pathlib
from pathlib import Path
import hashlib
from urllib.request import urlretrieve
import random
import pandas as pd
from sklearn.model_selection import train_test_split


# In[2]:


def download_and_checksum(url_base, file_info_dict, file_url_paths):
    """
    Downloads files from the specified base URL and computes the SHA-256 checksum for each file.

    Parameters:
    ----------
    url_base : str
        The base URL from which the files will be downloaded.

    file_info_dict : dict
        A dictionary where the keys are file names and the values are the local file paths where 
        the files will be saved.

    file_url_paths : dict
        A dictionary mapping each file name to the corresponding sub-path (if any) that should 
        be appended to the base URL before downloading the file.
    """
    for name, path in file_info_dict.items():
        # Determine the correct URL with or without sub-path
        file_url = f"{url_base}{file_url_paths[name]}{name}"
        
        # Download the file
        urlretrieve(file_url, path)

        # Compute checksum
        md5_hash = hashlib.md5()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                md5_hash.update(byte_block)

        print(f"{name} checksum: {md5_hash.hexdigest()}")


# In[3]:


# Base URL
url_base = 'https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/'
data_dir = pathlib.Path('data').resolve()


# In[4]:


# Dictionary for file names and their paths
file_info_dict = {
    'target_RSEM_gene_fpkm.gz': Path('data/target_RSEM_gene_fpkm.gz'),
    'TARGET_phenotype.gz': Path('data/TARGET_phenotype.gz'),
    'gencode.v23.annotation.gene.probemap': Path('data/gencode.v23.annotation.gene.probemap')
}

file_url_paths = {
    "target_RSEM_gene_fpkm.gz": "",
    "gencode.v23.annotation.gene.probemap": "probeMap/",
    "TARGET_phenotype.gz": ""
}


# In[5]:


# Create the 'data' directory
Path('data').mkdir(exist_ok=True)


# In[6]:


# Call the function
download_and_checksum(url_base, file_info_dict, file_url_paths)


# # Process TARGET PanCancer Data
# Retrive the downloaded expression data, update gene identifiers to entrez, and curate sample IDs. The script will also identify a balanced hold-out test set to compare projection performance into learned latent spaces across algorithms. 

# In[7]:


random.seed(0)


# # Read Phenotype Information

# In[8]:


pheno_file = data_dir / 'TARGET_phenotype.parquet'
if not pheno_file.is_file():
    pheno_df = pd.read_table(data_dir / 'TARGET_phenotype.gz')
    pheno_df.to_parquet(pheno_file, index=False)
else:
    pheno_df = pd.read_parquet(pheno_file)

print(pheno_df.shape)
pheno_df.head(3)


# # Read Entrez ID Curation Information
# Load curated gene names from versioned resource. See https://github.com/cognoma/genes for more details

# In[9]:


# Commit from https://github.com/cognoma/genes
genes_commit = 'ad9631bb4e77e2cdc5413b0d77cb8f7e93fc5bee'


# In[10]:


url = 'https://raw.githubusercontent.com/cognoma/genes/{}/data/genes.tsv'.format(genes_commit)
gene_df = pd.read_table(url)

# Only consider protein-coding genes
gene_df = (
   gene_df.query("gene_type == 'protein-coding'")
)


print(gene_df.shape)
gene_df.head(2)


# In[11]:


# Load gene updater - old to new Entrez gene identifiers
url = 'https://raw.githubusercontent.com/cognoma/genes/{}/data/updater.tsv'.format(genes_commit)
updater_df = pd.read_table(url)
old_to_new_entrez = dict(zip(updater_df.old_entrez_gene_id,
                            updater_df.new_entrez_gene_id))


# # Read Probe Mapping Info

# In[12]:


probe_map_file = data_dir / 'gencode.v23.annotation.gene.probemap.parquet'
if not probe_map_file.is_file():
    probe_map_df = pd.read_table(data_dir / 'gencode.v23.annotation.gene.probemap')
    probe_map_df.to_parquet(probe_map_file, index=False)
else:
    probe_map_df = pd.read_parquet(probe_map_file)

# Inner merge gene df to get ensembl to entrez mapping
probe_map_df = probe_map_df.merge(gene_df, how='inner', left_on='gene', right_on='symbol')
ensembl_to_entrez = dict(zip(probe_map_df.id, probe_map_df.entrez_gene_id))

print(probe_map_df.shape)
probe_map_df.head(3)


# # Read Gene Expression Data

# In[13]:


expr_file = data_dir / 'target_RSEM_gene_fpkm.parquet'
if not expr_file.is_file():
    expr_df = pd.read_table(data_dir / 'target_RSEM_gene_fpkm.gz', index_col=0)
    expr_df.to_parquet(expr_file)
else:
    expr_df = pd.read_parquet(expr_file)

print(expr_df.shape)
expr_df.head(2)


# # Process gene expression matrix 

# In[14]:


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

# In[15]:


strat = pheno_df.set_index('sample_id').reindex(expr_df.index).primary_disease_code


# In[16]:


cancertype_count_df = pd.DataFrame(strat.value_counts()).reset_index().rename({'index': 'cancertype', 'primary_disease_code': 'n ='}, axis='columns')
sample_counts_file = data_dir / 'target_sample_counts.parquet'
cancertype_count_df.to_parquet(sample_counts_file)

cancertype_count_df


# In[17]:


train_df, test_df = train_test_split(expr_df,
                                    test_size=0.2,
                                    random_state=0,
                                    stratify=strat)


# In[18]:


print(train_df.shape)
test_df.shape


# In[19]:


train_file = data_dir / 'train_target_expression_matrix_processed.parquet'
train_df.to_parquet(train_file)


# In[20]:


test_file = data_dir / 'test_target_expression_matrix_processed.parquet'
test_df.to_parquet(test_file)


# # Sort genes based on median absolute deviation and output to file

# In[ ]:


mad_genes_df = pd.DataFrame(train_df.mad(axis=0).sort_values(ascending=False)).reset_index()
mad_genes_df.columns = ['gene_id', 'median_absolute_deviation']
mad_genes_file = data_dir / 'target_mad_genes.parquet'
mad_genes_df.to_parquet(mad_genes_file)

