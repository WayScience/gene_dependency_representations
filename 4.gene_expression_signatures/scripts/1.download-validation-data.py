#!/usr/bin/env python
# coding: utf-8

# # Download and Process Neuroblastoma RNAseq Data

# Code from: https://github.com/greenelab/BioBombe/blob/master/10.gene-expression-signatures/0.download-validation-data.ipynb
# 
# We are downloading the dataset associated with Harenza et al. 2017. The data profiles RNAseq data from 39 commonly used neuroblastoma (NBL) cell lines.
# 
# We are interested in the MYCN amplification status of these cell lines. We will test if the MYCN amplification score learned through the BioBombe signature approach applied to TARGET data generalizes to this cell line dataset.
# 
# MYCN Amplification refers to the number of copies of the MYCN gene. MYCN amplification is used as a biomarker for poor prognosis in neuroblastoma patients (Huang and Weiss 2013).

# In[1]:


import requests
import pathlib
import pandas as pd
from urllib.request import urlretrieve

from sklearn import preprocessing


# In[2]:


url = "https://ndownloader.figshare.com/files/14138792"
name = "2019-01-22-CellLineSTAR-fpkm-2pass_matrix.txt"
data_dir = pathlib.Path("data").resolve()
path = data_dir / name


# In[3]:


data_dir.mkdir(parents=True, exist_ok=True)


# In[4]:


urlretrieve(url, path)


# In[5]:


get_ipython().system(' md5sum "data/2019-01-22-CellLineSTAR-fpkm-2pass_matrix.txt"')


# # Download Phenotype Data

# In[6]:


url = "https://www.nature.com/articles/sdata201733/tables/4"
name = "nbl_cellline_phenotype.txt"
path = data_dir / name


# In[7]:


if not path.is_file():
    html = requests.get(url).content

    pheno_df = pd.read_html(html)[0]
    pheno_df['Cell Line'] = pheno_df['Cell Line'].str.replace("-", "")

    pheno_df.to_csv(path, sep='\t', index=False)

else:
    pheno_df = pd.read_parquet(path)

pheno_df.head()


# In[8]:


get_ipython().system(' md5sum "data/nbl_cellline_phenotype.txt"')


# # Process RNAseq Data

# In[9]:


raw_file = data_dir / "2019-01-22-CellLineSTAR-fpkm-2pass_matrix.txt"
raw_df = pd.read_table(raw_file, sep='\t')
raw_df.head()


# # Update Gene Names

# In[10]:


# Load curated gene names from versioned resource 
commit = '721204091a96e55de6dcad165d6d8265e67e2a48'
url = 'https://raw.githubusercontent.com/cognoma/genes/{}/data/genes.tsv'.format(commit)
gene_df = pd.read_table(url)

# Only consider protein-coding genes
gene_df = (
    gene_df.query("gene_type == 'protein-coding'")
)

symbol_to_entrez = dict(zip(gene_df.symbol,
                            gene_df.entrez_gene_id))


# In[11]:


# Add alternative symbols to entrez mapping dictionary
gene_df = gene_df.dropna(axis='rows', subset=['synonyms'])
gene_df.synonyms = gene_df.synonyms.str.split('|')

all_syn = (
    gene_df.apply(lambda x: pd.Series(x.synonyms), axis=1)
    .stack()
    .reset_index(level=1, drop=True)
)

# Name the synonym series and join with rest of genes
all_syn.name = 'all_synonyms'
gene_with_syn_df = gene_df.join(all_syn)

# Remove rows that have redundant symbols in all_synonyms
gene_with_syn_df = (
    gene_with_syn_df
    
    # Drop synonyms that are duplicated - can't be sure of mapping
    .drop_duplicates(['all_synonyms'], keep=False)

    # Drop rows in which the symbol appears in the list of synonyms
    .query('symbol not in all_synonyms')
)


# In[12]:


# Create a synonym to entrez mapping and add to dictionary
synonym_to_entrez = dict(zip(gene_with_syn_df.all_synonyms,
                             gene_with_syn_df.entrez_gene_id))

symbol_to_entrez.update(synonym_to_entrez)


# In[13]:


# Load gene updater
url = 'https://raw.githubusercontent.com/cognoma/genes/{}/data/updater.tsv'.format(commit)
updater_df = pd.read_table(url)
old_to_new_entrez = dict(zip(updater_df.old_entrez_gene_id,
                             updater_df.new_entrez_gene_id))


# In[14]:


gene_map = raw_df.GeneID.replace(symbol_to_entrez)
gene_map = gene_map.replace(old_to_new_entrez)


# In[15]:


raw_df.index = gene_map
raw_df.index.name = 'entrez_gene_id'
raw_df = raw_df.drop(['GeneID'], axis='columns')
raw_df = raw_df.loc[raw_df.index.isin(symbol_to_entrez.values()), :]

print(raw_df.shape)
raw_df.head()


# # Scale Data and Output

# In[16]:


#MinMax consistent with BetaVAE scaling
raw_scaled_df = preprocessing.MinMaxScaler().fit_transform(raw_df.transpose())
raw_scaled_df = (
    pd.DataFrame(raw_scaled_df,
                 columns=raw_df.index,
                 index=raw_df.columns)
    .sort_index(axis='columns')
    .sort_index(axis='rows')
)
raw_scaled_df.columns = raw_scaled_df.columns.astype(str)
raw_scaled_df = raw_scaled_df.loc[:, ~raw_scaled_df.columns.duplicated(keep='first')]

raw_scaled_df.head()


# In[17]:


# In[17]
processed_file = data_dir / 'nbl_celllines_processed_matrix.parquet'
raw_scaled_df.to_parquet(processed_file)
