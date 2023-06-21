#!/usr/bin/env python
# coding: utf-8

# ## Construct gene dictionary
# 
# The column names of the gene dependency files are of the format "gene symbol (entrez id)".
# 
# Additionally, the `depmap_gene_meta.tsv` contains genes that passed an initial QC (see Pan et al. 2022).
# 
# This notebooke will create a four column matrix that separates symbol from entrez id, retains the original column name, and includes a column of if the gene passed QC.
# 
# Example:
# 
# | entrez_id | symbol_id | dependency_column | qc_pass |
# | :-------: | :-------: | :---------------: | :-----: |
# | 1 | A1BG |A1BG (1)| True |
# | 29974 | A1CF | A1CF (29974) | True |
# |	2 	| A2M | A2M (2) | False |
# 
# *Note, the example qc_pass column above is an example and may not reflect truth.*

# In[1]:


import pathlib
import pandas as pd


# In[2]:


base_dir = "data/"

dependency_file = pathlib.Path(f"{base_dir}/CRISPRGeneEffect.csv")
qc_gene_file = pathlib.Path(f"{base_dir}/depmap_gene_meta.tsv")

output_gene_dict_file = pathlib.Path(f"{base_dir}/CRISPR_gene_dictionary.tsv")


# In[3]:


# Load gene dependency data
dependency_df = pd.read_csv(dependency_file, index_col=0)

print(dependency_df.shape)
dependency_df.head()


# In[4]:


# Load depmap metadata
gene_meta_df = pd.read_csv(qc_gene_file, sep="\t")
gene_meta_df.entrezgene = gene_meta_df.entrezgene.astype(str)

print(gene_meta_df.shape)
gene_meta_df.head(3)


# ## Obtain the intersection of the genes
# 
# Comparing the current DepMap release and the previous gene set qc (19Q2 depmap release)

# In[5]:


# Recode column names to entrez ids from dependency file
entrez_genes = [x[1].strip(")").strip() for x in dependency_df.iloc[:, 1:].columns.str.split("(")]

# Obtain intersection of entrez gene ids
entrez_intersection = list(
    set(gene_meta_df.entrezgene).intersection(set(entrez_genes))
)

print(f"The number of overlapping entrez gene ids: {len(entrez_intersection)}")

# Subset the gene metadata file to only those in common, which are ones that passed qc
gene_passed_qc_df = (
    gene_meta_df
    .query("entrezgene in @entrez_intersection")
    .set_index("entrezgene")
    .reindex(entrez_intersection)
    .reset_index()
    .loc[:, ["entrezgene", "Name", "symbol"]]
)

gene_passed_qc_df.head()


# ## Convert the initial dependency map input file to three parts
# 
# 1. Entrez ID
# 2. Symbol
# 3. The full column name

# In[6]:


entrez_genes = [x[1].strip(")").strip() for x in dependency_df.columns.str.split("(")]
symbol_genes = [x[0].strip() for x in dependency_df.columns.str.split("(")]

gene_dictionary_df = pd.DataFrame(
    [
        entrez_genes,
        symbol_genes,
        dependency_df.columns.tolist()
    ]
).transpose()

gene_dictionary_df.columns = ["entrez_id", "symbol_id", "dependency_column"]

print(gene_dictionary_df.shape)
gene_dictionary_df.head()


# ## Create the QC column

# In[7]:


gene_dictionary_qc_df = (
    # Merge gene dictionary with qc dataframe
    gene_dictionary_df.merge(
        gene_passed_qc_df,
        left_on="entrez_id",
        right_on="entrezgene",
        how="left"  # Note the left merge, to retain all genes from gene_dictionary_df
    )
    # Select only certain columns
    .loc[:, ["entrez_id", "symbol_id", "dependency_column", "entrezgene"]]
    # Values that are missing indicate genes that did not pass QC
    .fillna(value={"entrezgene": False})
    # Rename the column to be clearly defined
    .rename(columns={"entrezgene": "qc_pass"})
)

# Convert genes with entrez entries to those that indicate QC pass
gene_dictionary_qc_df.loc[gene_dictionary_qc_df.qc_pass != False, "qc_pass"] = True

# Output file
gene_dictionary_df.to_csv(output_gene_dict_file, index=False, sep="\t")

print(gene_dictionary_qc_df.qc_pass.value_counts())
print(gene_dictionary_qc_df.shape)

gene_dictionary_qc_df.head(3)

