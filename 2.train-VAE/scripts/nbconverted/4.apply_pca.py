#!/usr/bin/env python
# coding: utf-8

# ## Apply PCA to gene dependencies
# 
# Principal component analysis (PCA) is a commonly-used dimensionality reduction method that finds components that explain orthogonal variation in the data in a descending fashion.
# 
# We apply it to GeneEffect scores here and save latent representations (PCA components) for downstream comparative analyses (to compare with BetaVAE).

# In[1]:


import sys
import pathlib
import pandas as pd
import plotnine as gg
from sklearn.decomposition import PCA

sys.path.insert(0, "../0.data-download/scripts/")
from data_loader import load_train_test_data


# In[2]:


data_directory = pathlib.Path("../0.data-download/data")
dependency_file = pathlib.Path(f"{data_directory}/CRISPRGeneEffect.csv")
gene_dict_file = pathlib.Path(f"{data_directory}/CRISPR_gene_dictionary.tsv")

output_dir = pathlib.Path("results")
pca_output_file = pathlib.Path(f"{output_dir}/pca_latent.csv.gz")
output_pca_weights_file = pathlib.Path(f"{output_dir}/PCA_weight_matrix_gsea.csv")


# In[3]:


pca_components = 50


# In[4]:


# Load gene dependency data
dependency_df = pd.read_csv(dependency_file)

print(dependency_df.shape)
dependency_df.head(3)


# In[5]:


# Load gene dictionary (with QC columns)
gene_dict_df = (
    pd.read_csv(gene_dict_file, sep="\t")
    .query("qc_pass")
    .reset_index(drop=True)
)
gene_dict_df.entrez_id = gene_dict_df.entrez_id.astype(str)

print(gene_dict_df.shape)
gene_dict_df.head(3)


# ## Subset input data to common gene sets

# In[6]:


# Recode column names to entrez ids
entrez_genes = [x[1].strip(")").strip() for x in dependency_df.iloc[:, 1:].columns.str.split("(")]

entrez_intersection = list(
    set(gene_dict_df.entrez_id).intersection(set(entrez_genes))
)

print(len(entrez_intersection))

gene_dict_df = gene_dict_df.set_index("entrez_id").reindex(entrez_intersection)
gene_dict_df.head(3)


# In[7]:


# Subset dependencies to the genes that passed qc
dependency_df.columns = ["ModelID"] + entrez_genes

dependency_df = dependency_df.loc[:, ["ModelID"] + gene_dict_df.index.tolist()]
dependency_df.columns = ["ModelID"] + gene_dict_df.symbol_id.tolist()

dependency_df = dependency_df.dropna(axis="columns")

print(dependency_df.shape)
dependency_df.head()


# # Perform PCA

# In[8]:


pca = PCA(n_components=pca_components)
pca.fit(dependency_df.drop(columns=["ModelID"]))


# In[9]:


# Output explained variance and quickly visualize
explained_var = pd.DataFrame(pca.explained_variance_ratio_, columns=["explained_var"]).reset_index()

(
    gg.ggplot(explained_var, gg.aes(x="index", y="explained_var"))
    + gg.geom_bar(stat="identity")
)


# In[10]:


# Transform models into pca space
dependency_df_transformed = pd.DataFrame(
    pca.transform(dependency_df.drop(columns=["ModelID"]))
)

# Recode column space and add back model IDs
dependency_df_transformed.columns = [f"PCA_{x}" for x in range(0, dependency_df_transformed.shape[1])]
dependency_df_transformed = pd.concat([dependency_df.loc[:, "ModelID"], dependency_df_transformed], axis="columns")

dependency_df_transformed.to_csv(pca_output_file, sep=",", index=False)

print(dependency_df_transformed.shape)
dependency_df_transformed.head(3)


# In[11]:


# Obtain weights, which can be used in GSEA
pca_weights = pd.DataFrame(pca.components_, columns=dependency_df.drop(columns=["ModelID"]).columns.tolist()).transpose()
pca_weights.columns = [f"PCA_{x}" for x in range(0, pca_weights.shape[1])]

pca_weights = pca_weights.reset_index().rename(columns={"index": "genes"})

pca_weights.to_csv(output_pca_weights_file, index=False)

print(pca_weights.shape)
pca_weights.head(3)

