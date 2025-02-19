#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import sys

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import DataLoader, TensorDataset


sys.path.insert(0, "../utils/")
from data_loader import load_model_data
from pinwheels import compute_and_plot_latent_scores

sys.path.insert(0,"../6.RNAseq/utils")
from optimizing_utils import extract_latent_dims

sys.path.append("../5.drug-dependency")
from utils import load_utils


# In[2]:


data_directory = pathlib.Path("../0.data-download/data").resolve(strict=True)
dependency_file = pathlib.Path(f"{data_directory}/CRISPRGeneEffect.parquet").resolve(strict=True)
gene_dict_file = pathlib.Path(f"{data_directory}/CRISPR_gene_dictionary.parquet").resolve(strict=True)


# In[3]:


# Load metadata
metadata_df_dir = pathlib.Path("../0.data-download/data/metadata_df.parquet").resolve(strict=True)
metadata = pd.read_parquet(metadata_df_dir)
print(metadata.shape)

#Load dependency data
dependency_df, gene_dict_df = load_model_data(dependency_file, gene_dict_file)
dependency_df.head()

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Apply the scaler to the numeric columns
dependency_df[dependency_df.select_dtypes(include=['float64', 'int']).columns] = scaler.fit_transform(
    dependency_df.select_dtypes(include=['float64', 'int'])
)


# In[4]:


train_and_test_subbed_dir = pathlib.Path("../0.data-download/data/train_and_test_subbed.parquet").resolve(strict=True)
train_and_test_subbed = pd.read_parquet(train_and_test_subbed_dir)

# Convert DataFrame to NumPy and then Tensor
train_test_array = train_and_test_subbed.to_numpy()
train_test_tensor = torch.tensor(train_test_array, dtype=torch.float32)

#Create TensorDataset and DataLoader
tensor_dataset = TensorDataset(train_test_tensor)
train_and_test_subbed_loader = DataLoader(tensor_dataset, batch_size=32, shuffle=False)


# In[5]:


gsea_dir = pathlib.Path("../4.gene-expression-signatures/gsea_results/combined_z_matrix_gsea_results.parquet").resolve(strict=True)
gsea_df = pd.read_parquet(gsea_dir)


# In[6]:


# Define the location of the saved models and output directory for results
model_save_dir = pathlib.Path("../4.gene-expression-signatures/saved_models")
output_dir = pathlib.Path("results")
output_dir.mkdir(parents=True, exist_ok=True)


# In[7]:


# Placeholder for storing all latent representations
latent_dfs = []

# Iterate over all saved model files
for model_file in model_save_dir.glob("*.joblib"):
    model_file_name = model_file.stem
    try:
        parts = model_file_name.split("_")
        model_name = parts[0]
        num_components = int(parts[3])  # Example: Extract number of components
        init = int(parts[7])  # Extract initialization value
        seed = int(parts[9])  # Extract seed value
    except (IndexError, ValueError):
        print(f"Skipping file {model_file} due to unexpected filename format.")
        continue

    # Load the model
    print(f"Loading model from {model_file}")
    try:
        model = joblib.load(model_file)
    except Exception as e:
        print(f"Failed to load model from {model_file}: {e}")
        continue
    

    # Extract latent dimensions
    latent_df = extract_latent_dims(model_name, model, dependency_df, train_and_test_subbed_loader, metadata)

    # Add metadata to the latent dataframe
    latent_df["model"] = model_name
    latent_df["latent_dim_total"] = num_components
    latent_df["init"] = init
    latent_df["seed"] = seed

    # Move metadata columns to the front
    metadata_columns = ["model", "latent_dim_total", "init", "seed"]
    latent_df = latent_df.loc[:, ~latent_df.columns.duplicated()]
    latent_df.columns = latent_df.columns.astype(str)
    latent_df = latent_df[metadata_columns + [col for col in latent_df.columns if col not in metadata_columns]]
    latent_dfs.append(latent_df)

# Combine all latent representations into one dataframe
combined_latent_df = pd.concat(latent_dfs, ignore_index=True)


# In[8]:


# Combine all latent representations into one dataframe
combined_latent_df = pd.concat(latent_dfs, ignore_index=True)


# In[9]:


drug_dir = pathlib.Path("../5.drug-dependency/results/combined_latent_drug_correlations.parquet")
drug_df = pd.read_parquet(drug_dir)

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


# In[10]:


# Merge drug_df with prism_trt_df to replace drug IDs with drug names
drug_df = drug_df.merge(prism_trt_df[['column_name', 'name']], 
                        left_on='drug', 
                        right_on='column_name', 
                        how='left')

# Drop the redundant column_name column
drug_df = drug_df.drop(columns=['column_name'])

# Display the updated dataframe
drug_df.head()


# In[11]:


# Loop through each unique ModelID to process and plot
gsea_max = gsea_df.loc[gsea_df.groupby("reactome_pathway")["gsea_es_score"].idxmax()]
for model_id in combined_latent_df['ModelID'].unique():
    compute_and_plot_latent_scores(model_id, combined_latent_df, gsea_max, "reactome_pathway", "gsea_es_score", "Multi-Gene Dependency")


# In[12]:


# Loop through each unique ModelID to process and plot
drug_max = drug_df.loc[drug_df.groupby("name")["pearson_correlation"].idxmax()]
for model_id in combined_latent_df['ModelID'].unique():
    compute_and_plot_latent_scores(model_id, combined_latent_df, drug_max, "name", "pearson_correlation", "Drug")

