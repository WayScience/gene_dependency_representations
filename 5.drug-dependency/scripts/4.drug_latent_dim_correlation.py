#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
import pathlib
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np 

script_directory = pathlib.Path("../2.train-VAE/utils/").resolve()
sys.path.insert(0, str(script_directory))
from betavae import  extract_latent_dimensions
from betatcvae import tc_extract_latent_dimensions
from vanillavae import vvae_extract_latent_dimensions

from utils import load_utils

sys.path.insert(0, "../utils/")
from data_loader import load_model_data


# In[2]:


# Function to extract latent dimensions for PCA, ICA, and NMF models
def sklearn_extract_latent_dimensions(model, model_name, dependency_df):
    # Extract components from the model (latent vectors)
    original_feature_names = model.feature_names_in_
    reordered_df = dependency_df[original_feature_names]
    # Transform models into pca space
    latent_df = pd.DataFrame(
        model.transform(reordered_df)
    )

    # Recode column space and add back model IDs
    latent_df.columns = [f"z_{x}" for x in range(0, latent_df.shape[1])]
    latent_df = pd.concat([dependency_df.loc[:, "ModelID"], latent_df], axis="columns")
    
    return latent_df


def perform_correlation(latent_df, drug_df, model_name, num_components):
    """
    Perform Pearson correlation between latent dimensions and drug dependency scores.
    
    Parameters:
    latent_df (pd.DataFrame): Dataframe containing latent dimensions (e.g., PCA/ICA/NMF).
    drug_df (pd.DataFrame): Dataframe containing drug dependency scores with `ModelID`.
    model_name (str): Name of the model used to extract the latent dimensions (PCA, ICA, NMF).
    num_components (int): Number of latent dimensions/components in the model.
    
    Returns:
    pd.DataFrame: Correlation results between latent dimensions and drug scores.
    """
    correlation_results = []
    print("latent_df columns:", latent_df.columns)
    print("drug_df columns:", drug_df.columns)
    latent_df.set_index('ModelID', inplace=True)

    # Align both dataframes based on the ModelID
    common_model_ids = latent_df.index.intersection(drug_df.index)
    # Filter both dataframes to keep only common ModelIDs
    latent_df_filtered = latent_df.loc[common_model_ids]
    prism_df_filtered = drug_df.loc[common_model_ids]
    # Check the variance of each latent dimension and drug response column
    latent_variance = latent_df_filtered.var()
    prism_variance = prism_df_filtered.var()

    # Filter out constant columns (variance == 0)
    latent_df_filtered = latent_df_filtered.loc[:, latent_variance != 0]
    prism_df_filtered = prism_df_filtered.loc[:, prism_variance != 0]

    # Loop over each latent dimension and calculate correlation with each drug
    for latent_col in latent_df_filtered.columns:
        for drug_col in prism_df_filtered.columns:
            latent_values = latent_df_filtered[latent_col]
            drug_values = prism_df_filtered[drug_col]

            # Check if either column is constant
            if latent_values.nunique() <= 1 or drug_values.nunique() <= 1:
                corr = np.nan
            else:
                # Drop missing values for both columns
                valid_data = pd.concat([latent_values, drug_values], axis=1).dropna()
                latent_values_valid = valid_data[latent_col]
                drug_values_valid = valid_data[drug_col]

                if len(latent_values_valid) > 1 and len(drug_values_valid) > 1:
                    # Calculate Pearson correlation
                    corr, p_value = pearsonr(latent_values_valid, drug_values_valid)
                else:
                    corr = np.nan
                    print("nan")
        
                # Store the results
                result_row = {
                    "z": int(latent_col.replace("z_", "")),
                    "full_model_z": num_components,
                    "model": str(model_name),
                    "drug": str(drug_col),
                    "pearson_correlation": corr,
                    "p_value": p_value,
                    "shuffled": False
                }
                correlation_results.append(result_row)
    
    # Convert results into a dataframe
    correlation_results_df = pd.DataFrame(correlation_results)
    return correlation_results_df


# In[3]:


data_directory = pathlib.Path("../0.data-download/data").resolve()
dependency_file = pathlib.Path(f"{data_directory}/CRISPRGeneEffect.parquet").resolve()
gene_dict_file = pathlib.Path(f"{data_directory}/CRISPR_gene_dictionary.parquet").resolve()


# In[4]:


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

# Reset the index and name it ModelID
prism_df.reset_index(inplace=True)
prism_df.rename(columns={'index': 'ModelID'}, inplace=True)
prism_df.set_index('ModelID', inplace=True)
prism_df.head()


# In[5]:


# Load metadata
metadata_df_dir = pathlib.Path("../0.data-download/data/metadata_df.parquet")
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


# In[6]:


train_and_test_subbed_dir = pathlib.Path("../0.data-download/data/train_and_test_subbed.parquet")
train_and_test_subbed = pd.read_parquet(train_and_test_subbed_dir)


# Convert DataFrame to NumPy and then Tensor
train_test_array = train_and_test_subbed.to_numpy()
train_test_tensor = torch.tensor(train_test_array, dtype=torch.float32)

#Create TensorDataset and DataLoader
tensor_dataset = TensorDataset(train_test_tensor)
train_and_test_subbed_loader = DataLoader(tensor_dataset, batch_size=32, shuffle=False)


# In[7]:


# Define the location of the saved models and output directory for correlation results
model_save_dir = pathlib.Path("../4.gene_expression_signatures/saved_models")
output_dir = pathlib.Path("results")
output_dir.mkdir(parents=True, exist_ok=True)

# Latent dimensions and model names to iterate over
latent_dims = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200]
model_names = ["pca", "ica", "nmf", "vanillavae", "betavae", "betatcvae"]

# File to store the combined correlation results
final_output_file = output_dir / "combined_latent_drug_correlations.parquet"
try:
    combined_results_df = pd.read_parquet(final_output_file)
    print(f"Loaded existing results from {final_output_file}")
except FileNotFoundError:
    # If the file doesn't exist, initialize an empty DataFrame
    combined_results_df = pd.DataFrame()
    print(f"No existing file found. Initialized empty DataFrame.")

for num_components in latent_dims:
    for model_name in model_names:
        # Check if this model and latent dimension have already been processed
        if not combined_results_df.empty:
            if ((combined_results_df['model'] == model_name) & 
                (combined_results_df['full_model_z'] == num_components)).any():
                print(f"Skipping {model_name} with {num_components} dimensions as it is already processed.")
                continue  # Skip to the next iteration if this combination is already present
        
        # Load the saved model
        model_filename = model_save_dir / f"{model_name}_{num_components}_components_model.joblib"
        if model_filename.exists():
            print(f"Loading model from {model_filename}")
            model = joblib.load(model_filename)
            
            if model_name in ["pca", "ica", "nmf"]:
                # Extract the latent dimensions for these models
                latent_df = sklearn_extract_latent_dimensions(model, model_name, dependency_df)
            elif model_name == "betavae":
                latent_df = extract_latent_dimensions(model, train_and_test_subbed_loader, metadata)
            elif model_name == "betatcvae":
                latent_df = tc_extract_latent_dimensions(model, train_and_test_subbed_loader, metadata)
            elif model_name == "vanillavae":
                latent_df = vvae_extract_latent_dimensions(model, train_and_test_subbed_loader, metadata)

            latent_df.columns = ['ModelID'] + [f'z_{col}' if isinstance(col, int) else col for col in latent_df.columns[1:]]
            # Perform Pearson correlation between latent dimensions and drug data
            correlation_results_df = perform_correlation(latent_df, prism_df, model_name, num_components)
            
            # Concatenate results to the combined dataframe
            combined_results_df = pd.concat([combined_results_df, correlation_results_df], ignore_index=True)
        else:
            print(f"Model file {model_filename} not found. Skipping.")

# Save the combined results to a parquet file
combined_results_df.to_parquet(final_output_file)
print(f"Saved combined results to {final_output_file}")


# In[8]:


# Save the combined results to a parquet file
combined_results_df.to_parquet(final_output_file)
print(f"Saved combined results to {final_output_file}")

#Save as CSV for R 
csv_output_file = output_dir / "combined_latent_drug_correlations.csv"
combined_results_df.to_csv(csv_output_file, index=False)


# In[9]:


combined_results_df.sort_values(by='pearson_correlation', key=abs, ascending = False).head(50)

