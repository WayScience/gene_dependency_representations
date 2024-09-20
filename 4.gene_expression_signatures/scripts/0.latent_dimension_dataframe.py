#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib 
import optuna
import pandas as pd
import numpy as np
import random
import torch
import sys
import blitzgsea as blitz
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import ttest_ind

script_directory = pathlib.Path("../2.train-VAE/utils/").resolve()
sys.path.insert(0, str(script_directory))
from betavae import BetaVAE, train_vae, weights
from betatcvae import BetaTCVAE, train_tc_vae, tc_weights
from vanillavae import VanillaVAE, train_vvae, vanilla_weights
from optimize_utils import get_optimize_args, objective, get_optimizer
from optimize_utils_tcvae import get_optimize_args_tc, objective_tc, get_optimizer_tc
from optimize_utils_vvae import get_optimize_args_vvae, objective_vvae, get_optimizer_vvae

script_directory = pathlib.Path("../utils/").resolve()
sys.path.insert(0, str(script_directory))
from data_loader import load_train_test_data, load_model_data


# In[2]:


# Load command line arguments
args = get_optimize_args()
tc_args = get_optimize_args_tc()
vvae_args = get_optimize_args_vvae()

# Load data
data_directory = pathlib.Path("../0.data-download/data").resolve()

train_df, test_df, val_df, load_gene_stats = load_train_test_data(
    data_directory, train_or_test="all", load_gene_stats=True, zero_one_normalize=True
)
train_data = pd.DataFrame(train_df)

dependency_file = pathlib.Path(f"{data_directory}/CRISPRGeneEffect.parquet").resolve()
gene_dict_file = pathlib.Path(f"{data_directory}/CRISPR_gene_dictionary.parquet").resolve()
dependency_df, gene_dict_df= load_model_data(dependency_file, gene_dict_file)
gene_dict_df = pd.DataFrame(gene_dict_df)


# In[3]:


#Load weight data for VAEs
data_directory = pathlib.Path("../0.data-download/data").resolve()
weight_df = load_train_test_data(
    data_directory, train_or_test="train"
)

gene_list_passed_qc = gene_dict_df.loc[
    gene_dict_df["qc_pass"], "dependency_column"
].tolist()

weight_data = weight_df.filter(gene_list_passed_qc, axis=1)
weight_data.head()


# In[4]:


# Convert dataframes to tensors
train_tensor = torch.tensor(train_df, dtype=torch.float32)
test_tensor = torch.tensor(test_df, dtype=torch.float32)
val_tensor = torch.tensor(val_df, dtype=torch.float32)


# In[5]:


output_dir = pathlib.Path("./results").resolve()
output_dir.mkdir(parents=True, exist_ok=True)


# In[6]:


# Function to perform dimensionality reduction and extract z matrices for sklearn models
def extract_z_matrix_sklearn(model, data, model_name, num_components, output_dir):
    if model_name == "nmf":
        min_value = data.min().min()
        if min_value < 0:
            data = data - min_value
    # Fit the model and transform the data
    z_matrix = model.fit_transform(data)
    
    # Create a DataFrame and save it
    z_matrix_df = pd.DataFrame(z_matrix, columns=[f'{model_name}_{i}' for i in range(num_components)])
    z_matrix_df.insert(0, 'ModelID', train_data.index) 
    
    return z_matrix_df


# In[7]:


def extract_weights(model, model_name, H=None):
    if model_name in ["pca", "ica", "nmf"]:
        # Transpose PCA components and format columns
        weights_df = pd.DataFrame(model.components_, columns=dependency_df.drop(columns=["ModelID"]).columns.tolist()).transpose()
        weights_df.columns = [f"{x}" for x in range(0, weights_df.shape[1])]
        
    # Reset index to rename 'index' to 'genes'
    weights_df = weights_df.reset_index().rename(columns={"index": "genes"})
    
    return weights_df


def perform_gsea(weights_df, model_name, num_components):
    library = blitz.enrichr.get_library("Reactome_2022")
    seed = random.random()
    gsea_results = []
    # Define cutoff values
    lfc_cutoff = 0.584
    fdr_cutoff = 0.25
    # Perform GSEA for each component column in weights_df
    for col in weights_df.columns[1:]:  # Skip 'genes' column
        gene_signature = weights_df[['genes', col]]
        
        if gene_signature.shape[0] > 0:
            try:
                # Perform GSEA using the gene signature (weights)
                gsea_result = blitz.gsea(gene_signature, library, seed=seed)
                gsea_result = gsea_result.reset_index()
                for _, pathway_result in gsea_result.iterrows():
                    result_row = {
                        "z": int(col),
                        "full_model_z": num_components,
                        "model": str(model_name),
                        "reactome pathway": str(pathway_result['Term']),
                        "gsea es score": pathway_result['es'],
                        "nes score": pathway_result['nes'],
                        "p value": pathway_result['pval'],
                        "shuffled": False
                    }
                    gsea_results.append(result_row)
                    
            except ZeroDivisionError:
                print(f"Skipping GSEA for {col} due to zero division error.")
    
    # Convert GSEA results into a DataFrame and save to a Parquet file
    gsea_results_df = pd.DataFrame(gsea_results)
    # Filter for significant results
    significant_gsea_df = gsea_results_df[
            (gsea_results_df['gsea es score'].abs() > lfc_cutoff) &
            (gsea_results_df['p value'] < fdr_cutoff)
        ]
    print(significant_gsea_df)
    
    return significant_gsea_df


# Initialize a list to store the final results
final_gsea_results = []

# GSEA settings
latent_dims = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150]
model_names = ["pca", "ica", "nmf", "vanillavae", "betavae", "betatcvae"]

# Define the output file path
final_output_file = pathlib.Path(output_dir) / "combined_z_matrix_gsea_results.parquet"

# Try to load the existing combined results DataFrame if it exists
try:
    combined_results_df = pd.read_parquet(final_output_file)
    print(f"Loaded existing combined results from {final_output_file}")
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
        z_matrix_df = None
        
        if model_name in ["pca", "ica", "nmf"]:
            # Sklearn models (PCA, ICA, NMF)
            if model_name == "pca":
                model = PCA(n_components=num_components)
            elif model_name == "ica":
                model = FastICA(n_components=num_components)
            elif model_name == "nmf":
                model = NMF(n_components=num_components, init='nndsvd', max_iter=1000, random_state=0)
            
            model.fit(train_data)
            H = model.transform(train_data) if model_name == "nmf" else None
            weight_matrix_df = extract_weights(model, model_name, H)
                
        elif model_name == "betavae":
            # Optuna optimization for BetaVAE
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective(trial, train_tensor, train_tensor, train_data, latent_dim=num_components), n_trials=50)
            
            # Train the best BetaVAE model and extract z matrix
            best_trial = study.best_trial
            model = BetaVAE(input_dim=train_data.shape[1], latent_dim=num_components, beta=best_trial.params['beta'])
            train_loader = DataLoader(TensorDataset(train_tensor), batch_size=best_trial.params['batch_size'], shuffle=True)
            optimizer = get_optimizer(best_trial.params['optimizer_type'], model.parameters(), best_trial.params['learning_rate'])
            train_vae(model, train_loader, optimizer, best_trial.params['epochs'])
            
            weight_matrix_df = weights(model, weight_data)
            weight_matrix_df.rename(columns={0: 'genes'}, inplace=True)
        elif model_name == "betatcvae":
            # Optuna optimization for BetaTCVAE
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective_tc(trial, train_tensor, train_tensor, train_data, latent_dim=num_components), n_trials=50)
            # Train the best BetaTCVAE model and extract z matrix
            best_trial = study.best_trial
            model = BetaTCVAE(input_dim=train_data.shape[1], latent_dim=num_components, beta=best_trial.params['beta'])
            train_loader = DataLoader(TensorDataset(train_tensor), batch_size=best_trial.params['batch_size'], shuffle=True)
            optimizer = get_optimizer_tc(best_trial.params['optimizer_type'], model.parameters(), best_trial.params['learning_rate'])
            train_tc_vae(model, train_loader, optimizer, best_trial.params['epochs'])
            
            # Extract weight matrix
            weight_matrix_df = tc_weights(model, weight_data)
            weight_matrix_df.rename(columns={0: 'genes'}, inplace=True)
        
        elif model_name == "vanillavae":
            # Optuna optimization for VanillaVAE
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective_vvae(trial, train_tensor, train_tensor, train_data, latent_dim=num_components), n_trials=50)
            
            # Train the best BetaTCVAE model and extract z matrix
            best_trial = study.best_trial
            model = VanillaVAE(input_dim=train_data.shape[1], latent_dim=num_components)
            train_loader = DataLoader(TensorDataset(train_tensor), batch_size=best_trial.params['batch_size'], shuffle=True)
            optimizer = get_optimizer_vvae(best_trial.params['optimizer_type'], model.parameters(), best_trial.params['learning_rate'])
            train_vvae(model, train_loader, optimizer, best_trial.params['epochs'])
            
            # Extract weight matrix
            weight_matrix_df = vanilla_weights(model, weight_data)
            weight_matrix_df.rename(columns={0: 'genes'}, inplace=True)
        # If weight_matrix is generated, proceed to GSEA and append to combined dataframe
        if weight_matrix_df is not None:
            print(f"Running GSEA for {model_name}")
            gsea_results_df = perform_gsea(weight_matrix_df, model_name, num_components)
            combined_results_df = pd.concat([combined_results_df, gsea_results_df], ignore_index=True)
            


# In[8]:


# Save the combined dataframe to a file
final_output_file = output_dir / "combined_z_matrix_gsea_results.parquet"
combined_results_df.to_parquet(final_output_file, index=False)

print(f"Saved final combined z_matrix and GSEA results to {final_output_file}")

#Save as CSV for R 
csv_output_file = output_dir / "combined_z_matrix_gsea_results.csv"
combined_results_df.to_csv(csv_output_file, index=False)


# In[9]:


combined_results_df.sort_values(by='gsea es score', key=abs, ascending = False).head(50)


