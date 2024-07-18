#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import pathlib
import numpy as np
import pandas as pd
import hiplot
from optuna.visualization import plot_param_importances
import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from betavae import (
    BetaVAE,
    train_vae,
    evaluate_vae
)
from optimize_utils import(
    get_optimize_args
)


# In[ ]:


#Load command line arguments
args=get_optimize_args()

#Load data

output_dir = pathlib.Path("data")

sys.path.insert(0, "../0.data-download/scripts/")
from data_loader import load_train_test_data
data_directory = pathlib.Path("../0.data-download/data")
dfs = load_train_test_data(data_directory, train_or_test = "all", load_gene_stats = True)

train_feat = dfs[0]
test_feat = dfs[1]
load_gene_stats = dfs[2]

# Prepare data for training
train_features_df = train_feat.drop(columns=["ModelID", "age_and_sex"])
test_features_df = test_feat.drop(columns=["ModelID", "age_and_sex"])

# subsetting the genes

# create dataframe containing the genes that passed an initial QC (see Pan et al. 2022) and their corresponding gene label and extract the gene labels
gene_dict_df = pd.read_csv("../0.data-download/data/CRISPR_gene_dictionary.tsv", delimiter='\t')
gene_list_passed_qc = gene_dict_df.query("qc_pass").dependency_column.tolist()

# create new training and testing dataframes that contain only the corresponding genes
train_df = train_feat.filter(gene_list_passed_qc, axis=1)
test_df = test_feat.filter(gene_list_passed_qc, axis=1)


# In[ ]:


# Normalize data
train_data = train_df.values.astype(np.float32)
test_data = test_df.values.astype(np.float32)

# Normalize based on data distribution
train_data = (train_data - np.min(train_data, axis=0)) / (np.max(train_data, axis=0) - np.min(train_data, axis=0))
test_data = (test_data - np.min(test_data, axis=0)) / (np.max(test_data, axis=0) - np.min(test_data, axis=0))

# Convert dataframes to tensors
train_tensor = torch.tensor(train_data, dtype=torch.float32)
test_tensor = torch.tensor(test_data, dtype=torch.float32)


# In[ ]:


def objective(trial):
    """
    Optuna objective function: optimized by study
    """
    # Define hyperparameters
    latent_dim = trial.suggest_int('latent_dim', args.min_latent_dim, args.max_latent_dim)
    beta = trial.suggest_float('beta', args.min_beta, args.max_beta)
    learning_rate = trial.suggest_categorical('learning_rate', [5e-3, 1e-3, 1e-4, 1e-5, 1e-6])
    batch_size = trial.suggest_categorical('batch_size', [16, 48, 80, 112])
    epochs = trial.suggest_categorical('epochs', [5, 105, 205, 305, 405, 505, 605, 705, 805, 905])

    # Create DataLoader
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=batch_size, shuffle=False)

    model = BetaVAE(input_dim=train_df.shape[1], latent_dim=latent_dim, beta=beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    
    loss = train_vae(model, train_loader, optimizer, epochs=epochs)
    
    # Evaluate VAE
    val_loss = evaluate_vae(model, test_loader)
    
    return val_loss


# In[ ]:


# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=500)
# Save best hyperparameters
best_trial = study.best_trial
print(best_trial)
print(f'Best trial: {best_trial.values}')
print(f'Best hyperparameters: {best_trial.params}')

