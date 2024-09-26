#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib 
import optuna
import pandas as pd
import joblib
import torch
import sys

from sklearn.decomposition import PCA, FastICA, NMF
from torch.utils.data import DataLoader, TensorDataset

script_directory = pathlib.Path("../2.train-VAE/utils/").resolve()
sys.path.insert(0, str(script_directory))
from betavae import BetaVAE, train_vae
from betatcvae import BetaTCVAE, train_tc_vae
from vanillavae import VanillaVAE, train_vvae
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
train_data.head()


# In[3]:


# Convert dataframes to tensors
train_tensor = torch.tensor(train_df, dtype=torch.float32)
test_tensor = torch.tensor(test_df, dtype=torch.float32)
val_tensor = torch.tensor(val_df, dtype=torch.float32)


# In[4]:


# Directory where models will be saved
model_save_dir = pathlib.Path("saved_models")
model_save_dir.mkdir(parents=True, exist_ok=True)

# Define the optimization process for the models
latent_dims = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200]
model_names = ["pca", "ica", "nmf", "vanillavae", "betavae", "betatcvae"]

for num_components in latent_dims:
    for model_name in model_names:
        model = None  # Initialize model as None for each iteration
        
        if model_name in ["pca", "ica", "nmf"]:
            if model_name == "pca":
                model = PCA(n_components=num_components)
            elif model_name == "ica":
                model = FastICA(n_components=num_components)
            elif model_name == "nmf":
                model = NMF(n_components=num_components, init='nndsvd', max_iter=2000, random_state=0)
            
            # Fit model to data
            model.fit(train_data)
        
        elif model_name == "betavae":
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective(trial, train_tensor, train_tensor, train_data, latent_dim=num_components), n_trials=50)
            best_trial = study.best_trial
            model = BetaVAE(input_dim=train_data.shape[1], latent_dim=num_components, beta=best_trial.params['beta'])
            train_loader = DataLoader(TensorDataset(train_tensor), batch_size=best_trial.params['batch_size'], shuffle=True)
            optimizer = get_optimizer(best_trial.params['optimizer_type'], model.parameters(), best_trial.params['learning_rate'])
            train_vae(model, train_loader, optimizer, best_trial.params['epochs'])
        
        elif model_name == "betatcvae":
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective_tc(trial, train_tensor, train_tensor, train_data, latent_dim=num_components), n_trials=50)
            best_trial = study.best_trial
            model = BetaTCVAE(input_dim=train_data.shape[1], latent_dim=num_components, beta=best_trial.params['beta'])
            train_loader = DataLoader(TensorDataset(train_tensor), batch_size=best_trial.params['batch_size'], shuffle=True)
            optimizer = get_optimizer_tc(best_trial.params['optimizer_type'], model.parameters(), best_trial.params['learning_rate'])
            train_tc_vae(model, train_loader, optimizer, best_trial.params['epochs'])

        elif model_name == "vanillavae":
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective_vvae(trial, train_tensor, train_tensor, train_data, latent_dim=num_components), n_trials=50)
            
            best_trial = study.best_trial
            model = VanillaVAE(input_dim=train_data.shape[1], latent_dim=num_components)
            train_loader = DataLoader(TensorDataset(train_tensor), batch_size=best_trial.params['batch_size'], shuffle=True)
            optimizer = get_optimizer_vvae(best_trial.params['optimizer_type'], model.parameters(), best_trial.params['learning_rate'])
            train_vvae(model, train_loader, optimizer, best_trial.params['epochs'])

        # Save the trained model with joblib
        if model:
            model_filename = model_save_dir / f"{model_name}_{num_components}_components_model.joblib"
            joblib.dump(model, model_filename)
            print(f"Saved {model_name} with {num_components} components to {model_filename}")

