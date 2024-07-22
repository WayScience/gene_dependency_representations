#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pathlib
import sys

import numpy as np
import optuna
import pandas as pd
import torch
from betavae import BetaVAE, evaluate_vae, train_vae
from optimize_utils import get_optimize_args, objective
from torch.utils.data import DataLoader, TensorDataset

script_directory = pathlib.Path("../0.data-download/scripts/").resolve()
sys.path.insert(0, str(script_directory))
from data_loader import load_train_test_data

# In[3]:


# Load command line arguments
args = get_optimize_args()

# Load data
data_directory = pathlib.Path("../0.data-download/data").resolve()

train_data, test_data, val_data, load_gene_stats = load_train_test_data(
    data_directory, train_or_test="all", load_gene_stats=True, zero_one_normalize=True
)


# In[4]:


# Convert dataframes to tensors
train_tensor = torch.tensor(train_data, dtype=torch.float32)
test_tensor = torch.tensor(test_data, dtype=torch.float32)
val_tensor = torch.tensor(val_data, dtype=torch.float32)


# In[7]:


# Run Optuna optimization
study = optuna.create_study(direction="minimize")
study.optimize(
    lambda trial: objective(trial, train_tensor, val_tensor, train_data), n_trials=500
)


# In[ ]:


# Save best hyperparameters
best_trial = study.best_trial
print(best_trial)
print(f"Best trial: {best_trial.values}")
print(f"Best hyperparameters: {best_trial.params}")

