# THIS CODE WAS SOURCED FROM THE FOLLOWING URL: https://github.com/broadinstitute/cell-painting-vae/blob/master/scripts/optimize-hyperparameters.py

import sys
import pathlib
import numpy as np
import pandas as pd
import keras_tuner

from keras_tuner.tuners import BayesianOptimization
import tensorflow
from keras.callbacks import EarlyStopping 

from optimize_utils import HyperVAE, CustomBayesianTunerCellPainting, CustomBayesianTunerL1000, get_optimize_args

sys.path.insert(0, "../0.data-download/scripts/")
from data_loader import load_train_test_data

# Load command line arguments
args = get_optimize_args()

# Define architecture
encoder_architecture = []
decoder_architecture = []

# Load Data
data_directory = pathlib.Path("../0.data-download/data")
dfs = load_train_test_data(data_directory, train_or_test = "all")

train_feat = dfs[0]
test_feat = dfs[1]
gene_stats = dfs[2]


# Prepare data for training
train_features_df = train_feat.drop(columns= ["DepMap_ID", "age_and_sex"])
test_features_df = test_feat.drop(columns= ["DepMap_ID", "age_and_sex"])

# subsetting the genes 
# create dataframe containing the 1000 genes with the largest variances and their corresponding gene label and extract the gene labels
largest_var_df = gene_stats.nlargest(1000, "variance")
gene_list = largest_var_df["gene_ID"].tolist()
gene_list

# create new training and testing dataframes that contain only the corresponding genes
train_df = train_feat.filter(gene_list, axis = 1)
test_df = test_feat.filter(gene_list, axis = 1)

# scvale the data
def absolute_maximum_scale(series):
    return series / series.abs().max()
for col in train_df.columns:
    train_df[col] = absolute_maximum_scale(train_df[col])
for col in test_df.columns:
    test_df[col] = absolute_maximum_scale(test_df[col])
    
    
# Initialize hyper parameter VAE tuning
hypermodel = HyperVAE(
        input_dim=train_df.shape[1],
        min_latent_dim=args.min_latent_dim,
        max_latent_dim=args.max_latent_dim,
        min_beta=args.min_beta,
        max_beta=args.max_beta,
        learning_rate=[5e-2, 1e-2, 5e-3, 1e-3, 1e-4],
        encoder_batch_norm=True,
        encoder_architecture=encoder_architecture,
        decoder_architecture=decoder_architecture,
)


tuner = CustomBayesianTunerCellPainting(
        hypermodel,
        objective="val_loss",
        max_trials=1000,
        directory=args.directory,
        project_name=args.project_name,
        overwrite=True,
)

# Search over hyperparameter space to identify optimal combinations
tuner.search(
        train_df,
        validation_data=(test_df, None),
        callbacks=[EarlyStopping("val_loss", patience=10)],
        verbose=True,
)
