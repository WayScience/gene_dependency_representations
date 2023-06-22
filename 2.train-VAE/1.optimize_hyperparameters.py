# THIS CODE WAS SOURCED FROM THE FOLLOWING URL: https://github.com/broadinstitute/cell-painting-vae/blob/master/scripts/optimize-hyperparameters.py

import sys
import pathlib
import numpy as np
import pandas as pd
import keras_tuner

from keras_tuner.tuners import BayesianOptimization
import tensorflow
from keras.callbacks import EarlyStopping

from optimize_utils import (
    HyperVAE,
    CustomBayesianTunerCellPainting,
    CustomBayesianTunerL1000,
    get_optimize_args,
)

sys.path.insert(0, "../0.data-download/scripts/")
from data_loader import load_train_test_data

import random

random.seed(18)

# Load command line arguments
args = get_optimize_args()

# Define architecture
encoder_architecture = []
decoder_architecture = []

# Load Data
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


# Initialize hyper parameter VAE tuning
hypermodel = HyperVAE(
    input_dim=train_df.shape[1],
    min_latent_dim=args.min_latent_dim,
    max_latent_dim=args.max_latent_dim,
    min_beta=args.min_beta,
    max_beta=args.max_beta,
    learning_rate=[5e-3, 1e-3, 1e-4, 1e-5, 1e-6],
    encoder_batch_norm=True,
    encoder_architecture=encoder_architecture,
    decoder_architecture=decoder_architecture,
)


tuner = CustomBayesianTunerCellPainting(
    hypermodel,
    objective="val_loss",
    max_trials=500,
    directory=args.directory,
    project_name=args.project_name,
    overwrite=True,
)

# Search over hyperparameter space to identify optimal combinations
tuner.search(
    train_df,
    validation_data=(test_df, None),
    # callbacks=[EarlyStopping("val_loss", patience=10)], *We decided not to use this line because we want the model to continue training even if there's no improvement of val_loss after 10 epochs.
    verbose=True,
)
