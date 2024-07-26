import pathlib
from cgi import test
from copy import deepcopy
from random import sample

import numpy as np
import pandas as pd
from matplotlib import testing
from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler
import torch


def load_data(data_directory, adult_or_pediatric="all", id_column="ModelID"):

    # Define data paths
    data_directory = "../0.data-download/data/"
    model_file = pathlib.Path(data_directory, "Model.parquet")
    effect_data_file = pathlib.Path(data_directory, "CRISPRGeneEffect.parquet")

    # Load data
    model_df = pd.read_parquet(model_file)
    effect_df = pd.read_parquet(effect_data_file).dropna(axis=1)

    # rearrange model info and gene effect dataframe indices so id_column is in alphabetical order
    model_df = model_df.sort_index(ascending=True)
    model_df = model_df.reset_index()

    effect_df = effect_df.set_index(id_column).sort_index(ascending=True)
    effect_df = effect_df.reset_index()

    # searching for similar IDs FROM effect df IN model df
    eff_ids = effect_df[id_column].tolist()
    mod_ids = model_df[id_column].tolist()
    eff_vs_mod_ids = set(eff_ids) & set(mod_ids)

    # searching for similar IDs FROM model df In effect df
    mod_vs_eff_ids = set(mod_ids) & set(eff_ids)

    # subset data to only matching IDs (samples in both effect and model data)
    model_df = model_df.loc[model_df[id_column].isin(eff_vs_mod_ids)].reset_index(
        drop=True
    )
    effect_df = effect_df.loc[effect_df[id_column].isin(mod_vs_eff_ids)]

    if adult_or_pediatric != "all":
        model_df = model_df.query("age_categories == @adult_or_pediatric").reset_index(
            drop=True
        )
        model_to_keep = (
            model_df.reset_index(drop=True).ast.literal_eval(id_column).tolist()
        )
        effect_df = effect_df.query(id_column + " == @samples_to_keep").reset_index(
            drop=True
        )

    model_df = model_df.set_index(id_column)
    model_df = model_df.reindex(index=list(mod_vs_eff_ids)).reset_index()

    effect_df = effect_df.set_index(id_column)
    effect_df = effect_df.reindex(index=list(mod_vs_eff_ids)).reset_index()

    return model_df, effect_df


def load_train_test_data(
    data_directory,
    train_file="VAE_train_df.parquet",
    test_file="VAE_test_df.parquet",
    val_file="VAE_val_df.parquet",
    train_or_test="all",
    load_gene_stats=False,
    zero_one_normalize=False,
    ):

    # define directory paths
    training_data_file = pathlib.Path(data_directory, train_file)
    testing_data_file = pathlib.Path(data_directory, test_file)
    validation_data_file = pathlib.Path(data_directory, val_file)

    # load in the data
    train_file = pd.read_parquet(training_data_file)
    test_file = pd.read_parquet(testing_data_file)
    val_file = pd.read_parquet(validation_data_file)

    # overwrite if load_gene_stats is set to true
    if load_gene_stats is True:
        gene_statistics_file = pathlib.Path(
            data_directory, "genes_variances_and_t-tests_df.parquet"
        )
        load_gene_stats = pd.read_parquet(gene_statistics_file)
    else:
        load_gene_stats = None

    # Prepare data for training
    train_features_df = train_file.drop(columns=["ModelID", "age_and_sex"])
    test_features_df = test_file.drop(columns=["ModelID", "age_and_sex"])
    val_features_df = val_file.drop(columns=["ModelID", "age_and_sex"])

    # create dataframe containing the genes that passed an initial QC (see Pan et al. 2022) and their corresponding gene label and extract the gene labels
    gene_dict_df = pd.read_parquet(
        "../0.data-download/data/CRISPR_gene_dictionary.parquet"
    )
    gene_list_passed_qc = gene_dict_df.loc[
        gene_dict_df["qc_pass"], "dependency_column"
    ].tolist()

    # create new training and testing dataframes that contain only the corresponding genes
    train_data = train_features_df.filter(gene_list_passed_qc, axis=1)
    test_data = test_features_df.filter(gene_list_passed_qc, axis=1)
    val_data = val_features_df.filter(gene_list_passed_qc, axis=1)

    # Normalize data
    
    if zero_one_normalize == True:
        train_data = train_data.values.astype(np.float32)
        test_data = test_data.values.astype(np.float32)
        val_data = val_data.values.astype(np.float32)
        
        # Normalize based on data distribution
        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)
        val_data = scaler.transform(val_data)

    # return data based on what user wants
    if train_or_test == "test":

        return test_file

    elif train_or_test == "train":

        return train_file
    
    elif train_or_test == "validation":
        return val_file

    elif train_or_test == "all":

        return train_data, test_data, val_data, load_gene_stats
    
