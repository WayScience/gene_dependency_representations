from cgi import test
from copy import deepcopy
import pathlib
from random import sample
from matplotlib import testing
import pandas as pd
import numpy
from numpy import ndarray


def load_data(data_directory, adult_or_pediatric="all", id_column="ModelID"):

    # Define data paths
    data_directory = "../0.data-download/data/"
    model_file = pathlib.Path(data_directory, "Model.csv")
    effect_data_file = pathlib.Path(data_directory, "CRISPRGeneEffect.csv")

    # Load data
    model_df = pd.read_csv(model_file)
    effect_df = (
        pd.read_csv(effect_data_file, index_col=0).reset_index().dropna(axis=1)
    )

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
        model_to_keep = model_df.reset_index(drop=True).ast.literal_eval(id_column).tolist()
        effect_df = effect_df.query(
            id_column + " == @samples_to_keep"
        ).reset_index(drop=True)

    model_df = model_df.set_index(id_column)
    model_df = model_df.reindex(index=list(mod_vs_eff_ids)).reset_index()

    effect_df = effect_df.set_index(id_column)
    effect_df = effect_df.reindex(index=list(mod_vs_eff_ids)).reset_index()

    return model_df, effect_df


def load_train_test_data(
    data_directory,
    train_file="VAE_train_df.csv",
    test_file="VAE_test_df.csv",
    train_or_test="all",
    load_gene_stats=False,
):

    # define directory paths
    training_data_file = pathlib.Path(data_directory, train_file)
    testing_data_file = pathlib.Path(data_directory, test_file)

    # load in the data
    train_df = pd.read_csv(training_data_file)
    test_df = pd.read_csv(testing_data_file)

    # overwrite if load_gene_stats is set to true
    if load_gene_stats is True:
        gene_statistics_file = pathlib.Path(
            data_directory, "genes_variances_and_t-tests_df.csv"
        )
        load_gene_stats = pd.read_csv(gene_statistics_file)
    else:
        load_gene_stats = None

    # return data based on what user wants
    if train_or_test == "test":

        return test_df

    elif train_or_test == "train":

        return train_df

    elif train_or_test == "all":

        return train_df, test_df, load_gene_stats
