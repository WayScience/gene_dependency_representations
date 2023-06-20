from cgi import test
from copy import deepcopy
import pathlib
from random import sample
from matplotlib import testing
import pandas as pd
import numpy
from numpy import ndarray


def load_data(data_directory, adult_or_pediatric="all"):

    # Define data paths
    data_directory = "../0.data-download/data/"
    model_file = pathlib.Path(data_directory, "Model.csv")
    dependency_data_file = pathlib.Path(data_directory, "CRISPRGeneDependency.csv")

    # Load data
    dependency_df = (
        pd.read_csv(dependency_data_file, index_col=0).reset_index().dropna(axis=1)
    )
    model_df = pd.read_csv(model_file)
 
    # Ensure ID variable is ModelID; if not then rename
    try:
        dependency_df['ModelID']
    except:
        dependency_df = dependency_df.rename(columns={dependency_df.columns[0]:"ModelID"})

    try:
        model_df['ModelID']
    except ValueError:
        model_df = model_df.rename(columns={model_df.columns[0]:"ModelID"})

    
    # rearrange model info and gene dependency dataframe indices so ModelIDs are in alphabetical order
    model_df = model_df.sort_index(ascending=True)
    model_df = model_df.reset_index()

    dependency_df = dependency_df.set_index("ModelID").sort_index(ascending=True)
    dependency_df = dependency_df.reset_index()

    # searching for similar IDs FROM dependency df IN model df
    dep_ids = dependency_df["ModelID"].tolist()
    mod_ids = model_df["ModelID"].tolist()
    dep_vs_mod_ids = set(dep_ids) & set(mod_ids)

    # searching for similar IDs FROM model df In dependency df
    mod_vs_dep_ids = set(mod_ids) & set(dep_ids)

    # subset data to only matching IDs (samples in both dependency and model data)
    model_df = model_df.loc[model_df["ModelID"].isin(dep_vs_mod_ids)].reset_index(
        drop=True
    )
    dependency_df = dependency_df.loc[dependency_df["ModelID"].isin(mod_vs_dep_ids)]

    if adult_or_pediatric != "all":
        model_df = model_df.query("age_categories == @adult_or_pediatric").reset_index(
            drop=True
        )
        model_to_keep = model_df.reset_index(drop=True).ModelID.tolist()
        dependency_df = dependency_df.query(
            "ModelID == @samples_to_keep"
        ).reset_index(drop=True)

    model_df = model_df.set_index("ModelID")
    model_df = model_df.reindex(index=list(mod_vs_dep_ids)).reset_index()

    dependency_df = dependency_df.set_index("ModelID")
    dependency_df = dependency_df.reindex(index=list(mod_vs_dep_ids)).reset_index()

    return model_df, dependency_df


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
