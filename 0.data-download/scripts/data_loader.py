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
    sample_info_file = pathlib.Path(
        data_directory, "Model_age_column_cleaned.csv"
    )
    dependency_data_file = pathlib.Path(data_directory, "CRISPRGeneDependency.csv")

    # Load data
    dependency_df = (
        pd.read_csv(dependency_data_file, index_col=0).reset_index().dropna(1)
    )
    sample_df = pd.read_csv(sample_info_file, index_col=0)

    # rearrange sample info and gene dependency dataframe indices so DepMap_IDs are in alphabetical order
    sample_df = sample_df.sort_index(ascending=True)
    sample_df = sample_df.reset_index()

    dependency_df = (
        dependency_df.set_index("DepMap_ID").sort_index(ascending=True).reset_index()
    )

    # searching for similar IDs FROM dependency df IN sample df
    dep_ids = dependency_df["DepMap_ID"].tolist()
    dep_vs_samp_ids = set(dep_ids) & set(sample_df["DepMap_ID"].tolist())

    # searching for similar IDs FROM sample df In dependency df
    samp_ids = sample_df["DepMap_ID"].tolist()
    samp_vs_dep_ids = set(samp_ids) & set(sample_df["DepMap_ID"].tolist())

    # subset data to only matching IDs (samples in both dependency and sample data)
    sample_df = sample_df.loc[sample_df["DepMap_ID"].isin(dep_vs_samp_ids)].reset_index(
        drop=True
    )
    dependency_df = dependency_df.loc[dependency_df["DepMap_ID"].isin(samp_vs_dep_ids)]

    if adult_or_pediatric != "all":
        sample_df = sample_df.query(
            "age_categories == @adult_or_pediatric"
        ).reset_index(drop=True)
        samples_to_keep = sample_df.reset_index(drop=True).DepMap_ID.tolist()
        dependency_df = dependency_df.query(
            "DepMap_ID == @samples_to_keep"
        ).reset_index(drop=True)

    return sample_df, dependency_df


def load_train_test_data(data_directory, train_or_test="all"):
    # define directory paths
    training_data_file = pathlib.Path(data_directory, "VAE_train_df.csv")
    testing_data_file = pathlib.Path(data_directory, "VAE_test_df.csv")
    gene_statistics_file = pathlib.Path(
        data_directory, "genes_variances_and_t-tests_df.csv"
    )

    train_df = pd.read_csv(training_data_file)
    test_df = pd.read_csv(testing_data_file)
    gene_stats = pd.read_csv(gene_statistics_file)

    if train_or_test == "test":

        return test_df

    if train_or_test == "train":

        return train_df

    return train_df, test_df, gene_stats
