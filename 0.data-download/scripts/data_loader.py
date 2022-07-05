from copy import deepcopy
import pathlib
from random import sample
import pandas as pd
import numpy 
from numpy import ndarray

def load_data(data_directory, adult_or_pediatric = "all"):

    # Define data paths
    sample_info_file = pathlib.Path(data_directory, "age_visual_sample_info.csv")
    dependency_data_file = pathlib.Path(data_directory, "CRISPR_gene_dependency.csv")
        
    # Load data
    dependency_df = pd.read_csv(dependency_data_file, index_col=0).reset_index()
    sample_df = pd.read_csv(sample_info_file, index_col=0)
    sample_df = sample_df.set_index("DepMap_ID").reset_index()

    # searching for similar IDs FROM dependency df IN sample df
    dep_ids = dependency_df["DepMap_ID"].tolist()
    dep_vs_samp_ids = set(dep_ids) & set(sample_df["DepMap_ID"].tolist())

    # searching for similar IDs FROM s ample df In dependency df
    samp_ids = sample_df["DepMap_ID"].tolist()
    samp_vs_dep_ids = set(samp_ids) & set(sample_df["DepMap_ID"].tolist())
    
    # extract matching IDs
    sample_df = sample_df.loc[sample_df["DepMap_ID"].isin(dep_vs_samp_ids)]
    dependency_df = dependency_df.loc[dependency_df["DepMap_ID"].isin(samp_vs_dep_ids)]
    

    if adult_or_pediatric == "all":
       dependency_df = dependency_df
       samples_to_keep = sample_df.reset_index(drop=True).DepMap_ID.tolist()
       sample_df = sample_df.query("DepMap_ID == @samples_to_keep").reset_index(drop=True)
    elif adult_or_pediatric != "all":
       sample_df = sample_df.query("age_categories == @adult_or_pediatric").reset_index(drop=True)
       samples_to_keep = sample_df.reset_index(drop=True).DepMap_ID.tolist()  
       dependency_df = dependency_df.query("DepMap_ID == @samples_to_keep").reset_index(drop=True)
    return sample_df, dependency_df
