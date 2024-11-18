#!/usr/bin/env python
# coding: utf-8

# ## GSEA Analysis Pipeline for Dimensionality Reduction Models

# This script performs Gene Set Enrichment Analysis (GSEA) on weight matrices  extracted from various dimensionality reduction models (PCA, ICA, NMF, VanillaVAE,  BetaVAE, and BetaTCVAE). It iterates over different latent dimensions and model types, extracts the weight matrices, and computes GSEA scores. The results are combined into a single output file for downstream analysis.

# In[1]:


import joblib
import pandas as pd
import blitzgsea as blitz
import random
import pathlib
import sys

script_directory = pathlib.Path("../2.train-VAE/utils/").resolve()
sys.path.insert(0, str(script_directory))
from betavae import BetaVAE, weights
from betatcvae import BetaTCVAE, tc_weights
from vanillavae import VanillaVAE, vanilla_weights

script_directory = pathlib.Path("../utils/").resolve()
sys.path.insert(0, str(script_directory))
from data_loader import load_train_test_data, load_model_data


# In[2]:


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


# In[3]:


#Load weight data for VAEs
data_directory = pathlib.Path("../0.data-download/data").resolve()
weight_df = load_train_test_data(
    data_directory, train_or_test="train"
)

gene_list_passed_qc = gene_dict_df.loc[
    gene_dict_df["qc_pass"], "dependency_column"
].tolist()

weight_data = weight_df.filter(gene_list_passed_qc, axis=1)
weight_data.head()


# In[4]:


def extract_weights(
    model: object, 
    model_name: str, 
    weight_data: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Extracts weight matrix from a given model based on its type.

    Args:
        model (object): A fitted model (e.g., PCA, ICA, NMF, or a VAE).
        model_name (str): Name of the model (e.g., 'pca', 'ica', 'nmf', 'betavae', 'betatcvae', 'vanillavae').
        weight_data (pd.DataFrame, optional): Data required for weight extraction in VAE models.

    Returns:
        pd.DataFrame: DataFrame containing weights with genes as rows and components as columns.
    """
    if model_name in ["pca", "ica", "nmf"]:
        weights_df = pd.DataFrame(
            model.components_,
            columns=dependency_df.drop(columns=["ModelID"]).columns.tolist()
        ).transpose()
        weights_df.columns = [f"{x}" for x in range(0, weights_df.shape[1])]
    elif model_name == "betavae":
        weights_df = weights(model, weight_data)
        weights_df.rename(columns={0: 'genes'}, inplace=True)
    elif model_name == "betatcvae":
        weights_df = tc_weights(model, weight_data)
        weights_df.rename(columns={0: 'genes'}, inplace=True)
    elif model_name == "vanillavae":
        weights_df = vanilla_weights(model, weight_data)
        weights_df.rename(columns={0: 'genes'}, inplace=True)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    weights_df = weights_df.reset_index().rename(columns={"index": "genes"})
    return weights_df

def perform_gsea(weights_df: pd.DataFrame, model_name: str, num_components: int, lib: str = "CORUM") -> pd.DataFrame:
    """
    Performs Gene Set Enrichment Analysis (GSEA) for a given weight matrix.

    Args:
        weights_df (pd.DataFrame): DataFrame containing genes and their associated weights.
        model_name (str): Name of the model being analyzed.
        num_components (int): Number of components used in the model.
        lib (str): Name of the GSEA library (default: 'CORUM').

    Returns:
        pd.DataFrame: Results of GSEA with columns for pathway, enrichment scores, and other metrics.
    """
    
    library = blitz.enrichr.get_library(lib)
    random.seed(0)
    seed = random.random()
    gsea_results = []
    for col in weights_df.columns[1:]:  # Skip 'genes' column
        gene_signature = weights_df[['genes', col]]
        if gene_signature.shape[0] > 0:
            try:
                gsea_result = blitz.gsea(gene_signature, library, seed=seed)
                gsea_result = gsea_result.reset_index()
                for _, pathway_result in gsea_result.iterrows():
                    result_row = {
                        "z": int(col),
                        "full_model_z": num_components,
                        "model": str(model_name),
                        "reactome_pathway": str(pathway_result['Term']),
                        "gsea_es_score": pathway_result['es'],
                        "nes_score": pathway_result['nes'],
                        "p_value": pathway_result['pval'],
                        "shuffled": False
                    }
                    gsea_results.append(result_row)
                    
            except ZeroDivisionError:
                print(f"Skipping GSEA for {col} due to zero division error.")
    
    gsea_results_df = pd.DataFrame(gsea_results)
    return gsea_results_df

# Define the location of the saved models and output directory for GSEA results
model_save_dir = pathlib.Path("saved_models")
output_dir = pathlib.Path("gsea_results")
output_dir.mkdir(parents=True, exist_ok=True)

# Latent dimensions and model names to iterate over
latent_dims = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200]
model_names = ["pca", "ica", "nmf", "vanillavae", "betavae", "betatcvae"]

final_output_file = output_dir / "combined_z_matrix_gsea_results.parquet"
try:
    combined_results_df = pd.read_parquet(final_output_file)
    print(f"Loaded existing results from {final_output_file}")
except FileNotFoundError:
    # If the file doesn't exist, initialize an empty DataFrame
    combined_results_df = pd.DataFrame()
    print(f"No existing file found. Initialized empty DataFrame.")

for num_components in latent_dims:
    for model_name in model_names:
        # Load the saved model
        # Check if this model and latent dimension have already been processed
        if not combined_results_df.empty:
            if ((combined_results_df['model'] == model_name) & 
                (combined_results_df['full_model_z'] == num_components)).any():
                print(f"Skipping {model_name} with {num_components} dimensions as it is already processed.")
                continue  # Skip to the next iteration if this combination is already present
        model_filename = model_save_dir / f"{model_name}_{num_components}_components_model.joblib"
        if model_filename.exists():
            print(f"Loading model from {model_filename}")
            model = joblib.load(model_filename)

            # Extract the weight matrix
            try:
                weight_matrix_df = extract_weights(model, model_name, weight_data)
            except ValueError as e:
                print(e)
                continue

            # Perform GSEA
            gsea_results_df = perform_gsea(weight_matrix_df, model_name, num_components)
            combined_results_df = pd.concat([combined_results_df, gsea_results_df], ignore_index=True)
        else:
            print(f"Model file {model_filename} not found. Skipping.")

            


# In[5]:


# Save the combined dataframe to a file
final_output_file = output_dir / "combined_z_matrix_gsea_results.parquet"
combined_results_df.to_parquet(final_output_file, index=False)

print(f"Saved final combined z_matrix and GSEA results to {final_output_file}")


# In[6]:


combined_results_df.sort_values(by='gsea_es_score', key=abs, ascending = False).head(50)

