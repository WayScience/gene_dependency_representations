import pathlib
import sys
import pandas as pd

script_directory = pathlib.Path("../2.train-VAE/utils/").resolve()
sys.path.insert(0, str(script_directory))
from betatcvae import tc_extract_latent_dimensions
from betavae import extract_latent_dimensions
from vanillavae import vvae_extract_latent_dimensions

script_directory = pathlib.Path("../2.train-VAE/utils/").resolve()
sys.path.insert(0, str(script_directory))
from betavae import weights
from betatcvae import tc_weights
from vanillavae import vanilla_weights

# Function to extract latent dimensions for all models
def extract_latent_dims(model_name, model, dependency_df, train_and_test_subbed_loader, metadata):
    """
    Extracts latent dimensions from model
    and returns a DataFrame with the latent dimensions.
    
    Args:
        model (object): The trained dimensionality reduction model (PCA, ICA, or NMF).
        dependency_df (pd.DataFrame): Input DataFrame with features to transform.
        
    Returns:
        pd.DataFrame: A DataFrame with the latent dimensions (z columns) and ModelID.
    """
    if model_name in ["pca", "ica", "nmf"]:
        # Extract the latent dimensions for these models
        # Separate numerical columns for transformation
        numerical_columns = dependency_df.select_dtypes(include=[np.number]).columns
        numerical_df = dependency_df[numerical_columns].astype(np.float32)  # Cast to float32

        # Transform data into the reduced space
        latent_matrix = model.transform(numerical_df)

        # Create a DataFrame for the latent dimensions
        latent_df = pd.DataFrame(
            latent_matrix,
            columns=[f'{i}' for i in range(latent_matrix.shape[1])]
        )
        
        # Add ModelID back to the DataFrame if it exists in the original data
        if 'ModelID' in dependency_df.columns:
            latent_df.insert(0, 'ModelID', dependency_df['ModelID'].values)
            
    elif model_name == "betavae":
        latent_df = extract_latent_dimensions(model, train_and_test_subbed_loader, metadata)
    elif model_name == "betatcvae":
        latent_df = tc_extract_latent_dimensions(model, train_and_test_subbed_loader, metadata)
    elif model_name == "vanillavae":
        latent_df = vvae_extract_latent_dimensions(model, train_and_test_subbed_loader, metadata)
    
    return latent_df


def extract_weights(
    model: object, 
    model_name: str, 
    weight_data: pd.DataFrame = None,
    dependency_df: pd.DataFrame = None
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
            columns=dependency_df.drop(columns=["ModelID"]).columns.tolist()).transpose()
        # Reset index to get genes as a column
        weights_df = weights_df.reset_index()
        weights_df.rename(columns={"index": "genes"}, inplace=True)
    elif model_name == "betavae":
        weights_df = weights(model, weight_data)
    elif model_name == "betatcvae":
        weights_df = tc_weights(model, weight_data)
    elif model_name == "vanillavae":
        weights_df = vanilla_weights(model, weight_data)

    print(weights_df)
    # Ensure no duplicate or unintended columns
    weights_df = weights_df.loc[:, ~weights_df.columns.duplicated()]
        
    # Rename first column to 'genes', if appropriate
    if weights_df.columns[0] != "genes":
        weights_df.rename(columns={weights_df.columns[0]: "genes"}, inplace=True)

        # Reset index without adding duplicates
        weights_df = weights_df.reset_index(drop=True)

    return weights_df