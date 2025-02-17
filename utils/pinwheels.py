import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os 

def plot_pinwheel(sample_id, final_scores, id, name):
    """
    Generates and saves a polar plot (pinwheel) visualization for pathway scores.

    Parameters:
        sample_id (str): Identifier for the sample being visualized.
        final_scores (DataFrame): DataFrame containing pathway IDs and scores.
        id (str): Column name for pathway identifiers.
        name (str): Type of plot - e.g. pathway or drug.
    """
    save_dir = pathlib.Path("./visualize/pinwheels").resolve()
    
    # Get pathway names and scores
    ids = final_scores[id].values
    scores = final_scores["pathway_score"].values

    # Convert to radians for plotting
    num_pathways = len(ids)
    angles = np.linspace(0, 2 * np.pi, num_pathways, endpoint=False).tolist()

    # Create polar plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.fill(angles, scores, color="mediumpurple", alpha=0.8)

    # Get top N indices by score magnitude
    top_n_indices = np.argsort(scores)[-5:]
    
    # Add labels for the top N scores
    for i in top_n_indices:
        ax.text(
            angles[i], scores[i], ids[i],
            horizontalalignment="center", 
            verticalalignment="center", 
            fontsize=14, 
            color="black", 
            rotation=0
        )
        
    ax.set_yticklabels([])
    ax.set_xticklabels([])  # Remove angle numbers

    plt.title(f"{name} Graph for {sample_id}", y=1.1, fontsize=22)
    
    # Save the plot to the specified directory
    plot_path = os.path.join(save_dir, f"pinwheel_{name}_{sample_id}.png")
    plt.savefig(plot_path, bbox_inches='tight')  # Save with tight bounding box
    plt.close()  # Close the plot to free memory

def compute_and_plot_latent_scores(model_id, latent_df, comp_df, id, score, name):
    """
    Computes pathway scores based on latent dimensions and generates a pinwheel plot.

    Parameters:
        model_id (str): Identifier for the model.
        latent_df (DataFrame): DataFrame containing latent dimension scores.
        comp_df (DataFrame): DataFrame containing pathway or drug comparison scores.
        id (str): Column name for pathway or drug identifiers.
        score (str): Column name for the score used in calculations.
        name (str): Type of plot - e.g. pathway or drug.
    """
    # Filter by ModelID
    latent_filtered = latent_df[latent_df["ModelID"] == model_id].copy()
    
    # Melt latent dataframe
    latent_long = latent_filtered.melt(
        id_vars=["ModelID", "model", "latent_dim_total", "init", "seed"],
        var_name="z",  # Latent dimension names
        value_name="latent_score"
    )
    
    # Convert z from string to int
    latent_long["z"] = latent_long["z"].astype(str).str.replace(r"^z_", "", regex=True).astype(int)
        
    # Prepare GSEA/drug dataframe
    comp_long = comp_df.rename(columns={"full_model_z": "latent_dim_total"}).copy()
    comp_long = comp_long[[id, "latent_dim_total", "init", "model", "z", score]]
    comp_long["z"] = comp_long["z"].astype(str).str.replace(r"^z_", "", regex=True).astype(int)

    # Merge latent and GSEA data while keeping only one "model" and "init"
    merged_df = latent_long.merge(
        comp_long, 
        on=["z", "model", "latent_dim_total", "init"],
        how="inner"
    )

    # Compute final pathway scores
    merged_df["pathway_score"] = abs(merged_df["latent_score"]) * comp_long[score]
    merged_df = merged_df.dropna(subset=["pathway_score"])
    
    print(model_id)
    
    # Generate the plot
    plot_pinwheel(model_id, merged_df, id, name)
