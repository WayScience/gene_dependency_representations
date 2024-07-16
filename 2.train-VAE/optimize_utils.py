# THIS CODE WAS SOURCED FROM THE FOLLOWING URL: https://github.com/broadinstitute/cell-painting-vae/blob/master/scripts/optimize_utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F



class HyperVAE(nn.Module):
    def __init__(
        self, 
        input_dim, 
        latent_dim, 
        beta):
        super(HyperVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
        self.latent_dim = latent_dim
        self.beta = beta

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + self.beta * KLD




def get_optimize_args():
    """
    Get arguments for the hyperparameter optimization procedure
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, help="The name of the project")
    parser.add_argument(
        "--directory",
        default="hyperparameter",
        type=str,
        help="The name of the directory to save results",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="decision to overwrite already started hyperparameter search",
    )
    parser.add_argument(
        "--min_latent_dim",
        default=10,
        type=int,
        help="Minimum size of the internal latent dimensions",
    )
    parser.add_argument(
        "--max_latent_dim",
        default=100,
        type=int,
        help="Maximum size of the internal latent dimensions",
    )
    parser.add_argument(
        "--min_beta",
        default=1,
        type=int,
        help="Minimum beta penalty applied to VAE KL Divergence",
    )
    parser.add_argument(
        "--max_beta",
        default=10,
        type=int,
        help="Maximum beta penalty applied to VAE KL Divergence",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        nargs="+",
        help="learning rates to use in hyperparameter sweep",
    )
    parser.add_argument("--architecture", default="onelayer", help="VAE architecture")

    parser.add_argument(
        "--dataset", default="cell-painting", help="cell-painting or L1000 dataset"
    )

    args = parser.parse_args()
    return args
