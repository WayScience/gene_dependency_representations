import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
import numpy as np
import pathlib
import math

class BetaTCVAE(nn.Module):
    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self, input_dim, latent_dim, beta, hidden_dim, num_layers, anneal_steps, alpha, gamma):
        super(BetaTCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.anneal_steps = anneal_steps
        self.alpha = alpha
        self.gamma = gamma

        # Create encoder with variable number of layers
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        # The final layer outputs mu and log_var
        layers.append(nn.Linear(in_dim, latent_dim * 2))
        self.encoder = nn.Sequential(*layers)

        # Create decoder with variable number of layers
        layers = []
        in_dim = latent_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        # The final layer outputs the reconstructed input
        layers.append(nn.Linear(in_dim, input_dim))
        layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*layers)

    def reparameterize(self, mu, log_var):
        """
        Reparameterize from N(mu, var) to N(0,1)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass through the VAE.

        Args:

            x (Tensor): Input data.

        Returns:
            Reconstructed data, mean, and log variance of the latent space.
        """
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), x, mu, log_var, z
    
    def encode(self, x):
        """
        Encode the input data to the latent space.

        Args:
            x (Tensor): Input data.

        Returns:
            Mean and log variance of the latent space.
        """
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        return mu, log_var

    def log_density_gaussian(self, x, mu, log_var):
        """
        Computes the log pdf of the Gaussian with parameters mu and log_var at x
        """
        norm = -0.5 * (math.log(2 * math.pi) + log_var)
        log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-log_var))
        return log_density

    def loss_function(self, *args, **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        z = args[4]

        weight = 1

        recons_loss = F.mse_loss(recons, input, reduction='sum')

        log_q_zx = self.log_density_gaussian(z, mu, log_var).sum(dim=1)

        zeros = torch.zeros_like(z)
        log_p_z = self.log_density_gaussian(z, zeros, zeros).sum(dim=1)

        batch_size, latent_dim = z.shape
        mat_log_q_z = self.log_density_gaussian(z.view(batch_size, 1, latent_dim),
                                                mu.view(1, batch_size, latent_dim),
                                                log_var.view(1, batch_size, latent_dim))

        # Reference
        # [1] https://github.com/YannDubs/disentangling-vae/blob/535bbd2e9aeb5a200663a4f82f1d34e084c4ba8d/disvae/utils/math.py#L54
        dataset_size = (1 / kwargs['M_N']) * batch_size # dataset size
        strat_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))
        importance_weights = torch.Tensor(batch_size, batch_size).fill_(1 / (batch_size -1)).to(input.device)
        importance_weights.view(-1)[::batch_size] = 1 / dataset_size
        importance_weights.view(-1)[1::batch_size] = strat_weight
        importance_weights[batch_size - 2, 0] = strat_weight
        log_importance_weights = importance_weights.log()

        mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)

        log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
        log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)

        mi_loss  = (log_q_zx - log_q_z).mean()
        tc_loss = (log_q_z - log_prod_q_z).mean()
        kld_loss = (log_prod_q_z - log_p_z).mean()

        if self.training:
            self.num_iter += 1
            anneal_rate = min(0 + 1 * self.num_iter / self.anneal_steps, 1)
        else:
            anneal_rate = 1.

        loss = recons_loss/batch_size + \
               self.alpha * mi_loss + \
               weight * (self.beta * tc_loss +
                         anneal_rate * self.gamma * kld_loss)
        
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'KLD': kld_loss,
                'TC_Loss': tc_loss,
                'MI_Loss': mi_loss}


def train_model(model, train_loader, optimizer):
    """
    Definition of the VAE training model for use in both training and compiling
    Args:
        model (VAE): VAE model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for the model.
    Returns:
        The average training loss for the current epoch
    """
    model.train()
    train_loss = 0
    for batch in train_loader:
        data = batch[0]
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = model.loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    avg_train_loss = train_loss / len(train_loader.dataset)
    return avg_train_loss

def train_vae(model, train_loader, optimizer, epochs):
    """
    Train the VAE model.

    Args:
        model (VAE): VAE model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for the model.
        epochs (int, optional): Number of training epochs. Defaults to 5.
    Returns:
        Training history (loss)
    """
    train_loss_history = []
    model.train()
    for epoch in range(epochs):
        avg_train_loss = train_model(model, train_loader, optimizer)
        train_loss_history.append(avg_train_loss)
        print(f"Epoch {epoch}, Loss: {avg_train_loss}")

    return train_loss_history

def evaluate_vae(model, val_loader):
    """
    Evaluate the VAE model.

    Args:
        model (VAE): VAE model to be evaluated.
        test_loader (DataLoader): DataLoader for the test data.

    Returns:
        Average loss over the test dataset.
    """
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            data = batch[0]
            recon, mu, log_var = model(data)
            val_loss += model.loss_function(recon, data, mu, log_var).item()
    return val_loss / len(val_loader.dataset)


def compile_vae(model, train_loader, val_loader, test_loader, optimizer, epochs):
    model.train()
    train_loss_history = []
    val_loss_history = []
    test_loss_history = []

    for epoch in range(epochs):

        avg_train_loss = train_model(model, train_loader, optimizer)
        train_loss_history.append(avg_train_loss)

        avg_val_loss = evaluate_vae(model, val_loader)
        val_loss_history.append(avg_val_loss)

        avg_test_loss = evaluate_vae(model, test_loader)
        test_loss_history.append(avg_test_loss)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, Test Loss: {avg_test_loss}")

    return train_loss_history, val_loss_history, test_loss_history



def extract_latent_dimensions(model, data_loader, metadata):
    """
    Extract latent dimensions from the VAE model and save them with Model IDs.

    Args:
        model (VAE): Trained VAE model.
        data_loader (DataLoader): DataLoader for the data.
        metadata (DataFrame): Metadata containing Model IDs.

    Returns:
        DataFrame with latent dimensions and Model IDs.
    """
    model.eval()
    latent_space = []
    with torch.no_grad():
        for batch in data_loader:
            data = batch[0]
            mu, _ = model.encode(data)
            latent_space.append(mu.cpu().numpy())

    latent_space = np.concatenate(latent_space, axis=0)
    latent_df = pd.DataFrame(latent_space)
    latent_df.insert(0, 'ModelID', metadata['ModelID'])

    latent_df_dir = pathlib.Path("./results/latent_df.parquet")
    latent_df.to_parquet(latent_df_dir, index=False)

    return latent_df


def weights(model, subset_train_df):
    """
    Extract weight from the VAE model and save them with Model IDs.

    Args:
        model (VAE): Trained VAE model.
        subset_train_df: the qc training dataframe

    Returns:
        Gene weight dataframe
    """
    weight_matrix = model.encoder[0].weight.detach().cpu().numpy().T  # Transpose the weight matrix
    weight_df = pd.DataFrame(weight_matrix)

    # Save as parquet to use for heatmap
    weight_df_dir = pathlib.Path("./results/weight_matrix_encoder.parquet")
    weight_df.to_parquet(weight_df_dir, index=False)

    # Transpose, add gene names back in, transpose again, reset the index, renumber the columns 
    weight_df_T_df = weight_df.T
    gene_weight_df = pd.DataFrame(data=weight_df_T_df.values, columns=subset_train_df.columns)
    gene_weight_T_df = gene_weight_df.T

    gw_reindex_df = gene_weight_T_df.reset_index()
    gw_renumber_df = gw_reindex_df.rename(columns={x: y for x, y in zip(gw_reindex_df.columns, range(0, len(gw_reindex_df.columns)))})

    # Remove numbers from gene name column

    split_data_df = gw_renumber_df[0].str.split(" ", expand=True)
    gene_name_df = split_data_df.iloc[:, :1]
    trimmed_gene_weight_df = gw_renumber_df.iloc[:, 1:]

    final_gene_weights_df = gene_name_df.join(trimmed_gene_weight_df)

    # Save as parquet to use for GSEA
    gene_weight_dir = pathlib.Path("./results/weight_matrix_gsea.parquet")
    final_gene_weights_df.to_parquet(gene_weight_dir, index=False)

    return final_gene_weights_df

