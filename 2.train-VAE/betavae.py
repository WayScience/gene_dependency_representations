# Code based on https://github.com/1Konny/Beta-VAE/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaVAE(nn.Module):
    """
    Beta Variational Autoencoder (VAE) class.

    Args:
        input_dim (int): Dimension of the input data.
        hidden_dim (int): Dimension of the hidden layer. (this is utilized if two layers)
        latent_dim (int): Dimension of the latent space.
        beta (float): Weight for the Kullback-Leibler divergence term in the loss function.
    """

    def __init__(self, input_dim, latent_dim, beta):
        super(BetaVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim), 
            nn.BatchNorm1d(input_dim), 
            nn.Sigmoid()
        )
        self.latent_dim = latent_dim
        self.beta = beta

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
        return self.decoder(z), mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        """
        Compute the VAE loss function (Mean squared error).

        Args:
            recon_x (Tensor): Reconstructed data.
            x (Tensor): Original input data.
            mu (Tensor): Mean of the latent Gaussian.
            log_var (Tensor): Log variance of the latent Gaussian.

        Returns:
            Computed loss.
        """
        MSE = F.mse_loss(recon_x, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + self.beta * KLD


def train_vae(model, train_loader, optimizer, epochs=5):
    """
    Train the VAE model.

    Args:
        model (VAE): VAE model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for the model.
        epochs (int, optional): Number of training epochs. Defaults to 5.
    """
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch in train_loader:
            data = batch[0]
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(data)
            loss = model.loss_function(recon_batch, data, mu, log_var)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset)}")


def evaluate_vae(model, test_loader):
    """
    Evaluate the VAE model.

    Args:
        model (VAE): VAE model to be evaluated.
        test_loader (DataLoader): DataLoader for the test data.

    Returns:
        Average loss over the test dataset.
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            data = batch[0]
            recon, mu, log_var = model(data)
            test_loss += model.loss_function(recon, data, mu, log_var).item()
    return test_loss / len(test_loader.dataset)
