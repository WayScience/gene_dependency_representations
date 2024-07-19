import argparse

import torch
from betavae import BetaVAE, evaluate_vae, train_vae
from torch.utils.data import DataLoader, TensorDataset


def get_optimize_args():
    """
    Get arguments for the hyperparameter optimization procedure
    """

    parser = argparse.ArgumentParser()
    # dummy arguments for ipykernel error
    parser.add_argument("-f", "--fff", help="dummy argument", default="1")
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
    parser.add_argument(
        "--min_epochs",
        default=5,
        type=int,
        help="Minimum epochs",
    )
    parser.add_argument(
        "--max_epochs",
        default=1000,
        type=int,
        help="Maximum epochs",
    )
    parser.add_argument(
        "--epoch_step",
        default=100,
        type=int,
        help="Epoch step size",
    )
    parser.add_argument(
        "--min_batch_size",
        default=16,
        type=int,
        help="Minimum batch size",
    )
    parser.add_argument(
        "--max_batch_size",
        default=112,
        type=int,
        help="Maximum batch size",
    )
    parser.add_argument(
        "--batch_size_step",
        default=32,
        type=int,
        help="Batch size step size",
    )
    parser.add_argument(
        "--architecture", 
        default="onelayer", 
        help="VAE architecture")

    parser.add_argument(
        "--dataset", 
        default="cell-painting", 
        help="cell-painting or L1000 dataset"
    )

    args = parser.parse_args()
    return args


def objective(trial, train_tensor, test_tensor, train_df):
    args = get_optimize_args()
    """
    Optuna objective function: optimized by study
    """
    # Define hyperparameters
    latent_dim = trial.suggest_int(
        "latent_dim", args.min_latent_dim, args.max_latent_dim
    )
    beta = trial.suggest_float("beta", args.min_beta, args.max_beta)
    learning_rate = trial.suggest_categorical(
        "learning_rate", [5e-3, 1e-3, 1e-4, 1e-5, 1e-6]
    )
    batch_size = trial.suggest_int(
        "batch_size", args.min_batch_size, args.max_batch_size, args.batch_size_step
    )
    epochs = trial.suggest_int(
        "epochs", args.min_epochs, args.max_epochs, args.epoch_step
    )

    # Create DataLoader
    train_loader = DataLoader(
        TensorDataset(train_tensor), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_tensor), batch_size=batch_size, shuffle=False
    )

    model = BetaVAE(input_dim=train_df.shape[1], latent_dim=latent_dim, beta=beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_vae(model, train_loader, optimizer, epochs=epochs)

    # Evaluate VAE
    val_loss = evaluate_vae(model, test_loader)

    return val_loss
