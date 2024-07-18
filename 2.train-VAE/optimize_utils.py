def get_optimize_args():
    """
    Get arguments for the hyperparameter optimization procedure
    """
    import argparse

    parser = argparse.ArgumentParser()
    #dummy arguments for ipykernel error
    parser.add_argument("-f","--fff", help="dummy argument", default="1")
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
        default = 100,
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
        default = 32,
        type=int,
        help="Batch size step size",
    )
    parser.add_argument("--architecture", default="onelayer", help="VAE architecture")

    parser.add_argument(
        "--dataset", default="cell-painting", help="cell-painting or L1000 dataset"
    )

    args = parser.parse_args()
    return args
