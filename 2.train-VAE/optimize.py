# THIS CODE WAS SOURCED FROM THE FOLLOWING URL: https://github.com/broadinstitute/cell-painting-vae/blob/7c1762b9e8d4d1318130924f0cd7d8a275a3a539/scripts/optimize.py

import subprocess
import argparse
import pathlib

path = "./"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", default="cell-painting", help="cell-painting or L1000 dataset"
)
args = parser.parse_args()

directory = "parameter_sweep"


learning_rate = [1e-2, 1e-3, 1e-4, 1e-5]
min_latent_dim = 5
max_latent_dim = 150
max_beta = 1
architectures = ["onelayer", "twolayer", "threelayer"]


min_beta = 1

dataset = args.dataset

batch_norm = [True, False]

# Create commands
learning_rate_arg = ["--learning_rate"] + [str(x) for x in learning_rate]

params = [
    "python",
    pathlib.Path(path, "optimize_hyperparameters.py"),
    "--overwrite",
    "--directory",
    directory,
    "--min_latent_dim",
    str(min_latent_dim),
    "--max_latent_dim",
    str(max_latent_dim),
    "--min_beta",
    str(min_beta),
    "--max_beta",
    str(max_beta),
    "--dataset",
    str(dataset),
] + learning_rate_arg

all_commands = []
for architecture in architectures:
    project_params = params + [
        "--project_name",
        architecture,
        "--architecture",
        architecture,
    ]
    all_commands.append(project_params)

if __name__ == "__main__":
    for cmd in all_commands:
        subprocess.call(cmd)
