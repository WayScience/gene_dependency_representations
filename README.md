# Gene Dependency Representations

## Data

### Access

All data are publicly available.

Source: [Cancer Dependency Map resource](https://depmap.org/portal/download/).

## Repository Structure:

This repository is structured as follows:

| Order | Module | Description |
| :---- | :----- | :---------- |
| [0.data-download](0.data-download/) | Download required files | Download gene effect data and cell line information, and download gene QC and construct gene filtering dictionary |
| [1.data-exploration](1.data-exploration/) | Explore and visualize data | Create figures to visualize cell line information and split gene effect data into balanced test and train dataframes |
| [2.train-VAE](2.train-VAE/) | Train BVAE | Optimize hyperparameters and train Beta Variational Autoencoder with optimal hyperparameters and previously created test and train dataframes |
| [3.analysis](3.analysis/) | Analyze BVAE Outputs | Generate heatmaps to visualize death windows by cell line and by genes, run Gene Set Enrichment Analysis with BVAE synthesized data, and analyze extracted BVAE latent space data to compare similarity of cancer between different demographics |

## Goal
Current cancer treatments tend to be toxic and leave patients with lifelong side-effects. The future of drug development is based on synthetic lethality, where the combination of two genetic events results in cell death. It is used in molecular targeted cancer therapy, with the first example of a molecular targeted therapeutic exploiting a synthetic lethal exposed by an inactivated tumor suppressor gene (BRCA1 and 2) receiving FDA approval in 2016 (PARP inhibitor). The benefits of synthetic lethality-based treatment strategies include success against the majority of cancer mutations, simple identification of treatment-responding patients due to its selective nature of specific cancer cell genetic mutations, and reduced toxicity compared to traditional chemotherapy.

**The goal of this project is to discover multivariate gene vulnerability patterns in cancer.**
Using cancer cell line data from DepMap, we can find multivariate gene vulnerability patterns that can be applied to the development of novel cancer treatments. ​
We apply machine learning to gene knockout data to discover multivariate gene vulnerabilities.
We will apply statistical anaylses to determine the differences in multivariate gene vulnerabilities between pediatric and adult cancers.
Once we discover significant multigene vulnerabilities patterns, we hope to inform drug discovery to develop cancer treatments targeting these vulnerabilities.

## Environment Setup

Perform the following steps to set up the `gene_dependency_representations` environment necessary for processing data in this repository.

### Step 1: Create Gene Dependency Representations Environment

```sh
# Run this command to create the proper conda environment

conda env create --force --file environment.yml
```

### Step 2: Activate Gene Dependency Representations Environment

```sh
# Run this command to activate the conda environment for Gene Dependency Representations

conda activate gene_dependency_representations