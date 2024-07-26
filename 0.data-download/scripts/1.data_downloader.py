#!/usr/bin/env python
# coding: utf-8

# ## Download Dependency Data
# 
# Source: [Cancer Dependency Map resource](https://depmap.org/portal/download/).
# 
# - `CRISPRGeneDependency.parquet`: The data in this document describes the probability that a gene knockdown has an effect on cell-inhibition or death. These probabilities are derived from the data contained in CRISPRGeneEffect.parquet using methods described [here](https://doi.org/10.1101/720243)
# - `Model.parquet`: Metadata for all of DepMap’s cancer models/cell lines.
# - `CRISPRGeneEffect.parquet`: The data in this document are the Gene Effect Scores obtained from CRISPR knockout screens conducted by the Broad Institute. Negative scores notate that cell growth inhibition and/or death occurred following a gene knockout. Information on how these Gene Effect Scores were determined can be found [here](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02540-7)
# - `depmap_gene_meta.tsv`: Genes that passed QC and were included in the training model for Pan et al. 2022. We use this data to filter genes as input to our models. The genes were filtered based 1) variance, 2) perturbation confidence, and 3) high on target predictions based on high correlation across other guides.
# 
# > Pan J, Kwon JJ, Talamas JA, Borah AA, Vazquez F, Boehm JS, Tsherniak A, Zitnik M, McFarland JM, Hahn WC. Sparse dictionary learning recovers pleiotropy from human cell fitness screens. Cell Syst. 2022 Apr 20;13(4):286-303.e10. doi: 10.1016/j.cels.2021.12.005. Epub 2022 Jan 31. PMID: 35085500; PMCID: PMC9035054.

# In[1]:


import pathlib
import urllib.request


# In[2]:


def download_dependency_data(figshare_id, figshare_url, output_file):
    """
    Download the provided figshare resource
    """
    urllib.request.urlretrieve(f"{figshare_url}/{figshare_id}", output_file)


# In[3]:


# Set download constants
output_dir = pathlib.Path("data")
figshare_url = "https://ndownloader.figshare.com/files/"

download_dict = {
    "40448549": "CRISPRGeneDependency.parquet",
    "40448555": "CRISPRGeneEffect.parquet",
    "40448834": "Model.parquet",
    "29094531": "depmap_gene_meta.tsv",  # Data from Pan et al. 2022
}


# In[4]:


# Make sure directory exists
output_dir.mkdir(exist_ok=True)


# In[5]:


for figshare_id in download_dict:
    # Set output file
    output_file = pathlib.Path(output_dir, download_dict[figshare_id])

    # Download the dependency data
    print(f"Downloading {output_file}...")

    download_dependency_data(
        figshare_id=figshare_id, figshare_url=figshare_url, output_file=output_file
    )

