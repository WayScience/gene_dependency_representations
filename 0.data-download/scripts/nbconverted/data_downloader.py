#!/usr/bin/env python
# coding: utf-8

# ## Download Dependency Data
# 
# Source: [Cancer Dependency Map resource](https://depmap.org/portal/download/).
# 
# - `CRISPRGeneDependency.csv`: The data in this document describes the probability that a gene knockdown has an effect on cell-inhibition or death. These probabilities are derived from the data contained in CRISPRGeneEffect.csv using methods described [here](https://doi.org/10.1101/720243)
# - `Model.csv`: Metadata for all of DepMap’s cancer models/cell lines.
# - `CRISPRGeneEffect.csv`: The data in this document are the Gene Effect Scores obtained from CRISPR knockout screens conducted by the Broad Institute. Negative scores notate that cell growth inhibition and/or death occurred following a gene knockout. Information on how these Gene Effect Scores were determined can be found [here](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02540-7)

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
    "34990033": "CRISPRGeneDependency.csv",
    "35020903": "Model.csv"
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
        figshare_id=figshare_id,
        figshare_url=figshare_url,
        output_file=output_file
    )

