#!/usr/bin/env python
# coding: utf-8

# ## Download Dependency Data
# 
# Source: [Cancer Dependency Map resource](https://depmap.org/portal/download/).
# 
# - `CRISPR_gene_dependency.csv`: DESCRIBE HERE
# - `sample_info.csv`: DESCRIBE HERE

# In[2]:


import pathlib
import urllib.request


# In[3]:


def download_dependency_data(figshare_id, figshare_url, output_file):
    """
    Download the provided figshare resource
    """
    urllib.request.urlretrieve(f"{figshare_url}/{figshare_id}", output_file)


# In[4]:


# Set download constants
output_dir = pathlib.Path("data")
figshare_url = "https://ndownloader.figshare.com/files/"

download_dict = {
    "34990033": "CRISPR_gene_dependency.csv",
    "35020903": "sample_info.csv"
}


# In[5]:


# Make sure directory exists
output_dir.mkdir(exist_ok=True)


# In[6]:


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

