# Downloading collaborator data

In this module, we download and test RNA-seq data from a collaborator on the project (Dr. Adam Green) to extend and test real-world applications. The get_geo_data.sh script will download this data from the NCBI database. 


## Files Downloaded
| **File**                                   | **Description**                                           |
|-------------------------------------------|-----------------------------------------------------------|
| `GSE231858_norm_counts_TPM_GRCh38.p13_NCBI.tsv.gz` | Normalized RNA-seq counts in TPM format.                 |
| `Human.GRCh38.p13.annot.tsv.gz`            | Human genome annotation file for GRCh38.p13 reference.    |


---

## Setup and Usage

Follow these steps to use the script:

1. **Navigate to the Project Directory**  
   Move into the directory where the script is located:
   ```bash
   cd 7.collab-data
2. **Make the script executable**
    Update the script permissions:
    ```bash
    chmod +x get_geo_data.sh
3. **Run the script**
    Execute the script to download the data:
    ```bash
    ./get_geo_data.sh
