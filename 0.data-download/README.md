# Description of the Data

## DepMap Public 22Q4

This DepMap release contains data from in vitro studies of genetic dependencies in cancer cell lines using CRISPR/Cas9 loss-of-function screens from project Achilles, as well as genomic characterization data from the CCLE project.

Data resource:
https://depmap.org/portal/download/all/

## Files used in our multivariate gene dependency project

### CRISPRGeneDependency.csv

This data comprises scores from individual CRISPR gene knockout screens in cancer cell lines for every gene across many different ages and cancer types.
The values represent gene dependency probability estimates for cell survival and growth for all models in the integrated gene effect.
The higher the score, the greater the gene dependency. 

The probability estimate is derived from the CRISPRGeneEffect estimates for all models which were introduced to an algorithm, Chronos, to infer gene knockout fitness effects based on an explicit model of cell proliferation dynamics after CRISPR gene knockout.

Columns (18,333): gene

Rows (1,087): the ModelID

### Model.csv

This file gives details on the cell lines, type of cancer, sex, age, and unique IDs of the patient.

Columns (29): description of cancer, and patient's biological and ID information

Rows (1,841): the ModelID
