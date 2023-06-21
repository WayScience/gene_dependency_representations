# Description of the Data

## DepMap Public 23Q2

This DepMap release contains data from in vitro studies of genetic dependencies in cancer cell lines using CRISPR/Cas9 loss-of-function screens from project Achilles, project Sanger, and several other datasets (e.g. CCLE)

Data resource:
https://depmap.org/portal/download/all/

The 23Q2 release notes are described here: https://forum.depmap.org/t/announcing-the-23q2-release/2518

## Files used in our multivariate gene dependency project

See the following resource for more information: https://forum.depmap.org/t/depmap-genetic-dependencies-faq/131

### CRISPRGeneEffect.csv

Integrated dataset processed by a joint Chronos run for the Achilles dataset (Avana library) combined with the Sanger dataset (KY library).
The two datasets were adjusted for by Chronos 2.0 and integrated using a ComBat-based algorithm called Harmonia.

Chronos adjusts for sgRNA efficacy, screen quality, differential cell growth rates, and copy number differences across cell models.

> Dempster, J.M., Boyle, I., Vazquez, F. et al. Chronos: a cell population dynamics model of CRISPR experiments that improves inference of gene fitness effects. Genome Biol 22, 343 (2021). https://doi.org/10.1186/s13059-021-02540-7

The dataset is also adjusted for gene knockout effects that occur on the same chromosome arm.

### CRISPRGeneDependency.csv

This data comprises scores from individual CRISPR gene knockout screens in cancer cell lines for every gene across many different ages and cancer types.
The values represent gene dependency probability estimates for cell survival and growth for all models in the integrated gene effect.
The higher the score, the greater the gene dependency. 

The probability estimate is derived from the CRISPRGeneEffect estimates.

Columns (17,932): gene
Rows (1,095): the ModelID

### Model.csv

This file gives details on the cell lines, type of cancer, sex, age, and unique IDs of the patient.

Columns (30): description of cancer, and patient's biological and ID information
Rows (1,865): the ModelID
