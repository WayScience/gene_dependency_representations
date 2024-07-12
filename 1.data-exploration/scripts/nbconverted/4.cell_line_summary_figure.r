
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(cowplot))
library(tidyverse)

#Fixed path names
# Set i/o paths and files
data_dir <- file.path("0.data-download/data/")
fig_dir <- file.path("1.data-exploration/figures")

model_input_file <- file.path(data_dir, "Model.csv")
crispr_input_file <- file.path(data_dir, "CRISPRGeneEffect.csv")

figure_output_file <- file.path(fig_dir, "age_and_ped_model_distributions_test.png")

# Set figure sizes
text_size = 9

# Process dataset
model_df <- read_csv(
    model_input_file,
    col_types = readr::cols(
        .default = "c",
        Age = "d"
    )
)

print(dim(model_df))

crispr_df <- readr::read_csv(
    crispr_input_file,
    col_types = readr::cols(
        .default = "d",
        ModelID = "c"
    )
)

print(dim(crispr_df))

# Get common depmap identifiers
common_depmap_ids <- intersect(model_df$ModelID, crispr_df$ModelID)

# Subset the model dataframe to only those we have dependency data for
model_df <- model_df %>%
    dplyr::filter(ModelID %in% common_depmap_ids)

# The updated dimensions should be the same as crispr_df
print(dim(model_df))

# Show model_df
head(model_df, 3)

colnames(model_df)

age_distrib_gg = (
    ggplot(model_df, aes(x = Age))
    + geom_histogram()
    + theme_bw()
    + theme(
        axis.text = element_text(size = text_size),
        axis.title = element_text(size = text_size + 1)
    )
)

age_distrib_gg

# Subset to pediatric cancers only
ped_model_df <- model_df %>%
    dplyr::filter(AgeCategory == "Pediatric")

rev(sort(table(ped_model_df$OncotreePrimaryDisease)))

disease_type_recode <- ped_model_df$OncotreePrimaryDisease %>%
    dplyr::recode(
        `Diffuse Glioma` = "Other/Rare (13 unique)",
        `Epithelioid Sarcoma` = "Other/Rare (13 unique)",
        `Melanoma` = "Other/Rare (13 unique)",
        `Myeloproliferative Neoplasms` = "Other/Rare (13 unique)",
        `Ovarian Epithelial Tumor` = "Other/Rare (13 unique)",
        `Ovarian Germ Cell Tumor` = "Other/Rare (13 unique)",
        `Undifferentiated Pleomorphic Sarcoma/Malignant Fibrous Histiocytoma/High-Grade Spindle Cell Sarcoma` = "Other/Rare (13 unique)",
        `Hepatoblastoma` = "Other/Rare (13 unique)",
        `Renal Cell Carcinoma` = "Other/Rare (13 unique)",
        `Retinoblastoma` = "Other/Rare (13 unique)",
        `Rhabdoid Cancer` = "Other/Rare (13 unique)",
        `Synovial Sarcoma` = "Other/Rare (13 unique)",
        `T-Lymphoblastic Leukemia/Lymphoma` = "Other/Rare (13 unique)",
        `B-Lymphoblastic Leukemia/Lymphoma` = "B-ALL"
    )

ped_model_df <- ped_model_df %>%
    dplyr::mutate(disease_type_recoded = disease_type_recode)

ped_model_df$disease_type_recoded <- factor(
    ped_model_df$disease_type_recoded,
    levels = names(sort(rev(table(ped_model_df$disease_type_recoded))))
)

cancer_type_distrib_gg = (
    ggplot(ped_model_df, aes(x = disease_type_recoded))
    + geom_bar(aes(fill = Sex), position = "stack")
    + coord_flip()
    + theme_bw()
    + geom_text(
        stat = "count",
        aes(label = after_stat(count)),
        vjust = 0.5,
        hjust = -0.25,
        size = 3
    )
    + scale_fill_manual(
        values = c(
            "Male" = "#90CAF9",
            "Female" = "pink",
            "Unknown" = "black"
        )
    )
    + ylim(c(0, 40))
    + theme(
        axis.text = element_text(size = text_size),
        axis.title = element_text(size = text_size + 1),
        legend.text = element_text(size = text_size - 2),
        legend.title = element_text(size = text_size - 1),
        legend.position = c(0.75, 0.3),
        legend.key.size = unit(0.3, 'cm')
    )
    + labs(y = "Pediatric Count", x = "")
    + guides(fill = guide_legend(override.aes = list(size = 0.5)))
)

cancer_type_distrib_gg

full_gg <- cowplot::plot_grid(
    age_distrib_gg,
    cancer_type_distrib_gg,
    labels = c("A", "B"),
    ncol = 2,
    rel_widths = c(0.45, 1)
)

ggsave(figure_output_file, full_gg, width = 4.75, height = 1.75, dpi = 500)

full_gg
