# In[1]:


suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(cowplot))
suppressPackageStartupMessages(library(reticulate))


# In[2]:


#Load data
data_dir <- file.path("../4.gene_expression_signatures/results")

results_file <- file.path(data_dir, "combined_z_matrix_gsea_results.csv")
gsea_results_df <- readr::read_csv(
    results_file,
)


# In[3]:


# Prepare the data
# Filter for the single highest ES score for each model at each dimension
max_es_df <- gsea_results_df %>%
  group_by(model, full_model_z) %>%
  summarize(max_es = max(`gsea es score`, na.rm = TRUE)) %>%
  ungroup()


# In[4]:


# Plot the data
latent_plot <- ggplot(max_es_df, aes(x = factor(full_model_z), y = max_es, color = model, fill = model)) +
  geom_point(size = 3, shape = 21) +  # Points on the line
  geom_smooth(aes(group = model), method = "loess", se = TRUE, size = 1, alpha = 0.1) +  # Trend line with shading 
  scale_color_manual(name = "Algorithm",
                     values = c(
                      "#e41a1c", 
                      "#377eb8", 
                      "#4daf4a", 
                      "#e4e716", 
                      "#984ea3", 
                      "#ff7f00"),
                     labels = c(
                      "pca" = "PCA", 
                      "ica" = "ICA", 
                      "nmf" = "NMF", 
                      "vanillavae" = "VAE",
                      "betavae" = "BVAE", 
                      "betatcvae" = "BTCVAE"
                      )) +
  scale_fill_manual(name = "Algorithm",
                    values = c(
                      "#e41a1c", 
                      "#377eb8", 
                      "#4daf4a", 
                      "#e4e716", 
                      "#984ea3", 
                      "#ff7f00"),
                    labels = c(
                      "pca" = "PCA", 
                      "ica" = "ICA", 
                      "nmf" = "NMF", 
                      "vanillavae" = "VAE",
                      "betavae" = "BVAE", 
                      "betatcvae" = "BTCVAE")) +
  labs(x = "Latent Dimensions", y = "Highest ES Score", title = "Highest ES Score by Latent Dimension for Each Model") +
  theme(legend.position = "right")


# In[5]:


# Save the plot with custom dimensions
ggsave("latent_plot.png", plot = latent_plot, width = 10, height = 8, units = "in")

