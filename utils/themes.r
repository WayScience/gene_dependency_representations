# Define colors and labels for the models
model_colors <- c(
  "pca" = "#e41a1c", 
  "ica" = "#377eb8", 
  "nmf" = "#4daf4a", 
  "vanillavae" = "#e4e716", 
  "betavae" = "#984ea3", 
  "betatcvae" = "#ff7f00"
)

model_labels <- c(
  "pca" = "PCA", 
  "ica" = "ICA", 
  "nmf" = "NMF", 
  "vanillavae" = "VAE",
  "betavae" = "BVAE", 
  "betatcvae" = "BTCVAE"
)

# Custom theme function
custom_theme <- function() {
  theme(
    legend.position = "right"
  )
