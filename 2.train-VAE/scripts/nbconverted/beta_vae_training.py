#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pathlib
import numpy as np
import pandas as pd
sys.path.insert(0, ".././0.data-download/scripts/")
from data_loader import load_train_test_data

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.decomposition import PCA
from tensorflow import keras

from vae import VAE

from keras.models import Model, Sequential
import seaborn
import random as python_random
import tensorflow as tf


# In[2]:


# load the data
data_directory = pathlib.Path("../0.data-download/data")
dfs = load_train_test_data(data_directory, train_or_test = "all")
train_init = dfs[0]
test_init = dfs[1]
gene_stats = dfs[2]


# In[3]:


# drop the string values
train_df = train_init.drop(columns= ["DepMap_ID", "age_and_sex"])
test_df = test_init.drop(columns= ["DepMap_ID", "age_and_sex"])


# In[4]:


# subsetting the genes 
# create dataframe containing the 1000 genes with the largest variances and their corresponding gene label and extract the gene labels
largest_var_df = gene_stats.nlargest(1000, "variance")
gene_list = largest_var_df["gene_ID"].tolist()
gene_list

# create new training and testing dataframes that contain only the corresponding genes
subset_train_df = train_df.filter(gene_list, axis = 1)
subset_test_df = test_df.filter(gene_list, axis = 1)


# In[5]:


print(subset_train_df.shape)
subset_train_df.head(3)


# In[6]:


print(subset_test_df.shape)
subset_test_df.head(3)


# In[7]:


# scale the data
def absolute_maximum_scale(series):
    return series / series.abs().max()


for col in subset_train_df.columns:
    subset_train_df[col] = absolute_maximum_scale(subset_train_df[col])
for col in subset_test_df.columns:
    subset_test_df[col] = absolute_maximum_scale(subset_test_df[col])


# In[8]:


encoder_architecture = []
decoder_architecture = []


# In[9]:


cp_vae = VAE(
    input_dim=subset_train_df.shape[1],
    latent_dim=10,
    batch_size=16,
    encoder_batch_norm=False,
    epochs=40,
    learning_rate=0.005,
    encoder_architecture=encoder_architecture,
    decoder_architecture=decoder_architecture,
    beta=1.6,
    lam=0,
    verbose=True,
)

cp_vae.compile_vae()


# In[10]:


cp_vae.train(x_train = subset_train_df, x_test = subset_test_df)


# In[11]:


# display training history
history_df = pd.DataFrame(cp_vae.vae.history.history)
history_df


# In[12]:


# plot and save the figure
save_path = pathlib.Path('../1.data-exploration/figures/training_curve.png')

plt.figure(figsize=(7, 5), dpi = 400)
plt.plot(history_df["loss"], label="Training data")
plt.plot(history_df["val_loss"], label="Validation data")
plt.ylabel("MSE + KL Divergence")
plt.xlabel("No. Epoch")
plt.legend()
plt.savefig(save_path)
plt.show()

