#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pathlib
import numpy as np
import pandas as pd
sys.path.insert(0, "../0.data-download/scripts/")
from data_loader import load_data


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.decomposition import PCA
from tensorflow import keras

from vae import VAE

from tensorflow.keras.models import Model, Sequential
import seaborn
import random as python_random
import tensorflow as tf


# In[2]:


# load the data 
data_directory = "../0.data-download/data/"
dfs = load_data(data_directory, adult_or_pediatric = "all")
dependency_df = dfs[1]
sample_df = dfs[0]


# In[3]:


dependency_df


# In[4]:


dependency_df1 = dependency_df.drop(axis=0, columns= "DepMap_ID")
dependency_df1.reset_index(drop=True, inplace=True)
print(dependency_df1.shape)
dependency_df1.head(3)


# In[5]:


encoder_architecture = [250]
decoder_architecture = [250]


# In[6]:


cp_vae = VAE(
    input_dim= dependency_df1.shape[1],
    latent_dim=90,
    batch_size=32,
    encoder_batch_norm=True,
    epochs=58,
    learning_rate=0.0001,
    encoder_architecture=encoder_architecture,
    decoder_architecture=decoder_architecture,
    beta=0.06,
    verbose=True,
)

cp_vae.compile_vae()


# In[ ]:


cp_vae.train(x_train=dependency_df, x_test=dependency_df)

