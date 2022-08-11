#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pathlib
import numpy as np
import pandas as pd
sys.path.insert(0, ".././0.data-download/scripts/")
from data_loader import load_train_test_data, load_data

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, gcf

from sklearn.decomposition import PCA
from tensorflow import keras

from vae import VAE

from keras.models import Model, Sequential
import random as python_random
import tensorflow as tf
import seaborn as sns;sns.set_theme(color_codes=True)
import random


# In[2]:


random.seed(18)
print(random.random())


# In[3]:


random.seed(18)
print(random.random())


# In[4]:


# load the data
data_directory = pathlib.Path("../0.data-download/data")
dfs = load_train_test_data(data_directory, train_or_test = "all")
train_init = dfs[0]
test_init = dfs[1]
gene_stats = dfs[2]


# In[5]:


# drop the string values
train_df = train_init.drop(columns= ["DepMap_ID", "age_and_sex"])
test_df = test_init.drop(columns= ["DepMap_ID", "age_and_sex"])


# In[6]:


# subsetting the genes 
# create dataframe containing the 1000 genes with the largest variances and their corresponding gene label and extract the gene labels
largest_var_df = gene_stats.nlargest(1000, "variance")
gene_list = largest_var_df["gene_ID"].tolist()
gene_list

# create new training and testing dataframes that contain only the corresponding genes
subset_train_df = train_df.filter(gene_list, axis = 1)
subset_test_df = test_df.filter(gene_list, axis = 1)


# In[7]:


print(subset_train_df.shape)
subset_train_df.head(3)


# In[8]:


print(subset_test_df.shape)
subset_test_df.head(3)


# In[9]:


encoder_architecture = []
decoder_architecture = []


# In[10]:


# These optimal parameter values were fetched by running "optimize_hyperparameters.py" and then running "fetch_hyper_params.ipynb"
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


# In[11]:


cp_vae.train(x_train = subset_train_df, x_test = subset_test_df)


# In[12]:


# display training history
history_df = pd.DataFrame(cp_vae.vae.history.history)

# save the training history as a .csv
hist_dir = pathlib.Path("./results/beta_vae_training_history.csv")
history_df.to_csv(hist_dir, index = False)


# In[13]:


# plot and save the figure
save_path = pathlib.Path('../1.data-exploration/figures/training_curve.png')

plt.figure(figsize=(6, 5), dpi = 500)
plt.plot(history_df["loss"], label="Training data")
plt.plot(history_df["val_loss"], label="Validation data")
plt.ylabel("MSE + KL Divergence")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(save_path)
plt.show()


# In[14]:


cp_vae.vae
cp_vae.vae.evaluate(subset_test_df)


# In[15]:


encoder = cp_vae.encoder_block["encoder"]
decoder = cp_vae.decoder_block["decoder"]


# In[16]:


data_dir = "../0.data-download/data/"
datafs = load_data(data_dir, adult_or_pediatric = "all")
dependency_df = datafs[1]
sample_df = datafs[0]


# In[17]:


train_init['train_or_test'] = train_init.apply(lambda _: 'train', axis=1)
test_init['train_or_test'] = test_init.apply(lambda _: 'test', axis=1)


# In[18]:


# create a data frame of both test and train gene dependency data sorted by top 1000 highest gene variances
concat_frames = [train_init, test_init]
train_and_test = pd.concat(concat_frames).reset_index(drop=True)
train_and_test[["age_category", "sex"]] = train_and_test.age_and_sex.str.split(pat='_',expand=True)
train_and_test_subbed = train_and_test.filter(gene_list, axis = 1)
metadata_holder = []
metadata_holder = pd.DataFrame(metadata_holder)
metadata = metadata_holder.assign(
    DepMap_ID = train_and_test.DepMap_ID.astype(str), 
    age_category = train_and_test.age_category.astype(str), 
    sex = train_and_test.sex.astype(str), 
    train_or_test = train_and_test.train_or_test.astype(str)
    )
metadata


# In[19]:


from cv2 import DFT_COMPLEX_INPUT


latent_complete = np.array(encoder.predict(train_and_test_subbed)[2])
latent_df = pd.DataFrame(latent_complete)
latent_df_dir = pathlib.Path('./results/latent_df.csv')
latent_df.to_csv(latent_df_dir)


# In[20]:


age_category = metadata.pop("age_category")
sex = metadata.pop("sex")
train_test = metadata.pop("train_or_test")


# In[21]:


# display clustered heatmap of coefficients
lut_pal = sns.cubehelix_palette(age_category.unique().size, light=.9, dark=.1, reverse=True, start=1, rot=-2)
put_pal = sns.cubehelix_palette(sex.unique().size)
mut_pal = sns.color_palette("hls", train_test.unique().size)

lut = dict(zip(age_category.unique(), lut_pal))
put = dict(zip(sex.unique(), put_pal))
mut = dict(zip(train_test.unique(), mut_pal))

row_colors1 = age_category.map(lut)
row_colors2 = sex.map(put)
row_colors3 = train_test.map(mut)

network_node_colors = pd.DataFrame(row_colors1).join(pd.DataFrame(row_colors2).join(pd.DataFrame(row_colors3)))

sns.set(font_scale=1.4)
g = sns.clustermap(latent_df, method='ward', figsize=(10, 20), row_colors= network_node_colors, yticklabels=False, dendrogram_ratio=(.1,.04), cbar_pos=(1,.2, 0.02, .6))


xx = []
for label in age_category.unique():
    x = g.ax_row_dendrogram.bar(0, 0, color=lut[label], label=label, linewidth=0)
    xx.append(x)
# add the legend
legend3 = plt.legend(xx, age_category.unique(), loc="center", title='age category', ncol=2, bbox_to_anchor=(1.3, .55), bbox_transform=gcf().transFigure)


yy = []
for label in sex.unique():
    y = g.ax_row_dendrogram.bar(0, 0, color=put[label], label=label, linewidth=0)
    yy.append(y)  
# add the second legend
legend4 = plt.legend(yy, sex.unique(), loc="center", title='sex', ncol=3, bbox_to_anchor=(1.3, .5), bbox_transform=gcf().transFigure)
plt.gca().add_artist(legend3)


zz = []
for label in train_test.unique():
    z = g.ax_row_dendrogram.bar(0, 0, color=mut[label], label=label, linewidth=0)
    zz.append(z)
# add the third legend
legend5 = plt.legend(zz, train_test.unique(), loc="center", title='train or test', ncol=2, bbox_to_anchor=(1.3, .45), bbox_transform=gcf().transFigure)
plt.gca().add_artist(legend4)


# save the figure
heat_save_path = pathlib.Path('../1.data-exploration/figures/heatmap.png')
plt.savefig(heat_save_path, bbox_inches = 'tight', dpi=600)

