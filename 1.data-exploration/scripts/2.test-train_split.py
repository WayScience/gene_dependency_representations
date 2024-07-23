#!/usr/bin/env python
# coding: utf-8

# In[13]:


import sys
import pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, "../utils/")
from data_loader import load_data
from sklearn.model_selection import train_test_split
import random


# In[14]:


def scale_dataframe(df: pd.DataFrame):
    """
    Scales the gene effect data columns of a DataFrame to a 0-1 range.
    The first column (ID) and the last two columns (age and sex) are not scaled.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The scaled DataFrame.
    """
    col_num = df.shape[1]
    df_to_scale = df.iloc[:, 1:col_num-1]
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_df = scaler.fit_transform(df_to_scale)
    
    scaled_df = pd.DataFrame(scaled_df)
    scaled_df.insert(0, df.columns[0], df[df.columns[0]])
    scaled_df.insert(col_num-1, df.columns[col_num-1], df[df.columns[col_num-1]])
    scaled_df.columns = df.columns
    
    return scaled_df


# In[15]:


def save_dataframe(df, file_path: pathlib.Path):
    """
    Saves a DataFrame to a specified file path.

    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    file_path (str): The file path to save the DataFrame.
    """
    df = df.reset_index(drop=True)
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved to {file_path}. Shape: {df.shape}")
    print(df.head(3))


# In[16]:


random.seed(18)
print(random.random())


# In[17]:


# load all of the data
data_directory = "../0.data-download/data/"
model_df, effect_df = load_data(data_directory, adult_or_pediatric="all")


# In[18]:


# verifying that the ModelIDs in model_df and effect_df are alligned
model_df["ID_allignment_verify"] = np.where(
    effect_df["ModelID"] == model_df["ModelID"], "True", "False"
)
verrify = len(model_df["ID_allignment_verify"].unique())
print(model_df["ID_allignment_verify"])
print(
    f"There is {verrify} output object contained in the ID_allignment_verify column \n"
)


# In[19]:


# assign 'AgeCategory' and 'Sex' columns to the effect dataframe as a single column
presplit_effect_df = effect_df.assign(
    age_and_sex=model_df.AgeCategory.astype(str) + "_" + model_df.Sex.astype(str)
)
presplit_effect_df


# In[20]:


groups = model_df.groupby("AgeCategory")
df_list = []
for name, df in groups:

    # only looking for samples that contain Adult or Pediatric information
    if name == "Adult" or name == "Pediatric":
        df_list.append(df)

# merge sample dataframes through concatentation and reorganize so that ModelIDs are in alphabetical order
new_df = pd.concat(df_list, axis=0)
new_df = new_df.set_index("ModelID")
new_df = new_df.sort_index(ascending=True)
new_df = new_df.reset_index()


# In[21]:


# creating a list of ModelIDs that correlate to pediatric and adult samples
PA_effect_IDs = new_df["ModelID"].tolist()

PA_IDs = set(PA_effect_IDs) & set(presplit_effect_df["ModelID"].tolist())

# creating a new gene effect data frame containing correlating ModelIDs to the filtered sample info IDs
PA_effect_df = presplit_effect_df.loc[
    presplit_effect_df["ModelID"].isin(PA_IDs)
].reset_index(drop=True)


# In[22]:


# split the data based on age category and sex
train_df, testandvalidation_df = train_test_split(
    PA_effect_df, test_size=0.3, stratify=PA_effect_df.age_and_sex
)
train_df.reset_index(drop=True,inplace=True)
testandvalidation_df.reset_index(drop=True,inplace=True)
test_df, val_df = train_test_split(
    testandvalidation_df, test_size=0.5
)
test_df.reset_index(drop=True,inplace=True)
val_df.reset_index(drop=True,inplace=True)


# In[27]:


#save each dataframe
save_dataframe(train_df, pathlib.Path("../0.data-download/data/VAE_train_df.csv").resolve())
save_dataframe(test_df, pathlib.Path("../0.data-download/data/VAE_test_df.csv").resolve())
save_dataframe(val_df, pathlib.Path("../0.data-download/data/VAE_val_df.csv").resolve())

