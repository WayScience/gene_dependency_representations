#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pathlib
import numpy as np
import pandas as pd
import plotnine as p9
import matplotlib.pyplot as plt 
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, "../0.data-download/scripts/")
from data_loader import load_data, load_train_test_data


# In[2]:


# data path
data_directory = "../0.data-download/data/"


# In[3]:


# load the training data 
data = load_train_test_data(data_directory, train_or_test = "all", stats=False)
dfs = data[0]
dfs_test = data[1]


# In[4]:


# set a unique dataframe that can be appended from
training_df_age = dfs

# group by age and create new dataframes that can be appended to
groups = training_df_age.groupby("age_and_sex")
adult_dependency_df = pd.DataFrame()
ped_dependency_df = pd.DataFrame()
for name, training_df_age in groups:
    
    # append rows that contain Adult samples (male or female) to the new adult dependency dataframe
    if name == "Adult_Male" or name == "Adult_Female" or name == "Adult_nan":
        adult_dependency_df = adult_dependency_df.append(training_df_age)
        adult_dependency_df = adult_dependency_df.reset_index(drop=True)
        
    # append rows that contain Pediatric samples (male ore female) to the new pediatric dataframe
    else :
        ped_dependency_df = ped_dependency_df.append(training_df_age)
        ped_dependency_df = ped_dependency_df.reset_index(drop=True)           


# In[5]:


# set a unique dataframe that can be appended from
training_df_sex = dfs

# group by sex and create new dataframes to be appended to 
groups_sex = training_df_sex.groupby("age_and_sex")
male_dependency_df = pd.DataFrame()
female_dependency_df = pd.DataFrame()
for name, training_df_sex in groups_sex:
    
    # append rows that contain Male samples (Adult or Pediatric) to the new male dependency dataframe and filter out samples that contain no gender info
    if name == "Adult_Male" or name == "Pediatric_Male" and name != "Pediatric_nan" and name != "Adult_nan":
        male_dependency_df = male_dependency_df.append(training_df_sex)
        male_dependency_df = male_dependency_df.reset_index(drop=True)
        
    # append rows that contain Female samples (Adult or Pediatric) to the new female dependency dataframe and filter out samples that contain no gender info
    elif name == "Adult_Female" or name == "Pediatric_Female" and name != "Pediatric_nan" and name != "Adult_nan":
        female_dependency_df = female_dependency_df.append(training_df_sex)
        female_dependency_df = female_dependency_df.reset_index(drop=True)          


# In[6]:


print(adult_dependency_df.shape)
adult_dependency_df.head(3)


# In[7]:


print(ped_dependency_df.shape)
ped_dependency_df.head(3)


# In[8]:


print(male_dependency_df.shape)
male_dependency_df.head(3)


# In[9]:


print(female_dependency_df.shape)
female_dependency_df.head(3)


# In[10]:


#drop the string values from all dataframes
adult_dependency_df_float = adult_dependency_df.drop(columns= ["DepMap_ID", "age_and_sex"])
ped_dependency_df_float =ped_dependency_df.drop(columns= ["DepMap_ID", "age_and_sex"])

male_dependency_df_float = male_dependency_df.drop(columns= ["DepMap_ID", "age_and_sex"])
female_dependency_df_float =female_dependency_df.drop(columns= ["DepMap_ID", "age_and_sex"])

dependency_df = dfs.drop(columns= "age_and_sex")
dependency_df = dependency_df.set_index("DepMap_ID")


# In[11]:


# t_test comparing gene dependencies in adult vs pediatric samples
t_test = ttest_ind(adult_dependency_df_float, ped_dependency_df_float)
t_test = pd.DataFrame(t_test).T
t_test.columns = ['t_stat', 'p_value']
print(t_test.shape)
t_test.head(3)


# In[12]:


# t_test comparing gene dependencies in male vs female samples
t_test_sex = ttest_ind(male_dependency_df_float, female_dependency_df_float)
t_test_sex = pd.DataFrame(t_test_sex).T
t_test_sex.columns = ['t_stat', 'p_value']
print(t_test_sex.shape)
t_test_sex.head(3)


# In[14]:


print(dependency_df.shape)
dependency_df.head(3)


# In[15]:


# calculate variance of each gene then send the results plus the gene info into a new dataframe
variance = dependency_df.var()
variance_list = variance.tolist()
column_names = ['variance']
variance_df = pd.DataFrame(variance, columns= column_names)
variance_df = variance_df.sort_index(ascending=True).reset_index()
variance_df = variance_df.rename(columns= {"index":"gene_ID"})
print(variance_df.shape)
variance_df.head(3)


# In[16]:


# finding the smallest gene variation out of the 1000 largest gene variations to set the top 1000 gene variances threshold
n = variance_df["variance"].nlargest(1000)
variance_theshold = n.min().astype(float)

# plotting variance density chart and marking the 1000 largest gene variation cutoff
variance_density_plot = (p9.ggplot(variance_df, p9.aes(x = 'variance')) 
                         + p9.geom_density() 
                         + p9.geom_vline(xintercept = variance_theshold, linetype = "dashed", color = "red") 
                         + p9.theme(figure_size = (10,6))
                         )

# save the figure
density_path = pathlib.Path("./figures/variance_density_plot.png")
variance_density_plot.save(density_path)
variance_density_plot


# In[17]:


# first create new dataframe containing gene info as well as both adult-pediatric and male-female t-test results and variance results
df = variance_df.assign(ttest_A_vs_P = t_test.t_stat.astype(float), ttest_M_vs_F = t_test_sex.t_stat.astype(float)) 

# and save the new dataframe as a .csv
testing_df_output = pathlib.Path("../0.data-download/data/genes_variances_and_t-tests_df.csv")
df.to_csv(testing_df_output, index = False)
print(df.shape)
df.head(3)


# In[18]:


# plot adult-pediatric ttest versus variance
A_vs_P_by_variance_plot = (p9.ggplot(data = df, mapping = p9.aes(x = 'variance', y = 'ttest_A_vs_P')) 
                           + p9.geom_point(size = 0.4, alpha = 0.1, color = 'blue') 
                           + p9.theme(figure_size = (10,7))
                           )

# save the figure
adult_vs_pediatric_path = pathlib.Path("./figures/adult-pediatric_ttest_vs_variance.png")
A_vs_P_by_variance_plot.save(adult_vs_pediatric_path)
A_vs_P_by_variance_plot


# In[19]:


# plot male-female ttest versus gene variance
M_vs_F_by_variance_plot = (p9.ggplot(data = df, mapping = p9.aes(x = 'variance', y = 'ttest_M_vs_F')) 
                           + p9.geom_point(size = 0.4, alpha = 0.1, color = 'blue') 
                           + p9.theme(figure_size = (10,7))
                           )

# save the figure
male_vs_female_path = pathlib.Path("./figures/male-female_ttest_vs_variance.png")
M_vs_F_by_variance_plot.save(male_vs_female_path)
M_vs_F_by_variance_plot

