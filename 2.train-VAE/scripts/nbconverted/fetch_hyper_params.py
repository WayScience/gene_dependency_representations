#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json


# In[2]:


vis_datas = []
layers = ["asdf1"]
# layers = ['threelayer']
for layer in layers:
    vis_data = []
    rootdir = './parameter_sweep/' + layer
    for subdirs, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith("trial.json"):
                with open(subdirs + '/' + file, 'r') as json_file:
                    data = json_file.read()

                if 'loss' in json.loads(data)['metrics']['metrics']:
                    vis_data.append(json.loads(data))
                   
    vis_datas.append(vis_data)


# In[3]:


optimal_hyperparameters = [vis_datas[0][0]]


for i in range(len(vis_datas)):
    for j in range(len(vis_datas[i])):
        if vis_datas[i][j]['metrics']['metrics']['val_loss']['observations'][0]['value'] < optimal_hyperparameters[i]['metrics']['metrics']['val_loss']['observations'][0]['value']:
            optimal_hyperparameters[i] = vis_datas[i][j]


# In[4]:


for layer in optimal_hyperparameters:
    print('latent_dim:', layer['hyperparameters']['values']['latent_dim'])
    print('learning_rate:', layer['hyperparameters']['values']['learning_rate'])
    print('encoder_batch_norm:', layer['hyperparameters']['values']['encoder_batch_norm'])
    print('beta:', layer['hyperparameters']['values']['beta'])                      #added print beta statement
    print('batch_size:', layer['hyperparameters']['values']['batch_size'])
    print('epochs:', layer['hyperparameters']['values']['epochs'])
    print('loss:', layer['metrics']['metrics']['loss']['observations'][0]['value'])
    print('val_loss:', layer['metrics']['metrics']['val_loss']['observations'][0]['value'])
    print()


# In[5]:


# vis_data = vis_data[1:]

import hiplot as hip
data = [{
         'latent_dim': vis_data[idx]['hyperparameters']['values']['latent_dim'],
         'learning_rate': vis_data[idx]['hyperparameters']['values']['learning_rate'], 
         'beta': vis_data[idx]['hyperparameters']['values']['beta'], 
         'encoder_batch_norm': vis_data[idx]['hyperparameters']['values']['encoder_batch_norm'], 
         'batch_size': vis_data[idx]['hyperparameters']['values']['batch_size'],
         'epochs': vis_data[idx]['hyperparameters']['values']['epochs'], 
         'loss': vis_data[idx]['metrics']['metrics']['loss']['observations'][0]['value'][0],  
         'val_loss': vis_data[idx]['metrics']['metrics']['val_loss']['observations'][0]['value'][0], 
         } for idx in range(len(vis_datas[0]))] 


# In[6]:


cleaned_data = []
for q in range(len(data)):
    val_loss = data[q]['val_loss']
    loss = data[q]['loss']
    if val_loss < 300 and loss < 300:
        cleaned_data.append(data[q])


# In[7]:


# display HiPlot
hip.Experiment.from_iterable(cleaned_data).display()
# save as html
hip.Experiment.from_iterable(cleaned_data).to_html("./results/parameter_sweep_plot.html");

