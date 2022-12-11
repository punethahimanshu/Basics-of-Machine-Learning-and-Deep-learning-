#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the dataset from pycaret repository
from pycaret.datasets import get_data
anomaly = get_data('anomaly')
#import anomaly detection module


# In[2]:


#import anomaly detection module
from pycaret.anomaly import *
#intialize the setup
exp_ano = setup(anomaly)


# In[3]:


anomaly


# In[5]:


## creating a model 1
iforest=create_model('iforest')
## plotting a model 1
plot_model(iforest)


# In[11]:


## creating a model 2
knn =create_model('knn')
## plotting a model 2
plot_model(knn)


# In[13]:


# generate predictions using trained model
knn_predictions = predict_model(knn, data = anomaly)


# In[14]:


knn_predictions


# In[ ]:




