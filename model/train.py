
# coding: utf-8

# In[1]:


from __future__ import unicode_literals, division
import re
import sys
from tqdm import tqdm_notebook


# In[2]:


# !/glob/intel-python/versions/2018u2/intelpython2/bin/pip install tqdm --user


# In[3]:


# !python2 -m spacy download en_core_web_sm --user


# In[10]:


import pandas as pd
import numpy as np
import spacy

import tensorflow as tf


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (10,10)
sns.set()


# In[12]:



# In[13]:


from embedding import get_embedding
from config import Config
from data_utils import tokenize_sent


# ## POS Tags

# In[14]:


nlp = spacy.load('en_core_web_sm')
def get_pos_tags(word_list):
    tags = []
    sent = ' '.join(word_list)
    doc = nlp(unicode(sent))
    [tags.append(token.tag_) for token in doc]
    return tags


# In[15]:


get_pos_tags('i druuink tea'.split())


# # Get configurations

# In[16]:


config = Config()
max_len = 70
embedding_size = config.embedding_size

config.summary_dir = config.PROJECT_HOME + "/results_one/"
config.save_dir = config.save_dir + "/models_weights_one/"




# get_sent_embedding('i love reading')


import pickle


# In[19]:


# data_dict = {'train_data_all':train_data_all,
# 'dev_data_all':dev_data_all}


# In[20]:


# with open('data.pickle', 'wb') as handle:
#     pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[21]:


# Load data (deserialize)
with open('data.pickle', 'rb') as handle:
    unserialized_data = pickle.load(handle)


# In[22]:


train_data_all = unserialized_data['train_data_all']
dev_data_all = unserialized_data['dev_data_all']


# # Building and training Model

# In[23]:


from model import Model
model = Model()


# In[24]:


model.config.n_hidden = 256


# In[30]:


model.config.lr = 0.01


# In[25]:


model.config.batch_size = 32
model.config.n_layers =3
model.config.keep_prob = 0.7
model.config.lambda_l2_reg = 0
model.config.gradient_clipping = False
model.config.triplet_loss=True


# In[26]:


tf.reset_default_graph()
model.build()


# In[ ]:


model.train(train_data=train_data_all, dev_data=dev_data_all)


