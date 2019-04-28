
# coding: utf-8

# In[1]:


from __future__ import unicode_literals, division
import re
import sys
from tqdm import tqdm_notebook


# In[2]:


# ! pip install tqdm --user 


# In[3]:


import pandas as pd
import numpy as np
import spacy

import tensorflow as tf


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (10,10)
sns.set()


# In[5]:



from embedding import get_embedding
from config import Config
from data_utils import tokenize_sent


# ## POS Tags

# In[7]:


nlp = spacy.load('en_core_web_sm')
def get_pos_tags(word_list):
    tags = []
    sent = ' '.join(word_list)
    doc = nlp(sent)
    [tags.append(token.tag_) for token in doc]
    return tags


# In[30]:


get_pos_tags('i druuink tea'.split())


# # Get configurations

# In[9]:


config = Config()
max_len = 70
embedding_size = config.embedding_size


# In[10]:


path = '../train.csv'


# In[11]:


data = pd.read_csv(path)
data = data[:100000]


# In[12]:


columns = data.columns


# In[13]:


# columns


# In[14]:


# data.head(5)


# In[15]:


data.isna().sum()


# # sequence lengths

# In[16]:


data['seql_one'] = data['question1'].apply(lambda row: len(tokenize_sent(str(row).lower())))
data['seql_two'] = data['question2'].apply(lambda row: len(tokenize_sent(str(row).lower())))


# In[17]:


data = data.loc[(data.seql_one != 0)  & (data.seql_two!=0) ]


# In[18]:


# data.columns


# # spliting data to train and dev

# 80% train data and 20% test data

# In[19]:


split_ratio = 0.8


# In[20]:


l = len(data)


# In[21]:


train_length = int(l*split_ratio)


# In[31]:


print(data.groupby('is_duplicate').count())


# # word2vec features
# 

# In[23]:


def get_sent_embedding(sent):
    sent = str(sent).lower().strip()
    vector = []
    counter = 0
    vector = [get_embedding(w) for w in tokenize_sent(sent)]
    return vector


# In[24]:


# get_sent_embedding('dsfhgdsj sajkfgksdgf')


# In[25]:


len(get_sent_embedding('i love reading'))


# In[26]:


train_data = data[:train_length]


train_data['vec_1'] = train_data['question1'].apply(lambda row: get_sent_embedding(row))
train_data['vec_2'] = train_data['question2'].apply(lambda row: get_sent_embedding(row))

train_data_all = list(zip(train_data.vec_1, train_data.vec_2, 
                 train_data.seql_one, train_data.seql_two, 
                 train_data.is_duplicate))


# In[27]:


dev_data = data[train_length:]


dev_data['vec_1'] = dev_data['question1'].apply(lambda row: get_sent_embedding(row))
dev_data['vec_2'] = dev_data['question2'].apply(lambda row: get_sent_embedding(row))

dev_data_all = list(zip(dev_data.vec_1, dev_data.vec_2, 
               dev_data.seql_one, dev_data.seql_two, 
               dev_data.is_duplicate))


# In[28]:


print(len(dev_data))


# # Building and training Model

# In[106]:


from model import Model
model = Model()


# In[107]:

model.config.n_hidden = 64
model.config.lr = 0.01
model.config.batch_size = 64
model.config.n_layers =3
model.config.keep_prob = 0.7

# In[ ]:


tf.reset_default_graph()
model.build()


# In[ ]:


model.train(train_data=train_data_all, dev_data=dev_data_all)


# ## Restore Model
