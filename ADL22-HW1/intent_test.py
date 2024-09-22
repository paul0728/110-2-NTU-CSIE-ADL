#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from keras.models import load_model
from argparse import ArgumentParser, Namespace
import os
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--test-dataset-path', help='Path to the test dataset.', default='./data/intent/test.json')
    parser.add_argument('--preprocess-dir', help='Directory to the preprocessed files.', default='./prep/intent/')
    parser.add_argument('--pred-file-path', help='Path to the create predict file.')
    parser.add_argument('--max-words', help='Vocabulary size of train/validation/test dataset.', type=int, default=10000)
    parser.add_argument('--max-len', help='Maximum length of a sequence.', type=int, default=28)
    parser.add_argument('--model', help='Network model to test.', choices=['RNN', 'GRU', 'LSTM', 'BILSTM', 'CNN+BILSTM','2BILSTM','CNN+2BILSTM'], default='2BILSTM')
    parser.add_argument('--ckpt-load-path', help='Path to the load checkpoint file.')
    '''
    args = parser.parse_args(args=['--test-dataset-path', './data/intent/test.json', \
                                   '--preprocess-dir', './prep/intent/', \
                                   '--pred-file-path', None,\
                                   '--max-words','10000',\
                                   '--max-len','28',\
                                   '--model', '2BILSTM',\
                                   '--ckpt-load-path',None
                                  ])
    '''
    args = parser.parse_args()
    
    return args

args = parse_args()
# print(args.test_dataset_path)
# print(args.pred_file)

filename=f"{args.model}_nooov_mw_{args.max_words}_ml_{args.max_len}"
print(filename)


if args.pred_file_path==None:
    args.pred_file_path=f"./results/intent/{filename}"
    
if args.ckpt_load_path==None:
    args.ckpt_load_path=f"./ckpt/intent/{filename}"    
    
    
print(args.pred_file_path)


# # Load data

# In[9]:


index2label = np.load(f'{args.preprocess_dir}index2label.npy', allow_pickle='TRUE').item()
label2index = np.load(f'{args.preprocess_dir}label2index.npy', allow_pickle='TRUE').item()

test_sequences = np.load(f'{args.preprocess_dir}test_sequences.npy', allow_pickle='TRUE')

test=pd.read_json (args.test_dataset_path)

index2label


# In[10]:


label2index


# In[11]:


test_sequences


# # Load model

# In[12]:


# 載入模型
model = load_model(args.ckpt_load_path)


# In[13]:


model.summary()


# # Inference and create csv file

# In[14]:


result=np.argmax(model.predict(test_sequences),axis=1)
# print(result)


def index2label_(x):
    return index2label[x]
test["intent"]=list(map(index2label_,result))

os.makedirs('./results', exist_ok=True)
os.makedirs('./results/intent', exist_ok=True)

result=test.loc[:,["id","intent"]]
result.to_csv(args.pred_file_path,index=False) 
result


# In[ ]:




