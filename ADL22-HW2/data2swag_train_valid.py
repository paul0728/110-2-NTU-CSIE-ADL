#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
from argparse import ArgumentParser, Namespace
import os  


# In[13]:


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--train_file', help='The input training data file (a text file).',default='./data/train.json')
    parser.add_argument('--validation_file', help='An optional input evaluation data file to evaluate the perplexity on (a text file).',default='./data/valid.json')
    parser.add_argument('--context_file', help='An optional input context data file to evaluate the perplexity on (a text file).',default='./data/context.json')
    

#     args = parser.parse_args(args=['--context_file', './data/context.json',\
#                                    '--test_file', './data/test.json'
#                                   ])    
    
    
    
    
    args = parser.parse_args()
    
    return args


args = parse_args()    
# print(args.context_file)
# print(args.test_file)
context=pd.read_json(args.context_file)
# print(len(context[0]))
# context[0][836][108]
# print(context[0][6043])
# context


# In[14]:


# print(context[0][697])


# In[16]:


# print(context[0][5391])


# In[4]:


# print(context[0][7436])


# In[17]:


train=pd.read_json(args.train_file)
train.iloc[229]


# In[7]:


valid=pd.read_json(args.validation_file)
valid


# # Data Fields(swag)
# 
# 

# video-id: identification
# 
# fold-ind: identification
# 
# startphrase: the context to be filled
# 
# sent1: the first sentence
# 
# sent2: the start of the second sentence (to be filled)
# 
# gold-source: generated or comes from the found completion
# 
# ending0: first proposition
# 
# ending1: second proposition
# 
# ending2: third proposition
# 
# ending3: fourth proposition
# 
# label: the correct proposition
# 
# 只要合成id(為了predict用),sent,ending,label(load 進去時要有) 即可

# In[18]:


def pagraph2ending(paragraphs):
    ending0=[]
    ending1=[]
    ending2=[]
    ending3=[]
    for p in paragraphs:
        for i in range(4):
            if i==0:
                ending0.append(context[0][p[i]])
            elif i==1:
                ending1.append(context[0][p[i]])
            elif i==2:
                ending2.append(context[0][p[i]])
            else:
                ending3.append(context[0][p[i]])
    return ending0,ending1,ending2,ending3

def relevant2label(paragraphs,relevant):
    label=[]
    for p,r in zip(paragraphs,relevant):
        for i in range(4):
            if p[i]==r:
                label.append(i)
                break
    return label
            



def data2swagformat(data):
    ending0,ending1,ending2,ending3=pagraph2ending(data['paragraphs'])
    if 'relevant' not in data:
        data_swag_format={
            'id':data['id'],
            'sent1':data['question'],
            'sent2':["" for i in range(len(data))],
            'ending0':ending0,
            'ending1':ending1,
            'ending2':ending2,
            'ending3':ending3,
            'label':[0 for i in range(len(data))]
        }

        
    else:
        label=relevant2label(data['paragraphs'],data['relevant'])
        data_swag_format={
            'id':data['id'],
            'sent1':data['question'],
            'sent2':["" for i in range(len(data))],
            'ending0':ending0,
            'ending1':ending1,
            'ending2':ending2,
            'ending3':ending3,
            'label':label
        }
    return data_swag_format

    
    


# In[19]:


# 建立資料夾
os.makedirs('./preprocess/mc/', exist_ok=True)  


# In[21]:


train_swag_format=data2swagformat(train)
print(len(train_swag_format["sent1"]))
print(len(train_swag_format["sent2"]))
print(len(train_swag_format["ending0"]))
print(len(train_swag_format["ending1"]))
print(len(train_swag_format["ending2"]))
print(len(train_swag_format["ending3"]))
print(len(train_swag_format["label"]))

train_swag_format = pd.DataFrame(train_swag_format)
train_swag_format.to_csv('./preprocess/mc/train_swag_format.csv',index=False)
train_swag_format['ending3'][229]
train_swag_format


# In[11]:


valid_swag_format=data2swagformat(valid)
valid_swag_format = pd.DataFrame(valid_swag_format)
valid_swag_format.to_csv('./preprocess/mc/valid_swag_format.csv',index=False)

