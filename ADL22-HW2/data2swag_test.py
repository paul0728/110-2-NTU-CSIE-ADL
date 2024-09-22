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
    parser.add_argument('--test_file', help='An optional input testing data file to test the perplexity on (a text file).',default='./data/test.json')
    parser.add_argument('--context_file', help='An optional input context data file to evaluate the perplexity on (a text file).',default='./data/context.json')
    

#     args = parser.parse_args(args=['--context_file', './data/context.json',\
#                                    '--test_file', './data/test.json'
#                                   ])    
    
    
    
    
    args = parser.parse_args()
    
    return args


args = parse_args()    
print(args.context_file)
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


# In[8]:


test=pd.read_json(args.test_file)
test


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


# In[12]:


test_swag_format=data2swagformat(test)
test_swag_format = pd.DataFrame(test_swag_format)
print(len(test_swag_format))
test_swag_format.to_csv('./preprocess/mc/test_swag_format.csv',index=False)


# # output test dataset paragraphs column

# In[13]:


with open('./preprocess/mc/test_paragraphs.pickle', 'wb') as f:
    pickle.dump(test['paragraphs'], f)

