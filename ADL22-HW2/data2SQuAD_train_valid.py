#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import json
from argparse import ArgumentParser, Namespace
import os.path



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--train_file', help='The input training data file (a text file).',default='./data/train.json')
    parser.add_argument('--validation_file', help='An optional input evaluation data file to evaluate the perplexity on (a text file).',default='./data/valid.json')
    parser.add_argument('--context_file', help='An optional input context data file to evaluate the perplexity on (a text file).',default='./data/context.json')

#     args = parser.parse_args(args=['--trainer-state', './bert_base_chinese/qa/trainer_state.json'])
    
    args = parser.parse_args()
    
    return args


args = parse_args()    


# In[1]:


context=pd.read_json(args.context_file)
print(len(context[0]))
context[0][836][108]


# In[2]:


train=pd.read_json(args.train_file)
train


# In[3]:


valid=pd.read_json(args.validation_file)
valid


# # Data Fields(swag)
# 
# 

#  SQuAD dataset
#  
#  {
#  
#      "id": "1",
#      
#      "question": "Is this a test?",
#      
#      "context": "This is a test context.",
#      
#      "answers": 
#      {
#          "answer_start": [1],
#          
#          "text": ["This is a test text"]
#          
#      },
#      
#  }
#  
#  

# In[5]:


def qa_encoder(examples):
    num_samples = len(examples['id'])
    if 'answer' in examples:
        return{ 
            "id": examples['id'],
            "question": examples["question"],
            "context": list(map(lambda x:context[0][x], examples['relevant'])),
            "answers": list(map(lambda x:{"answer_start": [x["start"],], "text":[x["text"],]}, examples['answer'])),
     
        }
    else:
        return{ 
            "id": examples['id'],
            "question": examples["question"],
            "context": list(map(lambda x:context[0][x], examples['relevant'])),
        }        


# In[ ]:


# 建立資料夾
os.makedirs('./preprocess/qa/', exist_ok=True)  


# In[6]:


train_SQuAD_format=qa_encoder(train)
train_SQuAD_format=pd.DataFrame(train_SQuAD_format)
result = train_SQuAD_format.to_json(orient='records')
parsed = json.loads(result)
train_SQuAD_format_final={'data':parsed}
with open('./preprocess/qa/train_SQuAD_format.json', 'w') as f:
    json.dump(train_SQuAD_format_final, f)

train_SQuAD_format


# In[7]:


valid_SQuAD_format=qa_encoder(valid)
valid_SQuAD_format=pd.DataFrame(valid_SQuAD_format)
result = valid_SQuAD_format.to_json(orient='records')
parsed = json.loads(result)
valid_SQuAD_format_final={'data':parsed}
with open('./preprocess/qa/valid_SQuAD_format.json', 'w') as f:
    json.dump(valid_SQuAD_format_final, f)

valid_SQuAD_format

