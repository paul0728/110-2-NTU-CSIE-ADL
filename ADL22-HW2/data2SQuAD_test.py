#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import json
from argparse import ArgumentParser, Namespace
import os.path


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--context_file', help='An optional input context data file to evaluate the perplexity on (a text file).',default='./data/context.json')
    parser.add_argument('--MC_output_path',help="Path to MC output file")
    parser.add_argument('--model_name',help=" Hugging face model name ,remove '/',and replace '-' with '_' ")
#     args = parser.parse_args(args=['--trainer-state', './bert_base_chinese/qa/trainer_state.json'])
    
    args = parser.parse_args()
    
    return args


args = parse_args()    


print(args.MC_output_path)
print(args.model_name)
print(args.context_file)


# In[1]:


context=pd.read_json(args.context_file)
print(len(context[0]))
context[0][836][108]


# In[4]:


test=pd.read_json(args.MC_output_path)
test


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
os.makedirs('./preprocess/qa/test', exist_ok=True)  


# In[8]:


test_SQuAD_format=qa_encoder(test)
test_SQuAD_format=pd.DataFrame(test_SQuAD_format)
result = test_SQuAD_format.to_json(orient='records')
parsed = json.loads(result)
test_SQuAD_format_final={'data':parsed}
with open(f'./preprocess/qa/test/{args.model_name}_test_SQuAD_format.json', 'w') as f:
    json.dump(test_SQuAD_format_final, f)

test_SQuAD_format

