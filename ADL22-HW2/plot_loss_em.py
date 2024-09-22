#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
import os.path

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--trainer_state_path', help='Path to the trainer-state.',default='./bert_base_chinese/qa/trainer_state.json')
    
#     args = parser.parse_args(args=['--trainer_state_path', './bert_base_chinese/qa/trainer_state.json'])
    
    args = parser.parse_args()
    
    return args
args = parse_args()    
print(args.trainer_state_path)


# In[16]:


trainer_state=pd.read_json(args.trainer_state_path)['log_history']
trainer_state=pd.read_json(trainer_state.to_json(orient='records'))
#只取eval_loss,eval_exact_match
trainer_state=trainer_state[['step','eval_loss','eval_exact_match']]

#取出要的項目
drop_index=[i for i in range(len(trainer_state)) if i%2==0]
trainer_state=trainer_state.drop(drop_index)
trainer_state

trainer_state.set_index('step', inplace=True) 

chart_loss = trainer_state[['eval_loss']].plot(
                    xlabel='Step',  #x軸說明文字
                    ylabel='eval_loss',  #y軸說明文字
                    figsize=(10, 5))  # 圖表大小
plt.savefig(os.path.dirname(args.trainer_state_path)+"/chart_loss.png")
# plt.show()



chart_em = trainer_state[['eval_exact_match']].plot(
                    xlabel='Step',  #x軸說明文字
                    ylabel='EM',  #y軸說明文字
                    figsize=(10, 5))  # 圖表大小
plt.savefig(os.path.dirname(args.trainer_state_path)+"/chart_em.png")
# plt.show()

