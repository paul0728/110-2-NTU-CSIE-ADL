#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
import os.path

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--trainer_state_path', help='Path to the trainer-state.')
    
#     args = parser.parse_args(args=['--trainer_state_path', './BeamSearch_numbeams5/best_ckpt/trainer_state.json'])    
    args = parser.parse_args()
    
    return args
args = parse_args()    
print(args.trainer_state_path)


# In[5]:


trainer_state=pd.read_json(args.trainer_state_path)['log_history']
trainer_state=pd.read_json(trainer_state.to_json(orient='records'))
#只取eval_loss,eval_exact_match
trainer_state=trainer_state[['epoch','rouge-1_f','rouge-1_p','rouge-1_r','rouge-2_f','rouge-2_p','rouge-2_r','rouge-l_f','rouge-l_p','rouge-l_r']]

#取出要的項目
drop_index=[i for i in range(len(trainer_state)) if i%2==0]
trainer_state=trainer_state.drop(drop_index)
trainer_state



trainer_state.set_index('epoch', inplace=True) 

chart_f1 = trainer_state[['rouge-1_f','rouge-2_f','rouge-l_f']].plot(
                    xlabel='epoch',  #x軸說明文字
                    ylabel='f1_score',  #y軸說明文字
                    figsize=(10, 5))  # 圖表大小
plt.savefig(os.path.dirname(args.trainer_state_path)+"/chart_f1.png")
# plt.show()



chart_precision = trainer_state[['rouge-1_p','rouge-2_p','rouge-l_p']].plot(
                    xlabel='epoch',  #x軸說明文字
                    ylabel='precision',  #y軸說明文字
                    figsize=(10, 5))  # 圖表大小
plt.savefig(os.path.dirname(args.trainer_state_path)+"/chart_precision.png")
# plt.show()


chart_recall = trainer_state[['rouge-1_r','rouge-2_r','rouge-l_r']].plot(
                    xlabel='epoch',  #x軸說明文字
                    ylabel='recall',  #y軸說明文字
                    figsize=(10, 5))  # 圖表大小
plt.savefig(os.path.dirname(args.trainer_state_path)+"/chart_recall.png")
# plt.show()

