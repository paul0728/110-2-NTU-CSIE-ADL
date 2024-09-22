#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from argparse import ArgumentParser, Namespace
from keras.models import load_model
import os

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--test-dataset-path', help='Path to the test dataset.', default='./data/slot/test.json')
    parser.add_argument('--valid-dataset-path', help='Path to the validation dataset.', default='./data/slot/eval.json')
    parser.add_argument('--preprocess-dir', help='Directory to the preprocessed files.', default='./prep/slot/')
    parser.add_argument('--pred-file-path', help='Path to the create predict file.')
    parser.add_argument('--max-words', help='Vocabulary size of train/validation/test dataset.', type=int, default=1002)
    parser.add_argument('--max-len', help='Maximum length of a sequence.', type=int, default=35)
    parser.add_argument('--model', help='Network model to test.', choices=['RNN', 'GRU', 'LSTM', 'BILSTM', 'CNN+BILSTM','2BILSTM','CNN+2BILSTM'], default='2BILSTM')
    parser.add_argument('--ckpt-load-path', help='Path to the load checkpoint file.')
    '''
    args = parser.parse_args(args=['--test-dataset-path', './data/slot/test.json', \
                                   '--preprocess-dir', './prep/slot/', \
                                   '--pred-file-path', None,\
                                   '--max-words','1002',\
                                   '--max-len','35',\
                                   '--model', '2BILSTM',\
                                   '--ckpt-load-path','./ckpt/slot/cnn+bilstm'
                                  ])
    '''
    args = parser.parse_args()

    return args

args = parse_args()
# print(args.test_dataset_path)
# print(args.pred_file)

filename=f"{args.model}_mw_{args.max_words}_ml_{args.max_len}"
print(filename)


if args.pred_file_path==None:
    args.pred_file_path=f"./results/slot/{filename}"
    
if args.ckpt_load_path==None:
    args.ckpt_load_path=f"./ckpt/slot/{filename}"    
    
    
# print(args.pred_file_path)


# # Load data

# In[2]:


index2tag = np.load(f'{args.preprocess_dir}index2tag.npy', allow_pickle='TRUE').item()
tag2index = np.load(f'{args.preprocess_dir}tag2index.npy', allow_pickle='TRUE').item()
eval_sequences = np.load(f'{args.preprocess_dir}eval_sequences.npy', allow_pickle='TRUE')
test_sequences = np.load(f'{args.preprocess_dir}test_sequences.npy', allow_pickle='TRUE')
test=pd.read_json (args.test_dataset_path)
eval_=pd.read_json (args.valid_dataset_path)
index2tag


# In[3]:


tag2index


# In[4]:


test_sequences


# In[5]:


eval_


# # Load model

# In[6]:


# 載入模型
model = load_model(args.ckpt_load_path)


# In[7]:


model.summary()


# # Inference and create csv file

# In[8]:


prediction = model.predict(test_sequences)
tag_list = list(tag2index.keys())
index_list = list(tag2index.values())
test_result=[]
test_num=len(test)
for i,tokens in enumerate(test["tokens"]):
    slots = [tag_list[index_list.index(j)] for j in [np.argmax(x) for x in prediction[i][:]] if j in index_list]
#     print(slots)
#     slots[:] = (value for value in slots if value != "padding")
#     slots=slots[:len(tokens)]
    temp=[]
    for value in slots:

        if value == "padding":
            temp.append("O")
        else:
            temp.append(value)
            
    slots=temp[:len(tokens)]
    test_result.append(' '.join(slots))
# print(result)
# print(np.argmax(prediction))
# print(sentence)
# print(slots)

result_dic={
    "id":test["id"],
    "tags":test_result,
    
}


os.makedirs('./results',exist_ok = True)
os.makedirs('./results/slot',exist_ok = True)

result = pd.DataFrame(result_dic)
result.to_csv(args.pred_file_path,index=False)
result

print(tag2index)


# # seqeval

# In[16]:


from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2


# x_train = np.asarray(x_train).astype(np.float32)
# y_train = np.asarray(y_train).astype(np.float32)
eval_prediction = model.predict(eval_sequences)
eval_result=[]



    
for i,tags in enumerate(eval_["tags"]):
    
    slots = [tag_list[index_list.index(j)] for j in [np.argmax(x) for x in eval_prediction[i][:]] if j in index_list]
    # slots[:] = (value for value in slots if value != "padding")

    temp=[]
    for value in slots:

        if value == "padding":
            temp.append("O")
        else:
            temp.append(value)
            
    slots=temp[:len(tags)]
    
    eval_result.append(slots)

y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
# print(classification_report(eval_["tags"], eval_result, mode='strict', scheme=IOB2))
print(classification_report(y_true, y_pred, mode='strict', scheme=IOB2))



def cal_token_acc():
    eval_tags = sum(eval_["tags"],[])
    eval_results = sum(eval_result,[])
    token_num=len(eval_tags)
    correct_count=0
    for t,r in zip(eval_tags,eval_results):
        if t==r:
            correct_count+=1
    return correct_count/token_num



def cal_joint_acc():
    
    correct_count=0
    eval_num=len(eval_["tags"])
    for t,r in zip(eval_["tags"],eval_result):
        if t==r:
            correct_count+=1
    return correct_count/eval_num



token_acc=cal_token_acc()
joint_acc = cal_joint_acc()


print("token_acc:",token_acc)
print("joint_acc:",joint_acc)

