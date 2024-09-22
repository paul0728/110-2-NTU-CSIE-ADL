#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from argparse import ArgumentParser, Namespace
import numpy as np
import pandas as pd
import os

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--train-dataset-path', help='Path to the train dataset.', default='./data/intent/train.json')
    parser.add_argument('--valid-dataset-path', help='Path to the validation dataset.', default='./data/intent/eval.json')
    parser.add_argument('--test-dataset-path', help='Path to the test dataset.', default='./data/intent/test.json')
    parser.add_argument('--preprocess-dir', help='Directory to the preprocessed files.', default='./prep/intent/')
    parser.add_argument('--max-words', help='Vocabulary size of train/validation/test dataset.', type=int, default=10000)
    parser.add_argument('--max-len', help='Maximum length of a sequence.', type=int, default=28)
    '''
    args = parser.parse_args(args=['--train-dataset-path','./data/intent/train.json',\
                                   '--valid-dataset-path','./data/intent/eval.json',\
                                   '--test-dataset-path','./data/intent/test.json',\
                                   '--preprocess-dir','./prep/intent/',\
                                   '--max-words','10000',\
                                   '--max-len','28'
                                  ])
    
    '''
    args = parser.parse_args()
    return args
args=parse_args()
print(args.train_dataset_path)
print(args.max_words)


# # Load data

# In[2]:


train=pd.read_json (args.train_dataset_path)
eval_=pd.read_json (args.valid_dataset_path)
test=pd.read_json (args.test_dataset_path)
print(len(train))
print(len(eval_))
print(len(test))
train


# # Data preprocessing 

# In[3]:


all_data=pd.concat([pd.concat([train, eval_], ignore_index=True), test], ignore_index=True)
# all_texts=pd.concat([all_texts, test["text"]], ignore_index=True)

#最多args.max_words-1個字(含oov)
# tokenizer = Tokenizer(num_words=max_words,oov_token='<OOV>')
tokenizer = Tokenizer(num_words=args.max_words)
#all_data["text"]
print(all_data["intent"])



#all_texts=pd.concat([all_texts, test["text"]], ignore_index=True)
tokenizer.fit_on_texts(all_data["text"])
sequences = tokenizer.texts_to_sequences(all_data["text"])# 將文字轉成整數 list 的序列資料
sequences


# In[4]:


#建立word to index 和index to word 的dic
#word2index
word2index=tokenizer.word_index
num_of_words=len(word2index)
#6438
#index2word
index2word={v:k for (k,v) in word2index.items()}

#data preprocessing
train_sequences=sequences[:15000]
eval_sequences=sequences[15000:18000]
test_sequences=sequences[18000:]


# def cal_len(x): 
#     return len(x)
# train_sequences_len = max(map(cal_len,train_sequences))
# eval_sequences_len = max(map(cal_len,eval_sequences))
# test_sequences_len = max(map(cal_len,test_sequences))
# maxlen=max(train_sequences_len, eval_sequences_len, test_sequences_len)
# print(maxlen)


#data padding
train_sequences = pad_sequences(train_sequences, maxlen=args.max_len,padding='post', truncating='post')
eval_sequences = pad_sequences(eval_sequences, maxlen=args.max_len,padding='post', truncating='post')
test_sequences = pad_sequences(test_sequences, maxlen=args.max_len,padding='post', truncating='post')



#label preprocessing
max_labels=len(all_data["intent"][:18000].unique())
print(max_labels)
label2index={}
index2label={}
c=0
for i,label in enumerate(all_data["intent"][:18000]):
    if label2index.get(label) is None:
        label2index[label]=c
        c+=1
        
index2label={v:k for (k,v) in label2index.items()}
train_labels=np.asarray(train["intent"].map(label2index)).astype('float32')
eval_labels=np.asarray(eval_["intent"].map(label2index)).astype('float32')




# 產生label2index, index2tlabel, test_sequences file
os.makedirs('./prep',exist_ok = True)
os.makedirs('./prep/intent',exist_ok = True)
np.save("./prep/intent/label2index",label2index)
np.save("./prep/intent/index2label",index2label)

np.save('./prep/intent/train_sequences', train_sequences)
np.save('./prep/intent/eval_sequences', eval_sequences)
np.save('./prep/intent/test_sequences', test_sequences)


np.save('./prep/intent/train_labels', train_labels)
np.save('./prep/intent/eval_labels', eval_labels)
# print(index2label)


# # 解析 GloVe 文字嵌入向量檔案

# In[5]:


# glove_dir = r'/mnt/G/ADLJIZZ/hw1/40675020h/intent_classification'
embeddings_index = {}
f = open('glove.840B.300d.txt', encoding='UTF-8')
for line in f:
    values = line.split(" ")
#     print(values)
    #values = line.split()
    #print(values[1:])
    word = values[0]
    #print("word",word)
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
    
    
f.close()

print('共有 %s 個文字嵌入向量' % len(embeddings_index))


# # 準備 GloVe 文字嵌入向量矩陣

# In[6]:


embedding_dim = 300

embedding_matrix = np.zeros((args.max_words, embedding_dim))
for word, i in word2index.items():
    #print(word)
    if i < args.max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector  # ←嵌入向量索引中找不到的文字將為 0          
np.save('./prep/intent/embedding_matrix',embedding_matrix)

