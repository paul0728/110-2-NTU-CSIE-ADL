#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from argparse import ArgumentParser, Namespace
import numpy as np
import pandas as pd
import os
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--train-dataset-path', help='Path to the train dataset.', default='./data/slot/train.json')
    parser.add_argument('--valid-dataset-path', help='Path to the validation dataset.', default='./data/slot/eval.json')
    parser.add_argument('--test-dataset-path', help='Path to the test dataset.', default='./data/slot/test.json')
    parser.add_argument('--preprocess-dir', help='Directory to the preprocessed files.', default='./prep/slot/')
    parser.add_argument('--max-words', help='Vocabulary size of train/validation/test dataset.', type=int, default=1002)
    parser.add_argument('--max-len', help='Maximum length of a sequence.', type=int, default=35)
    '''
    args = parser.parse_args(args=['--train-dataset-path','./data/slot/train.json',\
                                   '--valid-dataset-path','./data/slot/eval.json',\
                                   '--test-dataset-path','./data/slot/test.json',\
                                   '--preprocess-dir','./prep/slot/',\
                                   '--max-words','1002',\
                                   '--max-len','35'
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
tokenizer = Tokenizer(num_words=args.max_words,oov_token='<OOV>')
# tokenizer = Tokenizer(num_words=args.max_words)

#all_texts=pd.concat([all_texts, test["text"]], ignore_index=True)
tokenizer.fit_on_texts(all_data["tokens"])
sequences = tokenizer.texts_to_sequences(all_data["tokens"])# 將文字轉成整數 list 的序列資料
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
train_sequences=sequences[:7244]
eval_sequences=sequences[7244:8244]
test_sequences=sequences[8244:]
'''
def cal_len(x): 
    return len(x)
train_sequences_len = max(map(cal_len,train_sequences))
eval_sequences_len = max(map(cal_len,eval_sequences))
test_sequences_len = max(map(cal_len,test_sequences))
maxlen=max(train_sequences_len, eval_sequences_len, test_sequences_len)
print(maxlen)
'''

#data padding
train_sequences = pad_sequences(train_sequences, maxlen=args.max_len,padding='post', truncating='post')
eval_sequences = pad_sequences(eval_sequences, maxlen=args.max_len,padding='post', truncating='post')
test_sequences = pad_sequences(test_sequences, maxlen=args.max_len,padding='post', truncating='post')


# #Reshape the input for Bi-LSTM
train_sequences = np.reshape(train_sequences, (train_sequences.shape[0], train_sequences.shape[1], 1))
eval_sequences = np.reshape(eval_sequences, (eval_sequences.shape[0], eval_sequences.shape[1], 1))
test_sequences = np.reshape(test_sequences, (test_sequences.shape[0], test_sequences.shape[1], 1))
print(train_sequences.shape, eval_sequences.shape, test_sequences.shape)

#tag(label) preprocessing
tag=[]
tag2index={}

for iob in all_data["tags"][:8244]:
    tag.extend(iob)
    
tag=set(tag)
# print(tag)

tag2index={'padding': 0, 'B-date': 1, 'B-time': 2, 'B-people': 3, 'B-first_name': 4, 'B-last_name': 5, 'I-people': 6, 'O': 7, 'I-date': 8, 'I-time': 9}
#set 順序不同,會導致dic 不一樣
# tag2index={"padding":0, **{t:i for i,t in enumerate(tag,1)}}

tag_num = len(tag2index)
print("\ntag2index:",tag2index)
print("\ntag_num:",tag_num)
       
index2tag={v:k for (k,v) in tag2index.items()}
print("\nindex2tag:",index2tag)


def tag2index_(x):
    return tag2index[x]

train_tags=[]
for seq in train["tags"]:
    train_tags.append(list(map(tag2index_,seq)))
    
eval_tags=[]
for seq in eval_["tags"]:
    eval_tags.append(list(map(tag2index_,seq)))


#tag padding(0代表padding)   
train_tags = pad_sequences(train_tags, maxlen=args.max_len,padding='post', truncating='post')
eval_tags = pad_sequences(eval_tags, maxlen=args.max_len,padding='post', truncating='post')


#Convert tags to one-hot vectors
train_tags_encoded = np_utils.to_categorical(train_tags)
eval_tags_encoded = np_utils.to_categorical(eval_tags)
print(train_tags_encoded.shape, eval_tags_encoded.shape)


# 產生tag2index, index2tag, test_sequences, eval_sequences file
os.makedirs('./prep',exist_ok = True)
os.makedirs('./prep/slot',exist_ok = True)

np.save("./prep/slot/tag2index",tag2index)
np.save("./prep/slot/index2tag",index2tag)

np.save('./prep/slot/train_sequences', train_sequences)
np.save('./prep/slot/eval_sequences', eval_sequences)
np.save('./prep/slot/test_sequences', test_sequences)


np.save('./prep/slot/train_tags_encoded', train_tags_encoded)
np.save('./prep/slot/eval_tags_encoded', eval_tags_encoded)


# # 解析 GloVe 文字嵌入向量檔案

# In[5]:


# glove_dir = r'/mnt/G/ADLJIZZ/hw1/40675020h/slot_classification'
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
np.save('./prep/slot/embedding_matrix',embedding_matrix)

