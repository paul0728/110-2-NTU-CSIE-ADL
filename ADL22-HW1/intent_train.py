#!/usr/bin/env python
# coding: utf-8

# In[27]:


from argparse import ArgumentParser, Namespace
import numpy as np
import random as python_random
import os


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--preprocess-dir', help='Directory to the preprocessed files.', default='./prep/intent/')
    parser.add_argument('--ckpt-save-path', help='Path to the save checkpoint file.')
    parser.add_argument('--max-words', help='Vocabulary size of train/validation/test dataset.', type=int, default=10000)
    parser.add_argument('--max-len', help='Maximum length of a sequence.', type=int, default=28)
    parser.add_argument('--model', help='Network model to train/validation', choices=['RNN', 'GRU', 'LSTM', 'BILSTM', 'CNN+BILSTM','2BILSTM','CNN+2BILSTM'], default='2BILSTM')
    parser.add_argument('--lr', help='Learning rate.', type=float, default=1e-3)
    parser.add_argument('--batch-size', help='Batch size.', type=int, default=64)
    parser.add_argument('--epoch', help='Maximum training epoch.', type=int, default=100)
    '''
    args = parser.parse_args(args=['--preprocess-dir','./prep/intent/',\
                                   '--ckpt-save-path',None,\
                                   '--max-words','10000',\
                                   '--max-len','28',\
                                   '--model','2BILSTM',\
                                   '--lr','1e-3',\
                                   '--batch-size','64',\
                                   '--epoch','100'
                                  ])
    '''
    args = parser.parse_args()
    return args

args = parse_args()

filename=f"{args.model}_nooov_mw_{args.max_words}_ml_{args.max_len}"
print(filename)

if args.ckpt_save_path==None:
    args.ckpt_save_path=f"./ckpt/intent/{filename}" 
    
#print(args.preprocess_dir)


# # Reproducibility

# In[28]:


#CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python your_program.py
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.python.client import device_lib
'''
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

gpus = get_available_gpus()
print(gpus)
'''




# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(123)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(123)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(1234)

# Rest of code follows ...


# # Load preprocessed data

# In[29]:


train_sequences = np.load(f'{args.preprocess_dir}train_sequences.npy', allow_pickle='TRUE')
eval_sequences = np.load(f'{args.preprocess_dir}eval_sequences.npy', allow_pickle='TRUE')
train_labels = np.load(f'{args.preprocess_dir}train_labels.npy', allow_pickle='TRUE')
eval_labels = np.load(f'{args.preprocess_dir}eval_labels.npy', allow_pickle='TRUE')
embedding_matrix = np.load(f'{args.preprocess_dir}embedding_matrix.npy', allow_pickle='TRUE')



# # Build model

# In[30]:


from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras import  regularizers 
from keras.layers import Dense, Embedding, SimpleRNN, GRU, LSTM, Bidirectional, TimeDistributed, Conv1D,BatchNormalization
from livelossplot import PlotLossesKerasTF
from livelossplot.outputs import MatplotlibPlot

ckpt = ModelCheckpoint(f"{args.ckpt_save_path}", monitor="val_sparse_categorical_accuracy",verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='min', baseline=None, restore_best_weights=False)
callbacks = [PlotLossesKerasTF(outputs=[MatplotlibPlot(figpath =f"./ckpt/intent/picture/{filename}.png")]), ckpt, early_stopping]
# callbacks = [ckpt, early_stopping]

#Nadam = tf.optimizers.Nadam(clipvalue=0.1,learning_rate=args.lr)
adam = tf.keras.optimizers.Adam(clipvalue=0.1,learning_rate=args.lr)
rmsprop=tf.keras.optimizers.RMSprop(clipvalue=0.1,learning_rate=args.lr)
EMBEDDING_DIM = 300
NUM_UNITS = 256
NUM_LABELS = 150


# model select
if args.model == 'RNN':
    model = Sequential()
    model.add(Embedding(input_dim = args.max_words, output_dim = EMBEDDING_DIM,mask_zero=True, input_length=args.max_len))
    model.add(BatchNormalization())
    model.add(SimpleRNN(NUM_UNITS,activation='relu'))
    model.add(Dense(NUM_LABELS, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
elif args.model == 'GRU':
    model = Sequential()
    model.add(Embedding(input_dim = args.max_words, output_dim = EMBEDDING_DIM,mask_zero=True, input_length=args.max_len))
    model.add(BatchNormalization())
    model.add(GRU(NUM_UNITS,activation='relu'))
    model.add(Dense(NUM_LABELS, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
elif args.model == 'LSTM':
    model = Sequential()
    model.add(Embedding(input_dim = args.max_words, output_dim = EMBEDDING_DIM,mask_zero=True, input_length=args.max_len))
    model.add(BatchNormalization())
    model.add(LSTM(NUM_UNITS,activation='relu'))
    model.add(Dense(NUM_LABELS, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))    
elif args.model == 'BILSTM':
    model = Sequential()
    model.add(Embedding(input_dim = args.max_words, output_dim = EMBEDDING_DIM,mask_zero=True, input_length=args.max_len))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(NUM_UNITS, activation='relu')))
    model.add(Dense(NUM_LABELS, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))) 
elif args.model == 'CNN+BILSTM':
    model = Sequential()
    model.add(Embedding(input_dim = args.max_words, output_dim = EMBEDDING_DIM,mask_zero=True, input_length=args.max_len))
    model.add(BatchNormalization())
    model.add(Conv1D(NUM_UNITS,3,activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(NUM_UNITS, activation='relu')))
    model.add(Dense(NUM_LABELS, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))) 
elif args.model == '2BILSTM':
    model = Sequential()
    model.add(Embedding(input_dim = args.max_words, output_dim = EMBEDDING_DIM,mask_zero=True, input_length=args.max_len))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(NUM_UNITS, activation='relu', return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(NUM_UNITS, activation='relu')))
    model.add(Dense(NUM_LABELS, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))) 
elif args.model == 'CNN+2BILSTM':
    model = Sequential()
    model.add(Embedding(input_dim = args.max_words, output_dim = EMBEDDING_DIM,mask_zero=True, input_length=args.max_len))
    model.add(BatchNormalization())
    model.add(Conv1D(NUM_UNITS,3,activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(NUM_UNITS, activation='relu', return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(NUM_UNITS, activation='relu')))
    model.add(Dense(NUM_LABELS, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))     


# 建立存放model以及picture 的資料夾
os.makedirs('./ckpt',exist_ok = True)
os.makedirs('./ckpt/intent',exist_ok = True)
os.makedirs('./ckpt/intent/picture',exist_ok = True)


# #  將預訓練的文字嵌入向量載入到嵌入向量層中

# In[31]:


model.layers[0].set_weights([embedding_matrix])
# model.layers[0].trainable = False
model.layers[0].trainable = True


# # model 架構

# In[32]:


model.summary()


# # Train

# In[7]:


# model complie
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
#Fit the model on training data
history = model.fit(train_sequences, 
                    train_labels, 
                    batch_size = args.batch_size, 
                    epochs = args.epoch, 
                    validation_data=(eval_sequences, eval_labels),
                    callbacks=callbacks)

