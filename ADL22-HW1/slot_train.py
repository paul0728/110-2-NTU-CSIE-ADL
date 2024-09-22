#!/usr/bin/env python
# coding: utf-8

# In[1]:


from argparse import ArgumentParser, Namespace
import numpy as np
import random as python_random
import os



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--preprocess-dir', help='Directory to the preprocessed files.', default='./prep/slot/')
    parser.add_argument('--ckpt-save-path', help='Path to the save checkpoint file.')
    parser.add_argument('--max-words', help='Vocabulary size of train/validation/test dataset.', type=int, default=1002)
    parser.add_argument('--max-len', help='Maximum length of a sequence.', type=int, default=35)
    parser.add_argument('--model', help='Network model to train/validation', choices=['RNN', 'GRU', 'LSTM', 'BILSTM', 'CNN+BILSTM','2BILSTM','CNN+2BILSTM'], default='2BILSTM')
    parser.add_argument('--lr', help='Learning rate.', type=float, default=1e-3)
    parser.add_argument('--batch-size', help='Batch size.', type=int, default=64)
    parser.add_argument('--epoch', help='Maximum training epoch.', type=int, default=100)
    '''
    args = parser.parse_args(args=['--preprocess-dir','./prep/slot/',\
                                   '--ckpt-save-path',None,\
                                   '--max-words','1002',\
                                   '--max-len','35',\
                                   '--model','2BILSTM',\
                                   '--lr','1e-3',\
                                   '--batch-size','64',\
                                   '--epoch','100'
                                  ])
    '''
    
    args = parser.parse_args()
    return args

args = parse_args()

filename=f"{args.model}_mw_{args.max_words}_ml_{args.max_len}"
print(filename)

if args.ckpt_save_path==None:
    args.ckpt_save_path=f"./ckpt/slot/{filename}" 
    
#print(args.preprocess_dir)


# # Reproducibility

# In[2]:


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


# # GPU RAM 分配

# In[3]:


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try: 
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# # Load preprocessed data

# In[4]:


train_sequences = np.load(f'{args.preprocess_dir}train_sequences.npy', allow_pickle='TRUE')
eval_sequences = np.load(f'{args.preprocess_dir}eval_sequences.npy', allow_pickle='TRUE')
train_tags_encoded = np.load(f'{args.preprocess_dir}train_tags_encoded.npy', allow_pickle='TRUE')
eval_tags_encoded = np.load(f'{args.preprocess_dir}eval_tags_encoded.npy', allow_pickle='TRUE')
embedding_matrix = np.load(f'{args.preprocess_dir}embedding_matrix.npy', allow_pickle='TRUE')


embedding_matrix


# In[5]:


print(train_sequences.shape)
print(eval_sequences.shape)
print(train_tags_encoded.shape)
print(eval_tags_encoded.shape)


# # Build model

# In[6]:


#embedding 層mask_zero=True時的計算方法(此法將label 中padding的值看成'O',並不準確)
# def joint_acc_mask(y_true, y_pred):
# #     print('y_true=',y_true.shape)
# #     print('y_pred=',y_pred.shape)
# #     tf.print(tf.one_hot(tf.math.argmax(y_pred,axis=2),tag_num))
#     y_true=y_true.numpy()
#     y_true=np.where(y_true==[1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0],y_true)
# #     print(y_true)
#     comparison=tf.one_hot(tf.math.argmax(y_pred,axis=2),tag_num)==y_true
#     comparison=tf.reduce_all(comparison,axis=-1)
#     acc_tensor=tf.reduce_all(comparison,axis=-1)
# #     print("acc_tensor=",tf.print(acc_tensor))
    
#     acc_count=0
#     for i in acc_tensor:
#         if i==True:
#             acc_count+=1
# #     print("len(acc_tensor):",len(acc_tensor))
#     return acc_count/len(acc_tensor)

#無embedding時的計算方法

# @tf.function
# def joint_acc(y_true, y_pred):
# #     print('y_true=',y_true.shape)
# #     print('y_pred=',y_pred.shape)
# #     tf.print(tf.one_hot(tf.math.argmax(y_pred,axis=2),tag_num))

#     comparison=tf.one_hot(tf.math.argmax(y_pred,axis=2),tag_num)==y_true
#     comparison=tf.reduce_all(comparison,axis=-1)
#     acc_tensor=tf.reduce_all(comparison,axis=-1)
# #     print("acc_tensor=",tf.print(acc_tensor))
    
#     acc_count=0
#     for i in acc_tensor:
#         if i==True:
#             acc_count+=1
# #     print("len(acc_tensor):",len(acc_tensor))
#     return acc_count/len(acc_tensor)


# In[7]:


from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras import regularizers 
from keras.layers import Dense, Embedding, SimpleRNN, GRU, LSTM, Bidirectional, TimeDistributed,LeakyReLU, Conv1D,BatchNormalization
from livelossplot import PlotLossesKerasTF
from livelossplot.outputs import MatplotlibPlot
# from keras.metrics import Precision, Recall
# from keras_contrib.layers.crf import CRF
# from keras_contrib.losses.crf_losses import crf_loss
# from keras_contrib.metrics.crf_accuracies import crf_accuracy


ckpt = ModelCheckpoint(f"{args.ckpt_save_path}", monitor="val_categorical_accuracy",verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='min', baseline=None, restore_best_weights=False)
callbacks = [PlotLossesKerasTF(outputs=[MatplotlibPlot(figpath =f"./ckpt/slot/picture/{filename}.png")]), ckpt, early_stopping]
# callbacks = [ckpt, early_stopping]

#Nadam = tf.optimizers.Nadam(clipvalue=0.1,learning_rate=args.lr)
adam = tf.keras.optimizers.Adam(clipvalue=0.1,learning_rate=args.lr)
rmsprop=tf.keras.optimizers.RMSprop(clipvalue=0.1,learning_rate=args.lr)
EMBEDDING_DIM = 300
NUM_UNITS = 256
NUM_LABELS = 150
tag_num = 10

#加dropout都會變差
#用swish 比relu爛
#用rmsprop 會train 不起來
# crf_layer = CRF(tag_num)
# model.add(crf_layer)

# model select
if args.model == 'RNN':
    model = Sequential()
    model.add(Embedding(input_dim = args.max_words, output_dim = EMBEDDING_DIM,mask_zero=True, input_length=args.max_len))
    model.add(BatchNormalization())
    model.add(SimpleRNN(NUM_UNITS,activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(tag_num, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))))
elif args.model == 'GRU':
    model = Sequential()
    model.add(Embedding(input_dim = args.max_words, output_dim = EMBEDDING_DIM,mask_zero=True, input_length=args.max_len))
    model.add(BatchNormalization())
    model.add(GRU(NUM_UNITS,activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(tag_num, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))))
elif args.model == 'LSTM':
    model = Sequential()
    model.add(Embedding(input_dim = args.max_words, output_dim = EMBEDDING_DIM,mask_zero=True, input_length=args.max_len))
    model.add(BatchNormalization())
    model.add(LSTM(NUM_UNITS,activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(tag_num, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))))    
elif args.model == 'BILSTM':
    model = Sequential()
    model.add(Embedding(input_dim = args.max_words, output_dim = EMBEDDING_DIM,mask_zero=True, input_length=args.max_len))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(NUM_UNITS, activation='relu', return_sequences=True)))
    model.add(TimeDistributed(Dense(tag_num, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))) 
elif args.model == 'CNN+BILSTM':
    model = Sequential()
    model.add(Embedding(input_dim = args.max_words, output_dim = EMBEDDING_DIM,mask_zero=True, input_length=args.max_len))
    model.add(BatchNormalization())
    model.add(Conv1D(NUM_UNITS,3,activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(NUM_UNITS, activation='relu', return_sequences=True)))
    model.add(TimeDistributed(Dense(tag_num, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))) 
elif args.model == '2BILSTM':
    model = Sequential()
    model.add(Embedding(input_dim = args.max_words, output_dim = EMBEDDING_DIM,mask_zero=True, input_length=args.max_len))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(NUM_UNITS, activation='relu', return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(NUM_UNITS, activation='relu', return_sequences=True)))
    model.add(TimeDistributed(Dense(tag_num, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))) 
elif args.model == 'CNN+2BILSTM':
    model = Sequential()
    model.add(Embedding(input_dim = args.max_words, output_dim = EMBEDDING_DIM,mask_zero=True, input_length=args.max_len))
    model.add(BatchNormalization())
    model.add(Conv1D(NUM_UNITS,3,activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(NUM_UNITS, activation='relu', return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(NUM_UNITS, activation='relu', return_sequences=True)))
    model.add(TimeDistributed(Dense(tag_num, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))))     


# 建立存放model以及picture 的資料夾
os.makedirs('./ckpt',exist_ok = True)
os.makedirs('./ckpt/slot',exist_ok = True)
os.makedirs('./ckpt/slot/picture',exist_ok = True)


# In[8]:


'''
unmasked_embedding = tf.cast(
    tf.tile(tf.expand_dims(train_sequences, axis=-1), [1, 1, 10]), tf.float32
)
print(unmasked_embedding.shape)
masking_layer = Masking()


masked_embedding = masking_layer(unmasked_embedding)

model.add(Masking(mask_value=0.,
          input_shape=(35, 10)))
'''


# In[9]:


# print(model.optimizer)
# print(model.loss)
# print(model.metrics)
# print(model.get_layer(index=0))
# print(model.input)
# print(model.output)
# print(model.name)
# print(model.layers) 
# print(model.get_config())
# for l in model.get_config()["layers"]:
#     print(l)

# with open("model.json", "w") as file:
#     json.dump(model.to_json(), file)
# df = pd.read_json(model.to_json())
# df["config"][0]


# #  將預訓練的文字嵌入向量載入到嵌入向量層中

# In[10]:


model.layers[0].set_weights([embedding_matrix])
# model.layers[0].trainable = False
model.layers[0].trainable = True


# # model 架構

# In[11]:


model.summary()


# # Train

# In[12]:


# model complie
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=["categorical_accuracy"])
#Fit the model on training data
history = model.fit(train_sequences, 
                    train_tags_encoded, 
                    batch_size = args.batch_size, 
                    epochs = args.epoch, 
                    validation_data=(eval_sequences, eval_tags_encoded),
                    callbacks=callbacks)


# In[13]:


# #Fit the model on training(use validation data as training data) data
# train_val_sequences=np.concatenate((train_sequences, eval_sequences), axis=0)
# train_val_tags_encoded=np.concatenate((train_tags_encoded, eval_tags_encoded), axis=0)
# history = model.fit(train_sequences, 
#                     train_tags_encoded, 
#                     batch_size = BATCH_SIZE, 
#                     epochs = EPOCHS, 
#                     callbacks=callbacks)

