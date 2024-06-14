#!/usr/bin/env python
# coding: utf-8

# # 1 Preprocessing of Text

# To encode each character into a one-hot vector as input of RNN

# In[1]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import warnings
warnings.filterwarnings("ignore")

#!pip install matplotlib
#!pip install scikit-learn

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import unicodedata
import re
import numpy as np
import os
import time
from pylab import *
from matplotlib.font_manager import FontProperties
import pandas as pd


# In[9]:


tf.__version__


# In[2]:


import io
#"C:/Users/cluster/Desktop/lee/DL_HW3/shakespeare_train.txt"

data_URL = "C:/Users/stat_835/Desktop/DL/DL_HW3/shakespeare_train.txt"
with io.open( data_URL , 'r' , encoding="utf8" ) as f :
    text=f.read()
print ('Length of text: {} characters'.format(len(text)))

vocab = sorted(set(text)) #set=unique
print ('{} unique characters'.format(len(vocab)))

vocab_to_int={c : i for i , c in enumerate( vocab )}
int_to_vocab = dict(enumerate( vocab ) )
train_data=np.array([vocab_to_int[c] for c in text],dtype=np.int32)


# In[3]:


int_to_vocab


# We find that the dataset has 67 unique characters.

# In[4]:


print ('{} characters mapped to int {}'.format(repr(text[:13]), [vocab_to_int[c] for c in text[:13]]))


# In[5]:


train_data


# In[6]:


data_URL = "C:/Users/stat_835/Desktop/DL/DL_HW3/shakespeare_valid.txt"
with io.open(data_URL, 'r', encoding ='utf8') as f:
    text2 = f.read()
    
valid_data = np.array([vocab_to_int[c] for c in text2], dtype = np.int32)


# In[7]:


print ('{} characters mapped to int {}'.format(repr(text2[:5]), [vocab_to_int[c] for c in text2[:5]]))


# In[8]:


valid_data


# One hot encoding

# In[10]:


train_one_hot = tf.one_hot(train_data, len(vocab))
valid_one_hot = tf.one_hot(valid_data, len(vocab))


# In[11]:


train_one_hot.shape,valid_one_hot.shape


# In[12]:


train_one_hot[0].numpy()


# # 2 Recurrent Neural Network

# In[13]:


idx2char = np.array(vocab)
idx2char


# ### Create training examples and targets

# In[14]:


# The maximum length sentence we want for a single input in characters
seq_length = 100

char_dataset = tf.data.Dataset.from_tensor_slices(train_one_hot)

for i in char_dataset.take(5):
    indices = tf.argmax(i, axis=-1).numpy()
    chars = idx2char[indices]
    print(chars)


# In[16]:


sequences = char_dataset.batch(seq_length+1, drop_remainder=True) #101

for item in sequences.take(5):
    #print(idx2char[item.numpy()])
    indices = tf.argmax(item , axis=-1).numpy()
    chars = idx2char[indices]
    print(repr(''.join(chars))) 


# In[114]:


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

#dataset contains pairs of input and target sequences
dataset = sequences.map(split_input_target)
dataset


# In[23]:


for input_example, target_example in  dataset.take(1):
    indices = tf.argmax(input_example, axis=-1).numpy()
    indices2 = tf.argmax(target_example, axis=-1).numpy()
    chars = "".join(idx2char[indices])
    chars2 = "".join(idx2char[indices2])
    
    print ('Input data: ', repr(''.join(chars)))
    print ('Target data:', repr(''.join(chars2)))


# In[24]:


for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    indices = tf.argmax(input_idx, axis=-1).numpy()
    indices2 = tf.argmax(target_idx, axis=-1).numpy()
    chars = "".join(idx2char[indices])
    chars2 = "".join(idx2char[indices2])
    print("  input: {} ({:s})".format(indices, repr(chars)))
    print("  expected output: {} ({:s})".format(indices2, repr(chars2)))


# ### Create training batches

# In[25]:


BATCH_SIZE = 64
examples_per_epoch=len(text)//(seq_length)
#BUFFER_SIZE = 10000

dataset_shuffle = dataset.shuffle(examples_per_epoch).batch(BATCH_SIZE, drop_remainder=True)
#drop_remainder=True 如果最後一個批次的數據樣本數不足一個完整的批次（小於batch size），則將該批次丟棄。

dataset_shuffle #這表示每個批次的元素有兩個部分 模型處理每個序列的大小為 100，並且每個批次有 64 個序列。


# In[26]:


examples_per_epoch


# ### Build The Model

# In[27]:


def seq_len_split(seq_length, batch_size):
    char_dataset = tf.data.Dataset.from_tensor_slices(train_one_hot)
    char_dataset_valid = tf.data.Dataset.from_tensor_slices(valid_one_hot)
    
    sequences = tf.data.Dataset.batch(char_dataset, seq_length + 1, drop_remainder = True)
    sequences_valid = tf.data.Dataset.batch(char_dataset_valid, seq_length + 1, drop_remainder = True)
    
    dataset = sequences.map(split_input_target)
    dataset_valid = sequences_valid.map(split_input_target)
    
    dataset_shuffle = dataset.shuffle(examples_per_epoch)
    train_data1 = dataset_shuffle.batch(batch_size, drop_remainder = True)
    valid_data1 = dataset_valid.batch(batch_size, drop_remainder=True)

    return train_data1, valid_data1


# In[28]:


def model_rnn(rnn_unit, batch_size):

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.SimpleRNN(
        input_dim=len(vocab),
        batch_size=batch_size,
        units=rnn_unit,
        return_sequences=True,
        stateful=True,
        recurrent_initializer='zeros'
    ))

    model.add(tf.keras.layers.Dense(len(vocab), activation='softmax'))

    return model


# In[29]:


def model_lstm(rnn_unit, batch_size):

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM(
        input_dim=len(vocab),
        batch_size=batch_size,
        units=rnn_unit,
        return_sequences=True,
        stateful=True,
        recurrent_initializer='zeros'
    ))

    model.add(tf.keras.layers.Dense(len(vocab), activation='softmax'))

    return model


# ## 1. Construct a standard RNN 
# 

# ### (1) network architecture

# ### RNN standard Model (seq_length=100, rnn_unit=512)

# In[69]:


train_data1, valid_data1= seq_len_split(seq_length=100, batch_size=64)
model_rnn_1 = model_rnn(rnn_unit=512, batch_size=64)
model_rnn_1.summary()


# In[73]:


model_rnn_1.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')


# In[74]:


checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath = os.path.join('model_rnn_1/checkpoints', 'ckpt_{epoch}'),
    save_weights_only=True
)


# In[75]:


model_rnn_1_history = model_rnn_1.fit(
    x = train_data1,
    validation_data = valid_data1,
    epochs = 60,
    callbacks=[checkpoint_callback]
)


# In[76]:


model_rnn_1.save("model_rnn_1.h5")


# In[78]:


history_rnn_1 = pd.DataFrame(model_rnn_1_history.history)
with open('history/history_rnn_1.json', 'w') as f:
    history_rnn_1.to_json(f)


# ### (2) learning curve

# We minimize the bits-per-character (BPC):  
# $$BPC=-\frac{1}{T}\sum_{t=1}^T\sum_{k=1}^K t_{t,k} log y_{t,k}(x_t,w)$$
# where y denotes the output from RNN and t denotes the corresponding target value, and K is the length of the one-hot vector.

# Because we use mini-batch of input data, consider the following objective function:
# $$E(w)=-\frac{1}{NT}\sum_{n=1}^N\sum_{t=1}^T\sum_{k=1}^K t_{t,k} log y_{t,k}^n(x_t^n,w)$$
# where N is the batch size, T is time step and K is the length of the one-hot vector.

# In[79]:


with open('./history/history_rnn_1.json', 'r') as f:
    history_rnn_1 = pd.read_json(f)


# In[89]:


import matplotlib.pyplot as plt

epochs = range(1, 61) 

plt.figure(figsize=(10, 3))
plt.subplots_adjust(wspace=0.3)

# 訓練損失
plt.subplot(1, 2, 1)
plt.plot(epochs,history_rnn_1['loss'], label='Training Loss')
plt.plot(epochs,history_rnn_1['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning curve of standard RNN model')
plt.legend()

# 訓練準確度
plt.subplot(1, 2, 2)
plt.plot(epochs,history_rnn_1['accuracy'], label='Training Accuracy')
plt.plot(epochs,history_rnn_1['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy curve of standard RNN model')
plt.legend()

plt.show()


# In the left-hand plot, we can see that 
# * the training loss is lower than the validation loss.
# * After epoch 30, two losses are tend to be stable.
# 
# In the right-hand plot, we can see that 
# * the training accuracy is higher than the validation accuracy.
# * After epoch 30, the accuracy rates are tend to be stable.

# ### (3) training error rate

#  $training\ error\ rate =1 -  training\ accuracy\ rate$

# We take the result of the last epoch (epoch=60).

# In[81]:


# 最後一個 epoch 的訓練training error rate
print(f"training error rate: {1-history_rnn_1['accuracy'].iloc[-1]}")


# ### (4) validation error rate

#  $ validation\ error\ rate =1 -   validation\ accuracy\ rate$

# In[82]:


# 最後一個 epoch 的validation error rate
print(f"validation error rate: {1-history_rnn_1['val_accuracy'].iloc[-1]}")


# ## 2. Choose 5 breakpoints during your training process to show how well your network learns through more epochs. Feed some part of your training text into RNN and show the text output.
# 

# We choose 5 breakpoints: epoch = $[1,15,30,45,60]$ during the training process, and show some part of output below.

# In[161]:


dataset = sequences.map(split_input_target)
dataset            #(seq_length, vocab_size), (seq_length, vocab_size)


# In[197]:


dataset_list = list(dataset.as_numpy_iterator())
len(dataset_list)  


# In[259]:


selected_batch=dataset_list[43081] #43081
len(selected_batch)


# In[331]:


selected_batch[0]


# In[260]:


selected_batch[0].shape #有100個字 67種字元


# In[230]:


rnn_model_1_pred = model_rnn(512,1)


# In[262]:


print("\n\n--------------------prediction---------------------")
def predict_print_output(model, ckpt_epochs):
    for epoch in ckpt_epochs:

        checkpoint_path = f'model_rnn_1/checkpoints/ckpt_{epoch}'
        
        # 載入模型權重
        model.load_weights(checkpoint_path)
        
        # 重置模型狀態
        model.reset_states()

        # 使用模型進行預測  rnn_model_1_pred(tf.expand_dims(selected_batch[0], 0)) #100*67
        predict = model(tf.expand_dims(selected_batch[0], 0)) #給他一個seq去預測 (100*67個機率)
        predict = predict.numpy()
        predict = predict.argmax(2)
        predict_result = predict.squeeze()

        print(f"\n\nOutput data (Epoch {epoch}): \n'", sep="", end="")
        for item in predict_result:
            print(int_to_vocab[item], sep="", end="")


predict_print_output(rnn_model_1_pred, ckpt_epochs=[1, 15, 30, 45, 60])

print("\n\n--------------------true---------------------")
print("Input data:")

for item in selected_batch[0]:
    idx=item.argmax()
    char=int_to_vocab[idx]
    print(char, sep="", end="")

print("\n\nTarget data:")

for item in selected_batch[1]:
    idx=item.argmax()
    char=int_to_vocab[idx]
    print(char, sep="", end="")


# * Epoch 1:
# 
# Output seems random and doesn't make much sense. The model is likely guessing.
# 
# * Epoch 15, 30
# 
# Some improvement, with English words appearing, for example, "be" "to" "and". Still not very meaningful.
# 
# * Epoch 45:
# 
# Output becomes more meaningful. "Shall" matches the target data.
# 
# * Epoch 60:
# More output makes sense, for example, "Shall" matches the target data. Overall, words are more similar to the target data.
# 
# 
# In summary, as training progresses, the network is getting better at generating meaningful text through more epochs.  

# ### RNN Model 2 (seq_len=70, rnn_unit=512)

# In[99]:


train_data2, valid_data2= seq_len_split(seq_length=70, batch_size=64)
model_rnn_2 = model_rnn(rnn_unit=256, batch_size=64)
model_rnn_2.summary()
model_rnn_2.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')


# In[100]:


checkpoint_callback_rnn_2=tf.keras.callbacks.ModelCheckpoint(
    filepath = os.path.join('model_rnn_2/checkpoints', 'ckpt_{epoch}'),
    save_weights_only=True
)


# In[101]:


model_rnn_2_history = model_rnn_2.fit(
    x = train_data2,
    validation_data = valid_data2,
    epochs = 60,
    callbacks=[checkpoint_callback_rnn_2]
)


# In[ ]:


model_rnn_2.save("model_rnn_2.h5")

history_rnn_2 = pd.DataFrame(model_rnn_2_history.history)
with open('./history/history_rnn2.json', 'w') as f:
    history_rnn_2.to_json(f)


# ### RNN Model 3 (seq_len=30, rnn_unit=512)

# In[ ]:


train_data3, valid_data3= seq_len_split(seq_length=30, batch_size=64)
model_rnn_3 = model_rnn(rnn_unit=512, batch_size=64)
model_rnn_3.summary()
model_rnn_3.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')


# In[25]:


checkpoint_callback_rnn_3=tf.keras.callbacks.ModelCheckpoint(
    filepath = os.path.join('model_rnn_3/checkpoints', 'ckpt_{epoch}'),
    save_weights_only=True
)


# In[105]:


model_rnn_3_history = model_rnn_3.fit(
    x = train_data3,
    validation_data = valid_data3,
    epochs = 60,
    callbacks=[checkpoint_callback_rnn_3]
)


# In[ ]:


model_rnn_3.save("model_rnn_3.h5")

history_rnn_3 = pd.DataFrame(model_rnn_3_history.history)
with open('./history/history_rnn3.json', 'w') as f:
    history_rnn_3.to_json(f)


# ### RNN Model 4 (seq_len=100, rnn_unit=1024)

# In[30]:


train_data4, valid_data4= seq_len_split(seq_length=100, batch_size=64)
model_rnn_4 = model_rnn(rnn_unit=256, batch_size=64)
model_rnn_4.summary()
model_rnn_4.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')


# In[31]:


checkpoint_callback_rnn_4=tf.keras.callbacks.ModelCheckpoint(
    filepath = os.path.join('model_rnn_4/checkpoints', 'ckpt_{epoch}'),
    save_weights_only=True
)


# In[32]:


model_rnn_4_history = model_rnn_4.fit(
    x = train_data4,
    validation_data = valid_data4,
    epochs = 60,
    callbacks=[checkpoint_callback_rnn_4]
)


# In[35]:


model_rnn_4.save("model_rnn_4.h5")

history_rnn_4 = pd.DataFrame(model_rnn_4_history.history)
with open('./history/history_rnn4.json', 'w') as f:
    history_rnn_4.to_json(f)


# ### RNN Model 5 (seq_len=100, rnn_unit=256)

# In[36]:


train_data5, valid_data5= seq_len_split(seq_length=100, batch_size=64)
model_rnn_5 = model_rnn(rnn_unit=128, batch_size=64)
model_rnn_5.summary()
model_rnn_5.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')


# In[37]:


checkpoint_callback_rnn_5=tf.keras.callbacks.ModelCheckpoint(
    filepath = os.path.join('model_rnn_5/checkpoints', 'ckpt_{epoch}'),
    save_weights_only=True
)


# In[38]:


model_rnn_5_history = model_rnn_5.fit(
    x = train_data5,
    validation_data = valid_data5,
    epochs = 60,
    callbacks=[checkpoint_callback_rnn_5]
)


# In[39]:


model_rnn_5.save("model_rnn_5.h5")

history_rnn_5 = pd.DataFrame(model_rnn_5_history.history)
with open('./history/history_rnn5.json', 'w') as f:
    history_rnn_5.to_json(f)


# ## 3. Compare the results of choosing different size of hidden states and sequence length by plotting the training loss vs. different parameters.

# In[63]:


import json

with open(f'./history/history_rnn_1.json', 'r') as f:
    history_rnn_1 = pd.DataFrame(json.load(f))   
with open(f'./history/history_rnn2.json', 'r') as f:
    history_rnn_2 = pd.DataFrame(json.load(f))
with open(f'./history/history_rnn3.json', 'r') as f:
    history_rnn_3 = pd.DataFrame(json.load(f))
with open(f'./history/history_lstm_5.json', 'r') as f:
    history_lstm_5 = pd.DataFrame(json.load(f))


# In the left plot below, we compare the results of choosing different sequence length=(30,70,100) under fixed size of hidden states=512.
# 
# In the right plot below, we compare the results of choosing different size of hidden states=(128,256,512) under fixed sequence length=100.

# In[91]:


epochs = range(1, 61)  # Assuming all models were trained for the same number of epochs

plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)
plt.subplots_adjust(wspace=0.3)

plt.plot(epochs, history_rnn_1['loss'], label='RNN Model 1 (seq_length=100)')
plt.plot(epochs, history_rnn_2['loss'], label='RNN Model 2 (seq_length=70)')
plt.plot(epochs, history_rnn_3['loss'], label='RNN Model 3 (seq_length=30)')

plt.title('Training Loss Comparison (hidden unit=512)')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, history_rnn_1['loss'], label='RNN Model 1 (hidden size=512)')
plt.plot(epochs, history_rnn_4['loss'], label='RNN Model 4 (hidden size=256)')
plt.plot(epochs, history_rnn_5['loss'], label='RNN Model 5 (hidden size=128)')

plt.title('Training Loss Comparison (seq_length=100)')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()

plt.show()


# According to the left plot, we can find that:
# * When the sequence length is increased under a fixed hidden unit, the training loss tends to be lower.
# 
# According to the right plot, we can find that:
# * When the size of hidden states is increased under a fixed seauence length, the training loss tends to be lower.
# 

# ## 4. Construct another RNN with LSTM then redo 1. to 3. Also discuss the difference of the results between standard RNN and LSTM.

# ### (1) network architecture

# ### LSTM Model 1

# In[40]:


train_data1, valid_data1= seq_len_split(seq_length=100, batch_size=64)
model_lstm_1 = model_lstm(rnn_unit=512, batch_size=64)
model_lstm_1.summary()
model_lstm_1.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')


# In[41]:


checkpoint_callback_lstm_1=tf.keras.callbacks.ModelCheckpoint(
    filepath = os.path.join('model_lstm_1/checkpoints', 'ckpt_{epoch}'),
    save_weights_only=True
)


# In[42]:


model_lstm_1_history = model_lstm_1.fit(
    x = train_data1,
    validation_data = valid_data1,
    epochs = 60,
    callbacks=[checkpoint_callback_lstm_1]
)


# In[43]:


model_lstm_1.save("model_lstm_1.h5")

history_lstm_1 = pd.DataFrame(model_lstm_1_history.history)
with open('./history/history_lstm_1.json', 'w') as f:
    history_lstm_1.to_json(f)


# ### (2) learning curve

# In[77]:


with open('./history/history_lstm_1.json', 'r') as f:
    history_lstm_1 = pd.read_json(f)


# In[79]:


import matplotlib.pyplot as plt

# 繪製訓練損失和準確度
plt.figure(figsize=(10, 3))

# 訓練損失
plt.subplot(1, 2, 1)
plt.plot(history_lstm_1['loss'], label='Training Loss')
plt.plot(history_lstm_1['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss') ,
plt.title('Learning curve of standard LSTM model')

plt.legend()

# 訓練準確度
plt.subplot(1, 2, 2)
plt.plot(history_lstm_1['accuracy'], label='Training Accuracy')
plt.plot(history_lstm_1['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy curve of standard LSTM model')
plt.legend()

plt.show()


# ### (3) training error rate

#  $training\ error\ rate =1 -  training\ accuracy\ rate$

# In[83]:


# 最後一個 epoch 的訓練training error rate
print(f"training error rate: {1-history_lstm_1['accuracy'].iloc[-1]}")


# The training error rate is lower than that of RNN Model 1.

# ### (4) validation error rate

# In[84]:


# 最後一個 epoch 的validation error rate
print(f"validation error rate: {1-history_lstm_1['val_accuracy'].iloc[-1]}")


# ### Choose 5 breakpoints during your training process to show how well your network learns through more epochs. Feed some part of your training text into LSTM and show the text output.

# We choose 5 breakpoints: epoch = $[1,15,30,45,60]$ during the training process, and show some part of output below.

# In[596]:


dataset = sequences.map(split_input_target)
dataset            #(seq_length, vocab_size), (seq_length, vocab_size)


# In[597]:


dataset_list = list(dataset.as_numpy_iterator())
len(dataset_list)  


# In[598]:


selected_batch=dataset_list[43081] #43081
len(selected_batch)


# In[599]:


selected_batch[0]


# In[600]:


selected_batch[0].shape #有100個字 67種字元


# In[601]:


lstm_model_1_pred = model_lstm(512,1)


# In[602]:


print("\n\n--------------------prediction---------------------")
def predict_print_output(model, ckpt_epochs):
    for epoch in ckpt_epochs:

        checkpoint_path = f'model_lstm_1/checkpoints/ckpt_{epoch}'
        
        # 載入模型權重
        model.load_weights(checkpoint_path)
        
        # 重置模型狀態
        model.reset_states()

        # 使用模型進行預測  rnn_model_1_pred(tf.expand_dims(selected_batch[0], 0)) #100*67
        predict = model(tf.expand_dims(selected_batch[0], 0)) #給他一個seq去預測 (100*67個機率)
        predict = predict.numpy()
        predict = predict.argmax(2)
        predict_result = predict.squeeze()

        print(f"\n\nOutput data (Epoch {epoch}): \n'", sep="", end="")
        for item in predict_result:
            print(int_to_vocab[item], sep="", end="")


predict_print_output(lstm_model_1_pred, ckpt_epochs=[1, 15, 30, 45, 60])

print("\n\n--------------------true---------------------")
print("Input data:")

for item in selected_batch[0]:
    idx=item.argmax()
    char=int_to_vocab[idx]
    print(char, sep="", end="")

print("\n\nTarget data:")

for item in selected_batch[1]:
    idx=item.argmax()
    char=int_to_vocab[idx]
    print(char, sep="", end="")


# * Epoch 1:
# 
# Output seems random and doesn't make much sense. The model is likely guessing.
# 
# * Epoch 15, 30
# 
# Some improvement, with English words appearing, for example, "and" "be" "in". Still not very meaningful.
# 
# * Epoch 45:
# 
# Output becomes more meaningful. "drums" matches the target data.
# 
# * Epoch 60:
# Output becomes more meaningful. "trumpets" matches the target data.
# Overall, words are more similar to the target data.
# 
# 
# In summary, as training progresses, the network is getting better at generating meaningful text through more epochs. 
# 
# The results generated by LSTM in Epoch 60 are more similar to the target data than those of the RNN model.

# ### LSTM Model 2

# In[44]:


train_data2, valid_data2= seq_len_split(seq_length=70, batch_size=64)
model_lstm_2 = model_lstm(rnn_unit=512, batch_size=64)
model_lstm_2.summary()
model_lstm_2.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')


# In[45]:


checkpoint_callback_lstm_2=tf.keras.callbacks.ModelCheckpoint(
    filepath = os.path.join('model_lstm_2/checkpoints', 'ckpt_{epoch}'),
    save_weights_only=True
)


# In[46]:


model_lstm_2_history = model_lstm_2.fit(
    x = train_data2,
    validation_data = valid_data2,
    epochs = 60,
    callbacks=[checkpoint_callback_lstm_2]
)


# In[47]:


model_lstm_2.save("model_lstm_2.h5")

history_lstm_2 = pd.DataFrame(model_lstm_2_history.history)
with open('./history/history_lstm_2.json', 'w') as f:
    history_lstm_2.to_json(f)


# ### LSTM Model 3

# In[48]:


train_data3, valid_data3= seq_len_split(seq_length=30, batch_size=64)
model_lstm_3 = model_lstm(rnn_unit=512, batch_size=64)
model_lstm_3.summary()
model_lstm_3.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')


# In[49]:


checkpoint_callback_lstm_3=tf.keras.callbacks.ModelCheckpoint(
    filepath = os.path.join('model_lstm_3/checkpoints', 'ckpt_{epoch}'),
    save_weights_only=True
)


# In[50]:


model_lstm_3_history = model_lstm_3.fit(
    x = train_data3,
    validation_data = valid_data3,
    epochs = 60,
    callbacks=[checkpoint_callback_lstm_3]
)


# In[51]:


model_lstm_3.save("model_lstm_3.h5")

history_lstm_3 = pd.DataFrame(model_lstm_3_history.history)
with open('./history/history_lstm_3.json', 'w') as f:
    history_lstm_3.to_json(f)


# ### LSTM Model 4

# In[52]:


train_data4, valid_data4= seq_len_split(seq_length=100, batch_size=64)
model_lstm_4 = model_lstm(rnn_unit=256, batch_size=64)
model_lstm_4.summary()
model_lstm_4.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')


# In[53]:


checkpoint_callback_lstm_4=tf.keras.callbacks.ModelCheckpoint(
    filepath = os.path.join('model_lstm_4/checkpoints', 'ckpt_{epoch}'),
    save_weights_only=True
)


# In[54]:


model_lstm_4_history = model_lstm_4.fit(
    x = train_data4,
    validation_data = valid_data4,
    epochs = 60,
    callbacks=[checkpoint_callback_lstm_4]
)


# In[55]:


model_lstm_4.save("model_lstm_4.h5")

history_lstm_4 = pd.DataFrame(model_lstm_4_history.history)
with open('./history/history_lstm_4.json', 'w') as f:
    history_lstm_4.to_json(f)


# ### LSTM Model 5

# In[ ]:


train_data5, valid_data5= seq_len_split(seq_length=100, batch_size=64)
model_lstm_5 = model_lstm(rnn_unit=128, batch_size=64)
model_lstm_5.summary()
model_lstm_5.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')


# In[ ]:


checkpoint_callback_lstm_5=tf.keras.callbacks.ModelCheckpoint(
    filepath = os.path.join('model_lstm_5/checkpoints', 'ckpt_{epoch}'),
    save_weights_only=True
)


# In[109]:


model_lstm_5_history = model_lstm_5.fit(
    x = train_data5,
    validation_data = valid_data5,
    epochs = 60,
    callbacks=[checkpoint_callback_lstm_5]
)


# In[ ]:


model_lstm_5.save("model_lstm_5.h5")

history_lstm_5 = pd.DataFrame(model_lstm_5_history.history)
with open('./history/history_lstm_5.json', 'w') as f:
    history_lstm_5.to_json(f)


# ### Compare the results of choosing different size of hidden states and sequence length by plotting the training loss vs. different parameters.

# In the left plot below, we compare the results of choosing different sequence length=(30,70,100) under fixed size of hidden states=512.
# 
# In the right plot below, we compare the results of choosing different size of hidden states=(128,256,512) under fixed sequence length=100.

# In[90]:


epochs = range(1, 61)  # Assuming all models were trained for the same number of epochs

plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)
plt.subplots_adjust(wspace=0.3)

plt.plot(epochs, history_lstm_1['loss'], label='LSTM Model 1 (seq_length=100)')
plt.plot(epochs, history_lstm_2['loss'], label='LSTM Model 2 (seq_length=70)')
plt.plot(epochs, history_lstm_3['loss'], label='LSTM Model 3 (seq_length=30)')

plt.title('Training Loss Comparison (hidden unit=512)')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, history_lstm_1['loss'], label='LSTM Model 1 (hidden size=512)')
plt.plot(epochs, history_lstm_4['loss'], label='LSTM Model 4 (hidden size=256)')
plt.plot(epochs, history_lstm_5['loss'], label='LSTM Model 5 (hidden size=128)')

plt.title('Training Loss Comparison (seq_length=100)')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()

plt.show()


# According to the left plot, we can find that:
# * When the sequence length is increased under a fixed hidden unit, the training loss tends to be lower.
# 
# According to the right plot, we can find that:
# * When the size of hidden states is increased under a fixed seauence length, the training loss tends to be lower.
# 

# ###  Discuss the difference of the results between standard RNN and LSTM.
# 

# In[111]:


plt.figure(figsize=(10, 3.5))
plt.subplots_adjust(wspace=0.3)

plt.subplot(1, 2, 1)
rnn_123_loss=[history_rnn_1['loss'].iloc[-1],history_rnn_2['loss'].iloc[-1],history_rnn_3['loss'].iloc[-1]]
lstm_123_loss=[history_lstm_1['loss'].iloc[-1],history_lstm_2['loss'].iloc[-1],history_lstm_3['loss'].iloc[-1]]

plt.plot([100,70,30],rnn_123_loss, marker='o', label='RNN Model')
plt.plot([100,70,30],lstm_123_loss, marker='o', label='LSTM Model')

plt.title('Training loss at Epoch 60 (hidden size=512))')
plt.xlabel('seqence length')
plt.ylabel('Training loss')
plt.legend()

plt.subplot(1, 2, 2)
rnn_145_loss=[history_rnn_1['loss'].iloc[-1],history_rnn_4['loss'].iloc[-1],history_rnn_5['loss'].iloc[-1]]
lstm_145_loss=[history_lstm_1['loss'].iloc[-1],history_lstm_4['loss'].iloc[-1],history_lstm_5['loss'].iloc[-1]]

plt.plot([512,256,128],rnn_145_loss, marker='o', label='RNN Model')
plt.plot([512,256,128],lstm_145_loss, marker='o', label='LSTM Model')

plt.title('Training loss at Epoch 60 (seq_length=100)')
plt.xlabel('hidden size')
plt.ylabel('Training loss')
plt.legend()

plt.show()


# According to the left plot, we can find that:
# *  With a fixed number of hidden size, regardless of the sequence length=(30,70,100), the RNN's loss is consistently higher than that of the LSTM.
# 
# According to the right plot, we can find that:
# *  With a fixed sequence length, regardless of the hidden size=(128,256,512), the RNN's loss is consistently higher than that of the LSTM.
# 

# ## 5. Use RNN or LSTM to generate some words by priming the model with a word related to your dataset. Priming the model means giving it some input text to create context and then take the output of the RNN. For example, use ”JULIET” as the prime text of Shakespeare dataset and run the model to generate 10 to 15 lines of output.

# * Use LSTM model 1 (seq_length=100, rnn_unit=512) to generate some words.

# In[592]:


def generate_text_with_primer(model, primer_text, num_char):
    # Convert the primer text to a sequence of indices
    primer_seq = [vocab_to_int[char] for char in primer_text]
    model.reset_states()

    generated_text = primer_text #'JULIET'

    for _ in range(num_char):
        # Use the model to predict the next character
        primer_seq_one_hot = tf.one_hot(primer_seq, len(vocab))
        predict = model(tf.expand_dims(primer_seq_one_hot, 0)).numpy().argmax(2)

        # Take the last predicted character
        predicted_char = int_to_vocab[predict[0, -1]]

        # Add the predicted character to the generated text
        generated_text += predicted_char

        # Update the primer sequence for the next iteration
        primer_seq = [vocab_to_int[predicted_char]]

    return generated_text


# In[593]:


# Primer text
primer_text = "JULIET"

checkpoint_path = f'model_lstm_1/checkpoints/ckpt_40'
lstm_model_1_pred = model_lstm(512,1)
lstm_model_1_pred.load_weights(checkpoint_path)


# In[594]:


output_prime=generate_text_with_primer(lstm_model_1_pred, primer_text, num_char=583)
print("\n\n--------------------LSTM Prediction with Primer---------------------")
print(output_prime, sep="", end="")


# * The model generates text in Shakespearean style, capturing different character voices and maintaining reasonable context. 
# * Despite some errors in individual words, the majority of the text is meaningful and makes sense. 
# * Compare to the predicted text generated by RNN Model 1 (in Part 2) , this LSTM Model has better performance.
