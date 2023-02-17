## regression model

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers


# load the dataset - linear
dataset_name_lin = '08.Version1/Data/GDP_86_linear_prior.csv'
data_post_name_lin = '08.Version1/Data/GDP_86_linear_post.csv'

dataset_lin = pd.read_csv(dataset_name_lin, delimiter=',')
post_set_lin = pd.read_csv(data_post_name_lin, delimiter=',')

# load the datasets - ND
dataset_name_ND = '08.Version1/Data/GDP_86_ND_prior.csv'
data_post_name_ND = '08.Version1/Data/GDP_86_ND_post.csv'

dataset_nd = pd.read_csv(dataset_name_ND, delimiter=',')
post_set_nd = pd.read_csv(data_post_name_ND, delimiter=',')

# split out right columsn for predictions later
prior_red_lin = dataset_lin.iloc[:,0:85]
post_red_lin = post_set_lin.iloc[:,0:85]
prior_red_nd = dataset_nd.iloc[:,0:85]
post_red_nd = post_set_nd.iloc[:,0:85]


# Split into training and test
train_dataset_lin = dataset_lin.sample(frac=0.8, random_state=0)
test_dataset_lin = dataset_lin.drop(train_dataset_lin.index)

train_dataset_nd = dataset_nd.sample(frac=0.8, random_state=0)
test_dataset_nd = dataset_nd.drop(train_dataset_nd.index)

# Reduce to right number of columns
train_dataset_red_lin = train_dataset_lin.iloc[:,0:85]
test_dataset_red_lin = test_dataset_lin.iloc[:,0:85]

train_dataset_red_nd = train_dataset_nd.iloc[:,0:85]
test_dataset_red_nd = test_dataset_nd.iloc[:,0:85]

# save labels (y targets)
train_labels_lin = train_dataset_lin.pop('Australia')
test_labels_lin = test_dataset_lin.pop('Australia')

train_labels_nd = train_dataset_nd.pop('Australia')
test_labels_nd = test_dataset_nd.pop('Australia')


# define the models
# Model 1
## Base Model MSE
def base_model_mse():
    model = keras.Sequential([
        layers.Dense(85, input_shape=(85,), activation='relu'),
        layers.Dense(85, activation='relu'),
        layers.Dense(1) 
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss ='mse',
                  optimizer=optimizer,
                  metrics = ['mae','mse'])
    
    return model

# Model 2
## Base Model MAE
def base_model_mae():
    model = keras.Sequential([
        layers.Dense(85, input_shape=(85,), activation='relu'),
        layers.Dense(85, activation='relu'),
        layers.Dense(1) 
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss ='mae',
                  optimizer=optimizer,
                  metrics = ['mae','mse'])
    
    return model

# Model 3
## Deep Model MSE
def deep_model_mse():
    model = keras.Sequential([
        layers.Dense(85, input_shape=(85,), activation='relu'),
        layers.Dense(68, activation='relu'),
        layers.Dense(51, activation='relu'),
        layers.Dense(34, activation='relu'),
        layers.Dense(17, activation='relu'),
        layers.Dense(1) 
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss ='mse',
                  optimizer=optimizer,
                  metrics = ['mae','mse'])
    
    return model

# Model 4
## Deep Model MAE
def deep_model_mae():
    model = keras.Sequential([
        layers.Dense(85, input_shape=(85,), activation='relu'),
        layers.Dense(68, activation='relu'),
        layers.Dense(51, activation='relu'),
        layers.Dense(34, activation='relu'),
        layers.Dense(17, activation='relu'),
        layers.Dense(1) 
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss ='mae',
                  optimizer=optimizer,
                  metrics = ['mae','mse'])
    
    return model

# Model 5
## Wide Model MSE
def wide_model_mse():
    model = keras.Sequential([
        layers.Dense(85, input_shape=(85,), activation='relu'),
        layers.Dense(170, activation='relu'),
        layers.Dense(42, activation='relu'),
        layers.Dense(1) 
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss ='mse',
                  optimizer=optimizer,
                  metrics = ['mae','mse'])
    
    return model

# Model 6
## Wide Model MAE
def wide_model_mae():
    model = keras.Sequential([
        layers.Dense(85, input_shape=(85,), activation='relu'),
        layers.Dense(170, activation='relu'),
        layers.Dense(42, activation='relu'),
        layers.Dense(1) 
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss ='mae',
                  optimizer=optimizer,
                  metrics = ['mae','mse'])
    
    return model

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch %100 == 0: print('')
        print('.', end = '')

## Linear Models
model_1a = base_model_mse()
model_2a = base_model_mae()
model_3a = deep_model_mse()
model_4a = deep_model_mae()
model_5a = wide_model_mse()
model_6a = wide_model_mae()

## ND Models
model_1b = base_model_mse()
model_2b = base_model_mae()
model_3b = deep_model_mse()
model_4b = deep_model_mae()
model_5b = wide_model_mse()
model_6b = wide_model_mae()


# number of epochs
EPOCHS = 5000

# run models
# 6 types of models
# 2 datasets

## run all linear dataset models
## Run Model 1a
history_1a=model_1a.fit(
    train_dataset_red_lin, train_labels_lin,
    epochs=EPOCHS, validation_split = 0.2, verbose =0,
    callbacks=[PrintDot()])

## Run Model 2a
history_2a=model_2a.fit(
    train_dataset_red_lin, train_labels_lin,
    epochs=EPOCHS, validation_split = 0.2, verbose =0,
    callbacks=[PrintDot()])

## Run Model 3a
history_3a=model_3a.fit(
    train_dataset_red_lin, train_labels_lin,
    epochs=EPOCHS, validation_split = 0.2, verbose =0,
    callbacks=[PrintDot()])

## Run Model 4a
history_4a=model_4a.fit(
    train_dataset_red_lin, train_labels_lin,
    epochs=EPOCHS, validation_split = 0.2, verbose =0,
    callbacks=[PrintDot()])

## Run Model 5a
history_5a=model_5a.fit(
    train_dataset_red_lin, train_labels_lin,
    epochs=EPOCHS, validation_split = 0.2, verbose =0,
    callbacks=[PrintDot()])

## Run Model 6a
history_6a=model_6a.fit(
    train_dataset_red_lin, train_labels_lin,
    epochs=EPOCHS, validation_split = 0.2, verbose =0,
    callbacks=[PrintDot()])


## run all ND dataset models
## Run Model 1b
history_1b=model_1b.fit(
    train_dataset_red_nd, train_labels_nd,
    epochs=EPOCHS, validation_split = 0.2, verbose =0,
    callbacks=[PrintDot()])

## Run Model 2b
history_2b=model_2b.fit(
    train_dataset_red_nd, train_labels_nd,
    epochs=EPOCHS, validation_split = 0.2, verbose =0,
    callbacks=[PrintDot()])

## Run Model 3b
history_3b=model_3b.fit(
    train_dataset_red_nd, train_labels_nd,
    epochs=EPOCHS, validation_split = 0.2, verbose =0,
    callbacks=[PrintDot()])

## Run Model 4b
history_4b=model_4b.fit(
    train_dataset_red_nd, train_labels_nd,
    epochs=EPOCHS, validation_split = 0.2, verbose =0,
    callbacks=[PrintDot()])

## Run Model 5b
history_5b=model_5b.fit(
    train_dataset_red_nd, train_labels_nd,
    epochs=EPOCHS, validation_split = 0.2, verbose =0,
    callbacks=[PrintDot()])

## Run Model 6b
history_6b=model_6b.fit(
    train_dataset_red_nd, train_labels_nd,
    epochs=EPOCHS, validation_split = 0.2, verbose =0,
    callbacks=[PrintDot()])  
    
## Save models
# Linear
model_1a.save("08.Version1/Models/model_1a.h5")    
model_2a.save("08.Version1/Models/model_2a.h5")    
model_3a.save("08.Version1/Models/model_3a.h5")    
model_4a.save("08.Version1/Models/model_4a.h5")    
model_5a.save("08.Version1/Models/model_5a.h5")    
model_6a.save("08.Version1/Models/model_6a.h5")    
# ND
model_1b.save("08.Version1/Models/model_1b.h5")
model_2b.save("08.Version1/Models/model_2b.h5")
model_3b.save("08.Version1/Models/model_3b.h5")
model_4b.save("08.Version1/Models/model_4b.h5")
model_5b.save("08.Version1/Models/model_5b.h5")
model_6b.save("08.Version1/Models/model_6b.h5")

## Histories as dataframe
hist_1a = pd.DataFrame(history_1a.history)
hist_2a = pd.DataFrame(history_2a.history)
hist_3a = pd.DataFrame(history_3a.history)
hist_4a = pd.DataFrame(history_4a.history)
hist_5a = pd.DataFrame(history_5a.history)
hist_6a = pd.DataFrame(history_6a.history)
hist_1b = pd.DataFrame(history_1b.history)
hist_2b = pd.DataFrame(history_2b.history)
hist_3b = pd.DataFrame(history_3b.history)
hist_4b = pd.DataFrame(history_4b.history)
hist_5b = pd.DataFrame(history_5b.history)
hist_6b = pd.DataFrame(history_6b.history)

## Include Epochs in hist
hist_1a['epoch'] = history_1a.epoch 
hist_2a['epoch'] = history_2a.epoch 
hist_3a['epoch'] = history_3a.epoch 
hist_4a['epoch'] = history_4a.epoch 
hist_5a['epoch'] = history_5a.epoch 
hist_6a['epoch'] = history_6a.epoch 
hist_1b['epoch'] = history_1b.epoch 
hist_2b['epoch'] = history_2b.epoch 
hist_3b['epoch'] = history_3b.epoch 
hist_4b['epoch'] = history_4b.epoch 
hist_5b['epoch'] = history_5b.epoch 
hist_6b['epoch'] = history_6b.epoch 

## Save Hist to CSV
hist_1a.to_csv('08.Version1/Histories/History_1a.csv',index=False)
hist_2a.to_csv('08.Version1/Histories/History_2a.csv',index=False)
hist_3a.to_csv('08.Version1/Histories/History_3a.csv',index=False)
hist_4a.to_csv('08.Version1/Histories/History_4a.csv',index=False)
hist_5a.to_csv('08.Version1/Histories/History_5a.csv',index=False)
hist_6a.to_csv('08.Version1/Histories/History_6a.csv',index=False)
hist_1b.to_csv('08.Version1/Histories/History_1b.csv',index=False)
hist_2b.to_csv('08.Version1/Histories/History_2b.csv',index=False)
hist_3b.to_csv('08.Version1/Histories/History_3b.csv',index=False)
hist_4b.to_csv('08.Version1/Histories/History_4b.csv',index=False)
hist_5b.to_csv('08.Version1/Histories/History_5b.csv',index=False)
hist_6b.to_csv('08.Version1/Histories/History_6b.csv',index=False)

## produce predictions of prior data
## Linear
test_pred_pre_1a = model_1a.predict(prior_red_lin)
test_pred_pre_2a = model_2a.predict(prior_red_lin)
test_pred_pre_3a = model_3a.predict(prior_red_lin)
test_pred_pre_4a = model_4a.predict(prior_red_lin)
test_pred_pre_5a = model_5a.predict(prior_red_lin)
test_pred_pre_6a = model_6a.predict(prior_red_lin)
## ND
test_pred_pre_1b = model_1b.predict(prior_red_nd)
test_pred_pre_2b = model_2b.predict(prior_red_nd)
test_pred_pre_3b = model_3b.predict(prior_red_nd)
test_pred_pre_4b = model_4b.predict(prior_red_nd)
test_pred_pre_5b = model_5b.predict(prior_red_nd)
test_pred_pre_6b = model_6b.predict(prior_red_nd)

## Change to Dataframe
test_predict_1a = pd.DataFrame(test_pred_pre_1a)
test_predict_2a = pd.DataFrame(test_pred_pre_2a)
test_predict_3a = pd.DataFrame(test_pred_pre_3a)
test_predict_4a = pd.DataFrame(test_pred_pre_4a)
test_predict_5a = pd.DataFrame(test_pred_pre_5a)
test_predict_6a = pd.DataFrame(test_pred_pre_6a)
test_predict_1b = pd.DataFrame(test_pred_pre_1b)
test_predict_2b = pd.DataFrame(test_pred_pre_2b)
test_predict_3b = pd.DataFrame(test_pred_pre_3b)
test_predict_4b = pd.DataFrame(test_pred_pre_4b)
test_predict_5b = pd.DataFrame(test_pred_pre_5b)
test_predict_6b = pd.DataFrame(test_pred_pre_6b)

##Save to CSV
test_predict_1a.to_csv('08.Version1/Predictions/pred_prior_1a.csv',index=False)
test_predict_2a.to_csv('08.Version1/Predictions/pred_prior_2a.csv',index=False)
test_predict_3a.to_csv('08.Version1/Predictions/pred_prior_3a.csv',index=False)
test_predict_4a.to_csv('08.Version1/Predictions/pred_prior_4a.csv',index=False)
test_predict_5a.to_csv('08.Version1/Predictions/pred_prior_5a.csv',index=False)
test_predict_6a.to_csv('08.Version1/Predictions/pred_prior_6a.csv',index=False)
test_predict_1b.to_csv('08.Version1/Predictions/pred_prior_1b.csv',index=False)
test_predict_2b.to_csv('08.Version1/Predictions/pred_prior_2b.csv',index=False)
test_predict_3b.to_csv('08.Version1/Predictions/pred_prior_3b.csv',index=False)
test_predict_4b.to_csv('08.Version1/Predictions/pred_prior_4b.csv',index=False)
test_predict_5b.to_csv('08.Version1/Predictions/pred_prior_5b.csv',index=False)
test_predict_6b.to_csv('08.Version1/Predictions/pred_prior_6b.csv',index=False)

## produce predictions of post data
# linear
post_pred_pre_1a = model_1a.predict(post_red_lin)
post_pred_pre_2a = model_2a.predict(post_red_lin)
post_pred_pre_3a = model_3a.predict(post_red_lin)
post_pred_pre_4a = model_4a.predict(post_red_lin)
post_pred_pre_5a = model_5a.predict(post_red_lin)
post_pred_pre_6a = model_6a.predict(post_red_lin)
#ND
post_pred_pre_1b = model_1b.predict(post_red_nd)
post_pred_pre_2b = model_2b.predict(post_red_nd)
post_pred_pre_3b = model_3b.predict(post_red_nd)
post_pred_pre_4b = model_4b.predict(post_red_nd)
post_pred_pre_5b = model_5b.predict(post_red_nd)
post_pred_pre_6b = model_6b.predict(post_red_nd)

## Change to Dataframe
post_pred_1a = pd.DataFrame(post_pred_pre_1a)
post_pred_2a = pd.DataFrame(post_pred_pre_2a)
post_pred_3a = pd.DataFrame(post_pred_pre_3a)
post_pred_4a = pd.DataFrame(post_pred_pre_4a)
post_pred_5a = pd.DataFrame(post_pred_pre_5a)
post_pred_6a = pd.DataFrame(post_pred_pre_6a)
post_pred_1b = pd.DataFrame(post_pred_pre_1b)
post_pred_2b = pd.DataFrame(post_pred_pre_2b)
post_pred_3b = pd.DataFrame(post_pred_pre_3b)
post_pred_4b = pd.DataFrame(post_pred_pre_4b)
post_pred_5b = pd.DataFrame(post_pred_pre_5b)
post_pred_6b = pd.DataFrame(post_pred_pre_6b)

## Save Post predictions
post_pred_1a.to_csv('08.Version1/Predictions/pred_post_1a.csv',index=False)
post_pred_2a.to_csv('08.Version1/Predictions/pred_post_2a.csv',index=False)
post_pred_3a.to_csv('08.Version1/Predictions/pred_post_3a.csv',index=False)
post_pred_4a.to_csv('08.Version1/Predictions/pred_post_4a.csv',index=False)
post_pred_5a.to_csv('08.Version1/Predictions/pred_post_5a.csv',index=False)
post_pred_6a.to_csv('08.Version1/Predictions/pred_post_6a.csv',index=False)
post_pred_1b.to_csv('08.Version1/Predictions/pred_post_1b.csv',index=False)
post_pred_2b.to_csv('08.Version1/Predictions/pred_post_2b.csv',index=False)
post_pred_3b.to_csv('08.Version1/Predictions/pred_post_3b.csv',index=False)
post_pred_4b.to_csv('08.Version1/Predictions/pred_post_4b.csv',index=False)
post_pred_5b.to_csv('08.Version1/Predictions/pred_post_5b.csv',index=False)
post_pred_6b.to_csv('08.Version1/Predictions/pred_post_6b.csv',index=False)


print('done')

