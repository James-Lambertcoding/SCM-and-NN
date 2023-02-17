## regression model

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers


# load the datasets - ND
dataset_name_ND = '11.Version5/Data/wb_auto_nn5_prior.csv'
data_post_name_ND = '11.Version5/Data/wb_auto_nn5_post.csv'

dataset_nd = pd.read_csv(dataset_name_ND, delimiter=',')
post_set_nd = pd.read_csv(data_post_name_ND, delimiter=',')

# split out right columsn for predictions later
prior_red_nd = dataset_nd.iloc[:,0:88]
post_red_nd = post_set_nd.iloc[:,0:88]


# Split into training and test
train_dataset_nd = dataset_nd.sample(frac=0.8, random_state=0)
test_dataset_nd = dataset_nd.drop(train_dataset_nd.index)

# Reduce to right number of columns
train_dataset_red_nd = train_dataset_nd.iloc[:,0:88]
test_dataset_red_nd = test_dataset_nd.iloc[:,0:88]

# save labels (y targets)
train_labels_nd = train_dataset_nd.pop('Australia')
test_labels_nd = test_dataset_nd.pop('Australia')


# define the models
# Model 1
## Base Model MSE
def base_model_mse():
    model = keras.Sequential([
        layers.Dense(88, input_shape=(88,), activation='relu'),
        layers.Dense(88, activation='relu'),
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
        layers.Dense(88, input_shape=(88,), activation='relu'),
        layers.Dense(88, activation='relu'),
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
        layers.Dense(88, input_shape=(88,), activation='relu'),
        layers.Dense(88, activation='relu'),
        layers.Dense(88, activation='relu'),
        layers.Dense(44, activation='relu'),
        layers.Dense(44, activation='relu'),
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
        layers.Dense(88, input_shape=(88,), activation='relu'),
        layers.Dense(88, activation='relu'),
        layers.Dense(88, activation='relu'),
        layers.Dense(44, activation='relu'),
        layers.Dense(44, activation='relu'),
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
        layers.Dense(88, input_shape=(88,), activation='relu'),
        layers.Dense(176, activation='relu'),
        layers.Dense(88, activation='relu'),
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
        layers.Dense(88, input_shape=(88,), activation='relu'),
        layers.Dense(176, activation='relu'),
        layers.Dense(88, activation='relu'),
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
EPOCHS = 1000

# run models
# 6 types of models
# 2 datasets


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
# ND
model_1b.save("11.Version5/Models/model_1b.h5")
model_2b.save("11.Version5/Models/model_2b.h5")
model_3b.save("11.Version5/Models/model_3b.h5")
model_4b.save("11.Version5/Models/model_4b.h5")
model_5b.save("11.Version5/Models/model_5b.h5")
model_6b.save("11.Version5/Models/model_6b.h5")

## Histories as dataframe
hist_1b = pd.DataFrame(history_1b.history)
hist_2b = pd.DataFrame(history_2b.history)
hist_3b = pd.DataFrame(history_3b.history)
hist_4b = pd.DataFrame(history_4b.history)
hist_5b = pd.DataFrame(history_5b.history)
hist_6b = pd.DataFrame(history_6b.history)

## Include Epochs in hist
hist_1b['epoch'] = history_1b.epoch 
hist_2b['epoch'] = history_2b.epoch 
hist_3b['epoch'] = history_3b.epoch 
hist_4b['epoch'] = history_4b.epoch 
hist_5b['epoch'] = history_5b.epoch 
hist_6b['epoch'] = history_6b.epoch 

## Save Hist to CSV
hist_1b.to_csv('11.Version5/Histories/History_1b.csv',index=False)
hist_2b.to_csv('11.Version5/Histories/History_2b.csv',index=False)
hist_3b.to_csv('11.Version5/Histories/History_3b.csv',index=False)
hist_4b.to_csv('11.Version5/Histories/History_4b.csv',index=False)
hist_5b.to_csv('11.Version5/Histories/History_5b.csv',index=False)
hist_6b.to_csv('11.Version5/Histories/History_6b.csv',index=False)

## produce predictions of prior data
## ND
test_pred_pre_1b = model_1b.predict(prior_red_nd)
test_pred_pre_2b = model_2b.predict(prior_red_nd)
test_pred_pre_3b = model_3b.predict(prior_red_nd)
test_pred_pre_4b = model_4b.predict(prior_red_nd)
test_pred_pre_5b = model_5b.predict(prior_red_nd)
test_pred_pre_6b = model_6b.predict(prior_red_nd)

## Change to Dataframe
test_predict_1b = pd.DataFrame(test_pred_pre_1b)
test_predict_2b = pd.DataFrame(test_pred_pre_2b)
test_predict_3b = pd.DataFrame(test_pred_pre_3b)
test_predict_4b = pd.DataFrame(test_pred_pre_4b)
test_predict_5b = pd.DataFrame(test_pred_pre_5b)
test_predict_6b = pd.DataFrame(test_pred_pre_6b)

##Save to CSV
test_predict_1b.to_csv('11.Version5/Predictions/pred_prior_1b.csv',index=False)
test_predict_2b.to_csv('11.Version5/Predictions/pred_prior_2b.csv',index=False)
test_predict_3b.to_csv('11.Version5/Predictions/pred_prior_3b.csv',index=False)
test_predict_4b.to_csv('11.Version5/Predictions/pred_prior_4b.csv',index=False)
test_predict_5b.to_csv('11.Version5/Predictions/pred_prior_5b.csv',index=False)
test_predict_6b.to_csv('11.Version5/Predictions/pred_prior_6b.csv',index=False)

## produce predictions of post data
# linear
#ND
post_pred_pre_1b = model_1b.predict(post_red_nd)
post_pred_pre_2b = model_2b.predict(post_red_nd)
post_pred_pre_3b = model_3b.predict(post_red_nd)
post_pred_pre_4b = model_4b.predict(post_red_nd)
post_pred_pre_5b = model_5b.predict(post_red_nd)
post_pred_pre_6b = model_6b.predict(post_red_nd)

## Change to Dataframe
post_pred_1b = pd.DataFrame(post_pred_pre_1b)
post_pred_2b = pd.DataFrame(post_pred_pre_2b)
post_pred_3b = pd.DataFrame(post_pred_pre_3b)
post_pred_4b = pd.DataFrame(post_pred_pre_4b)
post_pred_5b = pd.DataFrame(post_pred_pre_5b)
post_pred_6b = pd.DataFrame(post_pred_pre_6b)

## Save Post predictions
post_pred_1b.to_csv('11.Version5/Predictions/pred_post_1b.csv',index=False)
post_pred_2b.to_csv('11.Version5/Predictions/pred_post_2b.csv',index=False)
post_pred_3b.to_csv('11.Version5/Predictions/pred_post_3b.csv',index=False)
post_pred_4b.to_csv('11.Version5/Predictions/pred_post_4b.csv',index=False)
post_pred_5b.to_csv('11.Version5/Predictions/pred_post_5b.csv',index=False)
post_pred_6b.to_csv('11.Version5/Predictions/pred_post_6b.csv',index=False)


print('done')

