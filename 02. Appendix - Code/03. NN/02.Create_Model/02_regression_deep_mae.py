## regression model

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers


# load the dataset
dataset_name = '01.Data/Version1/all_gdp_total_titled.csv'
data_post_name = '01.Data/Version1/post_int_gdp.csv'
dataset = pd.read_csv(dataset_name, delimiter=',')
post_set = pd.read_csv(data_post_name, delimiter=',')

dataset_rec = dataset.iloc[:,0:85]
post_set_rec = post_set.iloc[:,0:85]

output_actual = dataset.iloc[:,85]

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_dataset_red = train_dataset.iloc[:,0:85]
test_dataset_red = test_dataset.iloc[:,0:85]

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('Australia')
test_labels = test_features.pop('Australia')

model_name = 'base_mode'

# define the base model
def build_model():
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

model = build_model()



example_batch = train_dataset_red
example_results = model.predict(example_batch)

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch %100 == 0: print('')
        print('.', end = '')



EPOCHS = 5000

history=model.fit(
    train_dataset_red, train_labels,
    epochs=EPOCHS, validation_split = 0.2, verbose =0,
    callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch 

hist.to_csv('06.Outputs/Version1/History_base1_mae_deep.csv',index=False)

test_pred_pre = model.predict(dataset_rec)

test_predict = pd.DataFrame(test_pred_pre)

test_predict.to_csv('06.Outputs/Version1/predictions1_mae_deep.csv',index=False)

post_pred_pre = model.predict(post_set_rec)
post_pred = pd.DataFrame(post_pred_pre)

post_pred.to_csv('06.Outputs/Version1/post_predictions1_mae_deep.csv',index=False)
    
model.save("05.Models/reg_model_mae_deep.h5")    
    
print('done')

