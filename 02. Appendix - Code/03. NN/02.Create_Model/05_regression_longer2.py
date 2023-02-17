## regression model

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers


# load the dataset
dataset_name = 'Data/wb_select_prior.csv'
dataset = pd.read_csv(dataset_name, delimiter=',')

dataset_rec = dataset.iloc[:,0:71]
output_actual = dataset.iloc[:,71]

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_dataset_red = train_dataset.iloc[:,0:71]
test_dataset_red = test_dataset.iloc[:,0:71]

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('Australia')
test_labels = test_features.pop('Australia')

model_name = 'long_model'

# define the base model
def build_model():
    model = keras.Sequential([
        layers.Dense(71, input_shape=(71,), activation='relu'),
        layers.Dense(142, activation='relu'),
        layers.Dense(142, activation='relu'),
        layers.Dense(142, activation='relu'),
        layers.Dense(142, activation='relu'),
        layers.Dense(71, activation='relu'),
        layers.Dense(1)
                
    
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss ='mse',
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



EPOCHS = 20000

history=model.fit(
    train_dataset_red, train_labels,
    epochs=EPOCHS, validation_data =(test_dataset_red,test_labels), verbose =0,
    callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch 

hist.to_csv('Outputs\History_base_long3.csv',index=False)

   
model.save("models/reg_model_long3.h5")    
    
print('done')

