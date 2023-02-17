
# load and evaluate a saved model
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
 
# load model
model = load_model('05.Models/reg_model_long_norm.h5')


# load data
dataset_name = '01.Data/wb_select_prior_norm.csv'
data_post_name = '01.Data/wb_select_post_norm.csv'
dataset = pd.read_csv(dataset_name, delimiter=',')
post_set = pd.read_csv(data_post_name, delimiter=',')

#split data
dataset_rec = dataset.iloc[:,0:71]
post_set_rec = post_set.iloc[:,0:71]


model_name = 'long_model_norm'

# predict based on prior
test_pred_pre = model.predict(dataset_rec)
# convert to dataframe
test_predict = pd.DataFrame(test_pred_pre)
# save to CSV
test_predict.to_csv('06.Outputs\predictions_long_norm.csv',index=False)

# predict based on post
post_pred_pre = model.predict(post_set_rec)
# convert to dataframe
post_pred = pd.DataFrame(post_pred_pre)
# save to CSV
post_pred.to_csv('06.Outputs\post_predictions_long_norm.csv',index=False)
     
# done    
print('done')

