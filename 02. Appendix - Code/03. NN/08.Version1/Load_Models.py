
# load and evaluate a saved model
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
 
 # load the datasets - ND
dataset_name_ND = '08.Version1/Data/GDP_86_ND_prior.csv'
data_post_name_ND = '08.Version1/Data/GDP_86_ND_post.csv'

dataset_nd = pd.read_csv(dataset_name_ND, delimiter=',')
post_set_nd = pd.read_csv(data_post_name_ND, delimiter=',')

prior_red_nd = dataset_nd.iloc[:,0:85]
post_red_nd = post_set_nd.iloc[:,0:85]
 
# load model ND
model_1b = load_model('08.Version1/Models/model_1b.h5')
model_2b = load_model('08.Version1/Models/model_2b.h5')
model_3b = load_model('08.Version1/Models/model_3b.h5')
model_4b = load_model('08.Version1/Models/model_4b.h5')
model_5b = load_model('08.Version1/Models/model_5b.h5')
model_6b = load_model('08.Version1/Models/model_6b.h5')


## produce predictions of post data
#ND
post_pred_pre_1b = model_1b.predict(post_red_nd)
post_pred_pre_2b = model_2b.predict(post_red_nd)
post_pred_pre_3b = model_3b.predict(post_red_nd)
post_pred_pre_4b = model_4b.predict(post_red_nd)
post_pred_pre_5b = model_5b.predict(post_red_nd)
post_pred_pre_6b = model_6b.predict(post_red_nd)

post_pred_1b = pd.DataFrame(post_pred_pre_1b)
post_pred_2b = pd.DataFrame(post_pred_pre_2b)
post_pred_3b = pd.DataFrame(post_pred_pre_3b)
post_pred_4b = pd.DataFrame(post_pred_pre_4b)
post_pred_5b = pd.DataFrame(post_pred_pre_5b)
post_pred_6b = pd.DataFrame(post_pred_pre_6b)

post_pred_1b.to_csv('08.Version1/Predictions/pred_post_1b.csv',index=False)
post_pred_2b.to_csv('08.Version1/Predictions/pred_post_2b.csv',index=False)
post_pred_3b.to_csv('08.Version1/Predictions/pred_post_3b.csv',index=False)
post_pred_4b.to_csv('08.Version1/Predictions/pred_post_4b.csv',index=False)
post_pred_5b.to_csv('08.Version1/Predictions/pred_post_5b.csv',index=False)
post_pred_6b.to_csv('08.Version1/Predictions/pred_post_6b.csv',index=False)