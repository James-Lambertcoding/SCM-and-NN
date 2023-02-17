
# load and evaluate a saved model
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
 
# load model
model = load_model('05.Models/reg_model_deep.h5')

model.summary()

print('done')

