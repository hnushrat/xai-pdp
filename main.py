import os

import pandas as pd
import tensorflow as tf

# IF GPU IS AVAILABLE Uncomment the code below
'''
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
'''

from PDP import PDP

os.makedirs('plots', exist_ok = True)

model_paths = []
data_paths = []

N_VALUES = 30 # (int)
FEATURE_INDEX = 2 # (int)
TARGET = '' # the target feature name (str)

model_x1 = tf.keras.models.load_model(model_paths[0])
model_x3 = tf.keras.models.load_model(model_paths[1])


input_x1 = pd.read_excel(data_paths[0])
input_x3 = pd.read_excel(data_paths[1])

# Modify the tabular dataframe if needed

'''
input_x1 = input_x1.iloc[:, 1:]
input_x3 = input_x3.iloc[:, 1:]
'''

PDP(n_values = N_VALUES, dataset = input_x3, feature_index = FEATURE_INDEX, model = model_x3, day = 1, target = TARGET).get_pdp()
