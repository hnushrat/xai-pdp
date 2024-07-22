import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


class PDP:
    def __init__(self, n_values = None, dataset = None, feature_index = None, start = -1, stop = 1, model = None, day = None, target = None):
        '''
        dataset :  the dataset
        n_values : number of unique values to try
        feature_index : get the PDP for this feature
        start_val : the first value of the range must be lower than stop
        stop_val : the last value of the range
        model : the trained model
        n_datasets : the dataset created with n different values for the feature index
        day : the PDP at which day -> 0 , 1, 2, ..
        target : target feature name
        '''
        self.dataset = dataset.copy()
        self.n_values = n_values
        self.feature_index = feature_index
        self.start_val = start
        self.stop_val = stop
        self.model = model
        self.n_datasets = []
        self.day = day
        self.columns = self.dataset.columns
        self.target = target
        

    def get_datasets(self):
        if self.feature_index >= len(self.columns):
            return 'The feature index is out of bounds'
        for value in np.linspace(start = self.start_val, stop = self.stop_val, num = self.n_values):
            dataset_copy = self.dataset.copy()
            dataset_copy.iloc[:, self.feature_index] = value
            self.n_datasets.append(dataset_copy)
        self.n_datasets = np.asarray(self.n_datasets)
        
        return self.n_datasets
    
    def get_pdp(self):

        self.n_datasets = self.get_datasets()
        if isinstance(self.n_datasets, str):
            print(self.n_datasets)
            return self.n_datasets
        pred = self.model.predict(self.n_datasets)[:, self.day - 1]
        plt.figure(figsize = (20,5))
        plt.rcParams['font.size'] = 12

        plt.title(f'Variation in the prediction of Day {self.day} with changing values of the feature {self.columns[self.feature_index]}')
        plt.plot(pred, c = 'r')
        plt.xticks(range(self.n_values), np.round(np.linspace(start = self.start_val, stop = self.stop_val, num = self.n_values), 3), rotation = 30)
        plt.xlabel(f'Range of Values ({self.n_values}) for the feature {self.columns[self.feature_index]}')
        plt.ylabel('Models Estimate')
        
        plt.savefig(f'plots/{self.target}_Feature_{self.columns[self.feature_index]}.png')
        # plt.show()