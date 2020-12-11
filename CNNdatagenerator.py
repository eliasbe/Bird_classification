"""
@author:
Elías 
adjusted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""

import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, columns=20, dim=(128,20),
                 n_classes=14, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size

        self.columns=columns        # length of each sample
        self.labels = labels        # A dictionary with ID keys and numerical catagorie values
        self.list_IDs = list_IDs    # The ID's to batch
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing (batch_size) number of samples'
        

        X = np.empty((self.batch_size,128,self.columns,1)) 
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
           
            # Loads a spectrogram from file
            M = np.load('/content/drive/My Drive/Colab Notebooks/Fuglar/Processed_Spectrograms/' + str(self.labels[ID]) + '/' + ID + '.npy')
            # Pick a random slice from spectrogram
            randomRange = M.shape[1]-self.columns+1 
            assert(randomRange > 0),(str(ID) +  "too short") # X_train has already filtered for short recordings
            idx = np.random.randint(randomRange)
            samplSlice = np.array(M[:,idx:idx+self.columns], ndmin=3)
            
            # Idea: adjust gain a little bit here for data augmentation
            # Add slice to batch and normalise to take value between -1 and 1
            X[i,] = samplSlice.reshape(128,self.columns,1)/40 + 1 

            # Store specie number
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
