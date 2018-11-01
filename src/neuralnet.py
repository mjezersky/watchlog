#!/usr/bin/python

## Neural network implementation - convolutional and LSTM variants
## rough draft / proof of concept
## 
## Author: Matous Jezersky

import numpy, pickle

from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
from keras.layers import Dropout
from keras.layers import LSTM, Conv1D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, to_categorical

INPUT_SIZE = 600
OUTPUT_SIZE = 1

def getDataset(filename):
    f = open(filename, "rb")
    out = pickle.load(f)
    f.close()
    return out

class TrainingData():
    def __init__(self, data):
        self.inputs = []
        self.outputs = []
        for tup in data:
            self.inputs.append(tup[0])
            self.outputs.append(tup[1])
        #self.inputs = numpy.array(self.inputs)
        #self.outputs = numpy.array(self.outputs)

class NeuralNet():
    def __init__(self):
        self.model = None
        self.X = None
        self.y = None
        self.callbacks = []
        self.checkpoint_path = "w-current.hdf5"
        self.reshape = True

    def set_data(self, ins, outs):
        self.X = numpy.array(ins)
        if self.reshape:
            print("expected: ", len(self.X), INPUT_SIZE, 1)
            self.X = self.reshape_data(self.X)
            #self.X.reshape(len(self.X), INPUT_SIZE, 1)
            #self.X = numpy.expand_dims(self.X, axis=2)
            #self.X = numpy.reshape(1,-1)
        self.y = numpy.array(outs)
        self.post_data_load()

    def post_data_load(self):
        print("NeuralNet - post_data_load NOP implementation")

    def build_model(self):
        print("NeuralNet - build_model NOP implementation")

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def compile_model(self):
        print("NeuralNet - compile_model NOP implementation")
    
    def enable_checkpoint(self, filename):
        checkpoint = ModelCheckpoint(filename, monitor='loss', verbose=1, save_best_only=True, mode='min')
        self.callbacks = [checkpoint]

    def fit(self, train_epochs=100, train_batch_size=32):
        self.model.fit(self.X, self.y, epochs=train_epochs, batch_size=train_batch_size, callbacks=self.callbacks)

    def reshape_data(self, data):
        print("NeuralNet - reshape NOP implementation")
        return data
        

    def feed(self, data):
        data = numpy.array(data)
        if self.reshape:
            data = self.reshape_data(data)
        return self.model.predict(data)

class ConvNet(NeuralNet):

    def reshape_data(self, data):
        return data.reshape(-1,600,1)
    
    def build_model(self):
        print("shape: " + str(self.X.shape))
        self.model = Sequential()
        self.model.add(Conv1D(300, 5, input_shape=(self.X.shape[1], self.X.shape[2])))
        self.model.add(Conv1D(200, 5))
        self.model.add(Flatten())
        self.model.add(Dense(200, activation='relu', input_dim=INPUT_SIZE))
        #self.model.add(Dense(100, activation='relu', input_dim=INPUT_SIZE))
        
        self.model.add(Dense(64, activation='relu', input_dim=INPUT_SIZE))
        self.model.add(Dense(OUTPUT_SIZE, activation='sigmoid'))

    def compile_model(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
  

class LstmNet(NeuralNet):

    def reshape_data(self, data):
        return data.reshape(-1,600)

    def post_data_load(self):
        self.y = to_categorical(self.y)
    
    def build_model(self):
        print("X shape: " + str(self.X.shape))
        print("y shape: " + str(self.y.shape))
        self.model = Sequential()
        self.model.add(Embedding(1000, 256, input_length=self.X.shape[1]))
        self.model.add(LSTM(200, recurrent_dropout=0.2, dropout=0.2))
        self.model.add(Dense(2,activation='softmax'))
        
    def compile_model(self):
        self.model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])


def train(weights_filename=None, NNImplementationClass=ConvNet):
    
    data = TrainingData(getDataset("dataset/dataset"))
    #print(data.inputs.shape)
    
    nn = NNImplementationClass()
    nn.set_data(data.inputs, data.outputs)
    nn.build_model()
    if not (weights_filename is None):
        nn.load_weights(weights_filename)
    nn.compile_model()
    nn.enable_checkpoint("w-current.hdf5")

    print("fitting...")
    nn.fit()

def predict_init(weights_filename, NNImplementationClass=ConvNet):
    
    data = TrainingData(getDataset("dataset/dataset"))
    #print(data.inputs.shape)
    
    nn = NNImplementationClass()
    nn.set_data(data.inputs, data.outputs)
    nn.build_model()
    if not (weights_filename is None):
        nn.load_weights(weights_filename)
    nn.compile_model()
    return nn

def build_overlapping_sequences(data, seq_len, step):
    sequences = []
    data_len = len(data)
    for i in range(0, data_len, step):
        sequences.append(data[i:i+seq_len])
    return sequences



