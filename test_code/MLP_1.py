import keras
from time import time
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.callbacks import TensorBoard
import torch
import pandas as pd
import os
import sys
from keras.utils import plot_model

KOS = pd.read_table('./KO.tsv')
TFS = pd.read_table('./TFs.tsv')
#test inputs
#tKO = pd.read_table('data/app18_net9s/test_KO.tsv')
#tTFS = pd.read_table('data/app18_net9s/test_TFs.tsv')
#outputs
NONTFS = pd.read_table('./NonTFs.tsv')


inp = np.array([KOS["G1"].tolist(),KOS["G2"].tolist(),KOS["G3"].tolist(),KOS["G4"].tolist(),KOS["G5"].tolist(),KOS["G6"].tolist(),KOS["G9"].tolist()], "float32")
out = np.array([NONTFS["G1"].tolist(),NONTFS["G2"].tolist(),NONTFS["G3"].tolist(),NONTFS["G4"].tolist(),NONTFS["G5"].tolist(),NONTFS["G6"].tolist(),NONTFS["G9"].tolist()], "float32")
inp = inp.reshape(3528,7)
out = out.reshape(3528,7)
model = Sequential()

#model.add(LSTM(10, input_shape=(3528, 7)))
#model.add(LSTM(7,activation="softmax"))
#model.add(Dense(7, activation='relu'))
#model.add(Dense(3, activation='sigmoid'))
#model.add(Dense(2, activation='sigmoid'))
model.add(Dense(7, input_dim=7,activation='sigmoid'))
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mae', 'acc'])
tensorboard = keras.callbacks.TensorBoard(log_dir='logs/')


model.fit(inp, out, epochs=250, verbose=2,callbacks=[tensorboard])