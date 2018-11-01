#!/usr/bin/python

import time
import numpy as np
import pickle

import neuralnet

def norm(value):
    return int(value)/255.0

def rep_to_string(rep):
    rep = list( map(convert_rep_elem , rep) )
    return " ".join(rep)

def get_rep(net, data):
    try:
        rep = net.feed(np.array([list(map(norm,data))]))[0]
        print( rep_to_string(rep))
        return rep
    except Exception as ex:
        print ("Fail")
        print (ex)
        return None


f = open("input.txt", "rb")
input_data = f.read()
f.close()

sequences = neuralnet.build_overlapping_sequences(input_data, 600, 100)

nn = neuralnet.predict_init("w-current.hdf5")

i = 0
for seq in sequences:
    seq_data = list(map(norm, seq))
    res = nn.feed(seq_data)[0][0]
    if res>0.95:
        print("i:"+str(i), res)
        #print(seq)
    i+=1

