#!/usr/bin/python

## Dataset creation tool
## rough draft / proof of concept
## 
## Author: Matous Jezersky

import pickle, numpy

def build_overlapping_sequences(data, seq_len, step):
    sequences = []
    data_len = len(data)
    for i in range(0, data_len, step):
        new_sequence = data[i:i+seq_len]
        if len(new_sequence)<seq_len:
            new_sequence += bytes(" "*(seq_len-len(new_sequence)), "utf-8")
        sequences.append(new_sequence)
    return sequences


class Dataset():
    def __init__(self):
        self.dataset = []
        

    def convert_and_normalize(self, char):
        return int(char)/255.0

    def add(self, filename, label):
        f = open(filename, "rb")
        data = f.read()
        f.close()
        samples = build_overlapping_sequences(data, 600, 100)
        for sample in samples:
            sample = list(map(self.convert_and_normalize, sample))
            print(len(sample))
            sample_array = numpy.array(sample)
            print(sample_array.shape)
            #break
            self.dataset.append([sample_array , [label]])

    def write(self):
        f = open("dataset", "wb")
        pickle.dump(self.dataset, f)
        f.close()
        self.dataset = []


def testload():
    f = open("dataset", "rb")
    ds = pickle.load(f)
    f.close()
    x = []
    for sample in ds:
        x.append(sample[0])
    print("DS info:")
    print("len", len(x))
    print("elem len", len(x[0]))
    x = numpy.array(x).reshape(-1, 600, 1)
    print("shape", x.shape)

d = Dataset()
d.add("bad/routerbad.log", 1)
#d.add("bad/serverbad.log", 1)
d.add("routermin.log", 0)
#d.add("serverok.log", 0)
#d.add("routerok.log", 0)
d.write()

testload()

    
