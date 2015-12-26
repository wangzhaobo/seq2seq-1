import sys
import operator
import matplotlib.pyplot as plt
import util
import numpy as np

class Dataset:
    def __init__(self, filestr):
        self.filestr = filestr
        self.data = util.loadFile("data/" + filestr)

    def load(self, name):
        start_time = util.now()
        print "[Loading %s:%s...]" % (self.filestr, name),
        sys.stdout.flush()

        rtn = self.data[name]
        print "[Took %d milliseconds]" % (util.now() - start_time)
        return rtn

def shuffle(a, b):
    start_time = util.now()
    print "[Shuffling...]", 
    sys.stdout.flush()

    assert len(a) == len(b), "Length of arrays is not equal." 
    combined = np.asarray([[x,y] for x,y in zip(a, b)])
    np.random.shuffle(combined)

    print "[Took %d milliseconds]" % (util.now() - start_time)
    return combined[:, 0], combined[:, 1]

def divide(a, b, ratio):
    start_time = util.now()
    print "[Dividing len %d data by %f ratio...]" % (len(a), ratio)
    sys.stdout.flush()

    assert len(a) == len(b), "Length of arrays is not equal." 
    index = round(len(a) * ratio)
    a_1, a_2 = a[:index], a[index:]
    b_1, b_2 = b[:index], b[index:]

    print "[Division of %d:%d]" % (len(a_1), len(a_2)),
    print "[Took %d milliseconds]" % (util.now() - start_time)
    return a_1, a_2, b_1, b_2

ds = Dataset("kjv.dat")
pairs = ds.load("pairs") 
data_x, data_y = pairs[:, 0], pairs[:, 1]
data_x, data_y = shuffle(data_x, data_y)

index2word = ds.load("index2word")
word2index = {}
for i,e in enumerate(index2word):
    word2index[e] = i

train_x, test_x, train_y, test_y = divide(data_x, data_y, 0.995)




