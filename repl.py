import util
from constants import *
import sys
from model import Model
import data
import glob
import os
import nltk

filestr = sys.argv[-1]

if filestr == "load":
    filestr = max(glob.iglob("models/*.npz"), key = os.path.getmtime)

m = Model.load(filestr)

def interactiveInput(model):
    inp = ""
    while inp != "q":
        inp = raw_input(">")
        if inp[0] == "#":
            vec = [data.word2index["START_TOKEN"]]+[data.word2index[x] for x in nltk.word_tokenize(inp)] + [data.word2index["END_TOKEN"]]
            out = model.vector_rep(vec)
            print out
        else:
            vec = [data.word2index["START_TOKEN"]]+[data.word2index[x] for x in nltk.word_tokenize(inp)] + [data.word2index["END_TOKEN"]]
            out = model.predict_class(vec)
            print " ".join(data.index2word[x] for x in out)

interactiveInput(m)
