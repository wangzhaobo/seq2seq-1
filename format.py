import re
import numpy as np
import sys
import nltk
import itertools
import operator
from constants import *

VOCAB_SIZE = 20000
START_TOKEN = "START_TOKEN"
END_TOKEN = "END_TOKEN"
UNK = "UNK"

numSkipped = 0

with open("data/words_definitions.csv") as f:
    content = [row.strip(" \t\n\r") for row in f.readlines()]
    joinedRows = [content[0]]

    #joining lines and skipping short ones
    for row in content[1:]:
        if joinedRows[-1][-1] == "\\":
            joinedRows[-1] += row
        else:
            joinedRows.append(row)

    formattedRows = []

    for row in joinedRows:
        x = re.sub(r'[^a-z0-9 \.,;\']', ' ', row.lower())
        word, definition = x[:x.index(',')], x[x.index(',') + 1:]
        word, definition = word.strip(), definition.strip()
        word = re.sub(r' ', '', word)

        if len(word) <= 3: #skipping nonmorphological words
            numSkipped += 1
            #print "skipping %s" % word
            continue

        for r in range(6):
            definition = re.sub(r'  ', ' ', definition)
            definition = re.sub(r' ,', ',', definition)
            definition = re.sub(r' ;', ';', definition)
            definition = re.sub(r' \.', '.', definition)
            definition = re.sub(r',,', ',', definition)
            definition = re.sub(r'\.\.', '.', definition)
        formattedRows.append([definition, word])

    print "Skipped %d words" % numSkipped
    print "Created %d word/definition pairs" % len(formattedRows)
    print "Parsing with NLTK..."

    parsedRows = []

    for row in formattedRows:
        row[0] = "%s %s %s" % (START_TOKEN, row[0], END_TOKEN)
        row[0] = nltk.word_tokenize(row[0])
        row[1] = [START_TOKEN] + list(row[1]) + [END_TOKEN]
        parsedRows.append(row)

    print "Converting to numpy..."
    parsedRows = np.asarray(parsedRows)
    wordFreq = nltk.FreqDist(itertools.chain(*parsedRows[:, 0]))
    vocab = sorted(wordFreq.items(), key=lambda x: (x[1], x[0]), reverse = True)[:VOCAB_SIZE - 2]
    print "Found %d unique words" % len(wordFreq.items())
    print "With vocab size %d, the least frequent word is %s" % (VOCAB_SIZE, vocab[-1][0])

    print "Creating indices..."
    sortedVocab = sorted(vocab, key = operator.itemgetter(1))
    index2word = [UNK] + [x[0] for x in sortedVocab]

    print "Replacing rare words..."
    for i, e in enumerate(parsedRows):
        parsedRows[i][0] = [w if w in index2word else UNK for w in e[0]]

    definitions, words = parsedRows[:, 0], parsedRows[:, 1]

    word2index = {}
    for i,e in enumerate(index2word):
        word2index[e] = i

    print "Converting to num when possible..."
    skipped = 0
    definitions_num = []
    words_num = []
    for i in range(len(definitions)):
        try:
            definitions_num.append([word2index[w] for w in definitions[i]])
            words_num.append([alpha2index[c] for c in words[i]])
        except KeyError:
            skipped += 1
            continue

    print "Skipped %d" % skipped

    print "Saving to %s.npz..." % sys.argv[-1]
    np.savez("data/" + sys.argv[-1], 
             definitions = definitions_num,
             words = words_num,
             index2word = index2word)
