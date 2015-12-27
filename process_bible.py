import random
import nltk
import operator
import itertools
import numpy

START_TOKEN = "START_TOKEN"
END_TOKEN = "END_TOKEN"
UNK = "UNK"
VOCAB_SIZE = 3000

def clean():
    with open("bible.txt") as f:
        raw = " ".join(f.readlines())
        sentences = nltk.sent_tokenize(raw)
        clauses = []
        d = ","
        for sent in sentences:
            c = [e + d for e in sent.split(d) if e != ""]
            clauses.extend(c)
        tokenized = [[START_TOKEN] + nltk.word_tokenize(clause) + [END_TOKEN] for clause in clauses]
        word_freq = nltk.FreqDist(itertools.chain(*tokenized))
        print "Found %d unique words" % len(word_freq.items())
        vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse = True)[:VOCAB_SIZE - 2]
        print "With vocab size %d, the least frequent word is %s with %d occs." % (VOCAB_SIZE, vocab[-1][0], vocab[-1][1])

        sortedVocab = sorted(vocab, key = operator.itemgetter(1))
        index2word = [UNK] + [x[0] for x in sortedVocab]


        print "Replacing rare words"
        common = []
        for sent in tokenized:
            common.append([w if w in index2word else UNK for w in sent])

        numpy.savez("cleaned_bible", common = common, index2word = index2word)

print "numifying"

f = numpy.load("cleaned_bible.npz")
index2word = f["index2word"]
common = f["common"]

word2index = {}
for i,e in enumerate(index2word):
    word2index[e] = i

pairs = []
print word2index["START_TOKEN"]

def toNum(sent):
    return [word2index[x] for x in sent]

for i, sent in enumerate(common[:-1]):
    if i < 100:
        print sent, common[i+1]
    pairs.append([toNum(sent), toNum(common[i+1])])


numpy.savez("num_bible", pairs = pairs, index2word = index2word, word2index = word2index)
