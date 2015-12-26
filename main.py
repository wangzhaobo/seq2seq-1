import numpy as np
import sys
import os
import time
import random
import data
from copy import deepcopy
from datetime import datetime
import model
from constants import index2alpha, alpha2index
import util

from data import train_x, train_y, test_x, test_y, index2word, word2index

HIDDEN_SIZE = 200

print len(train_x), len(test_x), len(train_y), len(test_y)

def train_with_sgd(m, learning_rate=0.005, evaluate_loss_after=1):
    # We keep track of the losses so we can plot them later
    losses = [( 0, 1000.0 )]
    num_examples_seen = 0

    print(chr(27) + "[2J")
    print "Beginning training..."
    while losses[-1][1] > 0.1:
        # Optionally evaluate the loss
        if (num_examples_seen % evaluate_loss_after == 0):
            loss = m.calculate_loss(test_x, test_y)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%m-%d-%H-%M-%S')
            print ""
            print "[%s: Loss after %d examples = %f]" % (time, num_examples_seen, loss)
            save_path = sys.argv[-1] + "-" + time + "-" + str(loss)[:5]
            model.save_model(m, save_path)

            for r in xrange(5):
                i = random.randint(0, len(test_x) - 1)

                model_out = m.predict_class(test_x[i])
                out = [index2word[x] for x in model_out]
                inp = [index2word[x] for x in test_x[i]]
                trans = "%s =>\n   %s" % (" ".join(inp), " ".join(out))

                print " * " + trans

            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print "Setting learning rate to %f" % learning_rate
                sys.stdout.flush()

        # For each training example...
        for x, y in zip(train_x, train_y):
            m.sgd_step(x, y, learning_rate)
            num_examples_seen += 1

m = model.Model(len(index2word), HIDDEN_SIZE, len(index2word))
train_with_sgd(m)

