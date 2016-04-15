#import seq2seq
import numpy as np
from keras.models import Graph
from keras.layers.embeddings import OneHotEmbedding
from keras.layers.ds_enhanced import exeFilter
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences

from keras.backend.common import _EPSILON
from utils import *

print "Hi"

# Load data
progs, labels, num_progs = load_char_data('task')

# Create list of distinct tokens
vocab = sorted(set([token for prog in progs for token in prog]))
vocab_size = len(vocab)+1
print "Found %d unique words tokens." % vocab_size

# Build token_to_index vectors
vocab.insert(0,'NULL')
token_to_index = dict([(w,i) for i,w in enumerate(vocab)])

# Create the training data
my_maxlen = len(max(progs,key=len))
idx_progs = [[token_to_index[token] for token in prog] for prog in progs]
train_X = pad_sequences(idx_progs, maxlen=my_maxlen)
train_y = pad_sequences(labels, maxlen=my_maxlen)
train_y = np.expand_dims(train_y,axis=2)

print "Example:"
x_example, y_example = train_X[29], train_y[29]
print "x:"
print progs[29]
print "y:"
print np.squeeze(train_y[29])
print train_y.shape


RMS = RMSprop(lr=0.01)

# Neural network description
model = Graph()
model.add_input(name='input', input_shape=(my_maxlen,), dtype='int')
model.add_node(OneHotEmbedding(output_dim=vocab_size, input_dim=vocab_size, mask_zero=True), name='one_hot', input='input')
model.add_node(exeFilter(filter_size=100), name='filter', input='one_hot')
model.add_output(name='output', input='filter')
model.compile(optimizer=RMS, loss={'output':'binary_crossentropy'})

print 'Training X:'
print progs[0:2]
print train_X[0:2]
print 'Training Y:'
print np.squeeze( train_y[0:2] )
print 'Predicted Y:'
pred_y = np.squeeze( model.predict({'input':train_X[0:2]})['output'] )
print pred_y
print 'Loss: ',
print model.evaluate({'input':train_X[0:2], 'output':train_y[0:2]})

pred_y = np.clip(pred_y, _EPSILON, 1-_EPSILON)
my_ce = np.squeeze( train_y[0:2] )*np.log(pred_y) + (1-np.squeeze( train_y[0:2] ))*np.log(1-pred_y)
my_one = np.ones_like(my_ce)
aa = sum(my_ce[0,10:]) + sum(my_ce[1,12:])
bb = sum(my_one[0,10:]) + sum(my_one[1,12:])
print -aa/bb

import sys
from cStringIO import StringIO


for i in xrange(20):
    #old_stdout = sys.stdout
    #redirected_output = sys.stdout = StringIO()
    exec("model.fit({'input':train_X, 'output':train_y}, nb_epoch=1, batch_size=100, show_accuracy=True, validation_split=0.1)")
    #sys.stdout = old_stdout

    print "Hi",
    print i

    print 'Training X:'
    print progs[0:2]
    print train_X[0:2]
    print 'Training Y:'
    print np.squeeze( train_y[0:2] )
    print 'Predicted Y:'
    pred_y = np.squeeze( model.predict({'input':train_X[0:2]})['output'] )
    print pred_y
    print 'Loss: ',
    print model.evaluate({'input':train_X[0:2], 'output':train_y[0:2]})
    pred_y = np.clip(pred_y, _EPSILON, 1-_EPSILON)
    my_ce = np.squeeze( train_y[0:2] )*np.log(pred_y) + (1-np.squeeze( train_y[0:2] ))*np.log(1-pred_y)
    my_one = np.ones_like(my_ce)
    aa = sum(my_ce[0,10:]) + sum(my_ce[1,12:])
    bb = sum(my_one[0,10:]) + sum(my_one[1,12:])
    print -aa/bb