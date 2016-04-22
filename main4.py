#import seq2seq
import numpy as np
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.embeddings import Embedding, OneHotEmbedding
from keras.layers.ds_enhanced import DeepSet, LSTM2, dumbFilter, dumbFilter2
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences

from keras.backend.common import _FLOATX, _EPSILON

from utils import *

print "Hi"

# Load data
progs, labels, num_progs = load_char_data1('task1')

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
# train_X = np.zeros((num_progs, my_maxlen, vocab_size), dtype='int8')
# for i, prog in enumerate(idx_progs):
#     for j, token in enumerate(prog):
#         train_X[i,j,token] = 1
train_y = np.array(labels, dtype='int8', ndmin=2).transpose()
#train_y = pad_sequences(labels, maxlen=my_maxlen)

#train_X = np.expand_dims(train_X,axis=2)
#train_y = np.expand_dims(train_y,axis=2)

progs = np.array(progs)

print "Example:"
x_example, y_example = train_X[29], train_y[29]
print "x:"
print progs[29]
#print " ".join([vocab[x] for x in x_example])
# for token in x_example:
#     if token:
#         print vocab[token],
print "y:"
print np.squeeze(train_y[29])
print train_y.shape


RMS = RMSprop(lr=0.005)

# Neural network description
model = Graph()
model.add_input(name='input', input_shape=(my_maxlen,), dtype='int')
model.add_node(OneHotEmbedding(output_dim=vocab_size, input_dim=vocab_size, mask_zero=True), name='one_hot', input='input')
#model.add_node(TimeDistributedDense(output_dim=50,activation='tanh'), name='inter0', input='one_hot')
#model.add_node(TimeDistributedDense(output_dim=100), name='hashed', input='inter0')
model.add_node(LSTM(output_dim=1, return_sequences=True, activation='tanh'), name='myLSTM', input='one_hot')
model.add_node(dumbFilter2(hidden_dim=1,filter_size=vocab_size), name='filter', inputs=['myLSTM','one_hot'])
model.add_node(SimpleRNN(output_dim=1, activation='tanh'), name='ander', input='filter')
model.add_node(Dense(output_dim=1,activation='sigmoid'),name='ander1', input='ander')
model.add_output(name='output', input='ander1')
model.compile(optimizer=RMS, loss={'output':'binary_crossentropy'})

# model.add(Embedding(output_dim=64, input_dim=vocab_size, mask_zero=True))
# model.add(DeepSet(hidden_dim=10,filter_size=10))
# #model.add(LSTM2(output_dim=26))
# model.add(SimpleRNN(output_dim=1, activation='sigmoid'))
# model.compile(optimizer=RMS, loss='binary_crossentropy')

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

# pred_y = np.clip(pred_y, _EPSILON, 1-_EPSILON)
# my_ce = np.squeeze( train_y[0:2] )*np.log(pred_y) + (1-np.squeeze( train_y[0:2] ))*np.log(1-pred_y)
# my_one = np.ones_like(my_ce)
# #aa = sum(my_ce[0,-7:]) + sum(my_ce[1,-8:])
# aa = sum(my_ce[0,10:]) + sum(my_ce[1,12:])
# bb = sum(my_one[0,10:]) + sum(my_one[1,12:])
# print -aa/bb

for i in xrange(20):
    model.fit({'input':train_X, 'output':train_y}, nb_epoch=1, batch_size=10, show_accuracy=True, validation_split=0.1)

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

    pred_y = np.squeeze( model.predict({'input':train_X})['output'] )
    pred_y = pred_y > 0.5
    print np.sum(pred_y != train_y[:,0])
    print progs[pred_y != train_y[:,0]]


    # pred_y = np.clip(pred_y, _EPSILON, 1-_EPSILON)
    # my_ce = np.squeeze( train_y[0:2] )*np.log(pred_y) + (1-np.squeeze( train_y[0:2] ))*np.log(1-pred_y)
    # my_one = np.ones_like(my_ce)
    # #aa = sum(my_ce[0,-7:]) + sum(my_ce[1,-8:])
    # aa = sum(my_ce[0,10:]) + sum(my_ce[1,12:])
    # bb = sum(my_one[0,10:]) + sum(my_one[1,12:])
    # print -aa/bb
