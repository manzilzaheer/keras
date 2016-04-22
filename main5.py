#import seq2seq
import numpy as np
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.noise import GaussianNoise
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
train_y = np.array(labels, dtype='int8', ndmin=2).transpose()
progs = np.array(progs)

# Test1
progs1, labels1, num_progs = load_char_data1('test1')
my_maxlen1 = len(max(progs1,key=len))
idx_progs = [[token_to_index[token] for token in prog] for prog in progs1]
test1_X = pad_sequences(idx_progs, maxlen=my_maxlen1)
test1_y = np.array(labels1, dtype='int8', ndmin=2).transpose()

# Test2
progs1, labels1, num_progs = load_char_data1('test2')
my_maxlen1 = len(max(progs1,key=len))
idx_progs = [[token_to_index[token] for token in prog] for prog in progs1]
test2_X = pad_sequences(idx_progs, maxlen=my_maxlen1)
test2_y = np.array(labels1, dtype='int8', ndmin=2).transpose()

# Test3
progs1, labels1, num_progs = load_char_data1('test3')
my_maxlen1 = len(max(progs1,key=len))
idx_progs = [[token_to_index[token] for token in prog] for prog in progs1]
test3_X = pad_sequences(idx_progs, maxlen=my_maxlen1)
test3_y = np.array(labels1, dtype='int8', ndmin=2).transpose()

# Test4
progs1, labels1, num_progs = load_char_data1('test4')
my_maxlen1 = len(max(progs1,key=len))
idx_progs = [[token_to_index[token] for token in prog] for prog in progs1]
test4_X = pad_sequences(idx_progs, maxlen=my_maxlen1)
test4_y = np.array(labels1, dtype='int8', ndmin=2).transpose()

# Test5
progs1, labels1, num_progs = load_char_data1('test5')
my_maxlen1 = len(max(progs1,key=len))
idx_progs = [[token_to_index[token] for token in prog] for prog in progs1]
test5_X = pad_sequences(idx_progs, maxlen=my_maxlen1)
test5_y = np.array(labels1, dtype='int8', ndmin=2).transpose()

# Test6
progs1, labels1, num_progs = load_char_data1('test6')
my_maxlen1 = len(max(progs1,key=len))
idx_progs = [[token_to_index[token] for token in prog] for prog in progs1]
test6_X = pad_sequences(idx_progs, maxlen=my_maxlen1)
test6_y = np.array(labels1, dtype='int8', ndmin=2).transpose()

print "Example:"
x_example, y_example = train_X[29], train_y[29]
print "x:"
print progs[29]
print "y:"
print np.squeeze(train_y[29])
print train_y.shape


RMS = RMSprop(lr=0.001)

# Neural network description
model = Graph()
model.add_input(name='input', input_shape=(my_maxlen,), dtype='int')
model.add_node(OneHotEmbedding(output_dim=vocab_size, input_dim=vocab_size, mask_zero=True), name='one_hot', input='input')
#model.add_node(TimeDistributedDense(output_dim=200,activation='tanh'), name='inter0', input='one_hot')
#model.add_node(TimeDistributedDense(output_dim=100), name='hashed', input='inter0')
model.add_node(LSTM(output_dim=1, return_sequences=True, activation='tanh'), name='myLSTM', input='one_hot')
model.add_node(GaussianNoise(0.1),name='myLSTM_noise', input='myLSTM')
model.add_node(dumbFilter2(hidden_dim=1,filter_size=vocab_size), name='filter', inputs=['myLSTM_noise','one_hot'])
model.add_node(SimpleRNN(output_dim=1, activation='tanh'), name='ander', input='filter')
model.add_node(Dense(output_dim=1,activation='sigmoid'),name='ander1', input='ander')
model.add_node(GaussianNoise(0.01),name='ander2', input='ander1')
model.add_output(name='output', input='ander2')
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


for i in xrange(20):
    model.fit({'input':train_X, 'output':train_y}, nb_epoch=1, batch_size=100, show_accuracy=True, validation_split=0.1)

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

    pred_y = np.squeeze( model.predict({'input':test1_X})['output'] )
    pred_y = pred_y > 0.5
    print "Test 1: ",
    print np.sum(pred_y != test1_y[:,0])

    pred_y = np.squeeze( model.predict({'input':test2_X})['output'] )
    pred_y = pred_y > 0.5
    print "Test 2: ",
    print np.sum(pred_y != test2_y[:,0])

    pred_y = np.squeeze( model.predict({'input':test3_X})['output'] )
    pred_y = pred_y > 0.5
    print "Test 3: ",
    print np.sum(pred_y != test3_y[:,0])

    pred_y = np.squeeze( model.predict({'input':test4_X})['output'] )
    pred_y = pred_y > 0.5
    print "Test 4: ",
    print np.sum(pred_y != test4_y[:,0])

    pred_y = np.squeeze( model.predict({'input':test5_X})['output'] )
    pred_y = pred_y > 0.5
    print "Test 5: ",
    print np.sum(pred_y != test5_y[:,0])

    pred_y = np.squeeze( model.predict({'input':test6_X})['output'] )
    pred_y = pred_y > 0.5
    print "Test 6: ",
    print np.sum(pred_y != test6_y[:,0])

