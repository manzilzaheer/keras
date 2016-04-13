#import seq2seq
import numpy as np
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.embeddings import Embedding, OneHotEmbedding
from keras.layers.ds_enhanced import DeepSet, LSTM2, dumbFilter
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences

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
# train_X = np.zeros((num_progs, my_maxlen, vocab_size), dtype='int8')
# for i, prog in enumerate(idx_progs):
#     for j, token in enumerate(prog):
#         train_X[i,j,token] = 1
train_y = np.array(labels, dtype=bool, ndmin=2).transpose()
train_y = pad_sequences(labels, maxlen=my_maxlen, dtype=float, value=0.5)
train_y = np.expand_dims(train_y,axis=2)

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
model.add_input(name='input', input_shape=(my_maxlen,), dtype='int32')
model.add_node(OneHotEmbedding(output_dim=vocab_size, input_dim=vocab_size, mask_zero=True), name='one_hot', input='input')
model.add_node(LSTM(output_dim=100, return_sequences=True), name='LSTM', input='one_hot')
model.add_node(dumbFilter(hidden_dim=100,filter_size=vocab_size), name='filter', inputs=['one_hot','LSTM'])
model.add_output(name='output', input='filter')
model.compile(optimizer=RMS, loss={'output':'binary_crossentropy'})

# model.add(Embedding(output_dim=64, input_dim=vocab_size, mask_zero=True))
# model.add(DeepSet(hidden_dim=10,filter_size=10))
# #model.add(LSTM2(output_dim=26))
# model.add(SimpleRNN(output_dim=1, activation='sigmoid'))
# model.compile(optimizer=RMS, loss='binary_crossentropy')

print model.predict({'input':train_X[0:10]})

#model.fit(train_X, train_y, nb_epoch=20, batch_size=100, show_accuracy=True)

print "Hi2"
