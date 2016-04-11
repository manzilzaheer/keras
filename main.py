#import seq2seq
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.ds_enhanced import DeepSet, LSTM2
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

print "Example:"
x_example, y_example = train_X[29], train_y[29]
print "x:"
print progs[29]
#print " ".join([vocab[x] for x in x_example])
# for token in x_example:
#     if token:
#         print vocab[token],
print "y: %d" % y_example


# Neural network description
model = Sequential()
model.add(Embedding(output_dim=64, input_dim=vocab_size, mask_zero=True))
model.add(DeepSet(hidden_dim=10,filter_size=10))
#model.add(LSTM2(output_dim=26))
model.add(SimpleRNN(output_dim=1, activation='sigmoid'))
RMS = RMSprop(lr=0.005)
model.compile(optimizer=RMS, loss='binary_crossentropy')


model.fit(train_X, train_y, nb_epoch=20, batch_size=100, show_accuracy=True)

print len(model.layers)

print "Hi2"
