import theano
from theano import tensor as T
from keras.models import Graph
from keras.layers.embeddings import OneHotEmbedding
from keras.preprocessing.sequence import pad_sequences
import numpy as np

from keras.backend.common import _FLOATX, _EPSILON

def variable(value, dtype=_FLOATX, name=None):
    '''Instantiate a tensor variable.
    '''
    value = np.asarray(value, dtype=dtype)
    return theano.shared(value=value, name=name, strict=False)

def eye(N, M=None, k=0, dtype=_FLOATX, name=None):
    '''Instantiate an identity variable.
    '''
    return variable(np.eye(N, M, k), dtype, name)

def gather(reference, indices):
    '''reference: a tensor.
    indices: an int tensor of indices.

    Return: a tensor of same type as reference.
    '''
    return reference[indices]

train_X = np.random.randint(0,9,(2,20))

X = variable(train_X, dtype='int32')
B = eye(10)
out = gather(B, X)

print train_X
print out.eval()

train_X = [[1,2,3], [4,5,6,7,8,9]]
train_X = pad_sequences(train_X, maxlen=6)

model = Graph()
model.add_input(name='input', input_shape=(6,), dtype='int32')
model.add_node(OneHotEmbedding(output_dim=10, input_dim=10, mask_zero=True), name='one_hot', input='input')
model.add_output(name='output', input='one_hot')
model.compile(optimizer='SGD', loss={'output':'binary_crossentropy'})

print model.evaluate()

print model.predict({'input':train_X[0:10]})