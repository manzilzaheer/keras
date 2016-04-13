import theano
from theano import tensor as T
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