import numpy as np

from .. import backend as K
from .recurrent import Recurrent, time_distributed_dense
from .. import activations, initializations, regularizers

class DeepSet(Recurrent):
    '''Deep Bloom Filter.

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    '''
    def __init__(self, hidden_dim, filter_size,
                 init='glorot_uniform', inner_init='glorot_uniform',
                 activation='tanh', inner_activation='sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.filter_size = filter_size
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U
        kwargs['return_sequences'] = True
        super(DeepSet, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        input_dim = input_shape[2]
        self.input_dim = input_dim

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensors of shape (output_dim)
            self.states = [None, None]

        # input-hidden transformation
        self.W_i = self.init((input_dim, self.hidden_dim), name='{}_W_i'.format(self.name))
        self.U_i = self.inner_init((self.hidden_dim, self.hidden_dim), name='{}_U_i'.format(self.name))
        self.b_i = K.zeros((self.hidden_dim,), name='{}_b_i'.format(self.name))

        # add gate transformation
        self.W_a = self.init((input_dim,1), name='{}_W_a'.format(self.name))
        self.U_a = self.inner_init((self.hidden_dim,1), name='{}_U_a'.format(self.name))
        self.b_a = K.zeros((self.output_dim,), name='{}_b_a'.format(self.name))

        # query gate transformation
        # self.W_q = self.init((input_dim,self.output_dim), name='{}_W_q'.format(self.name))
        # self.U_q = self.inner_init((self.hidden_dim,self.output_dim), name='{}_U_q'.format(self.name))
        # self.b_q = K.zeros((self.output_dim,), name='{}_b_q'.format(self.name))

        # add value transformation
        self.W_va = self.init((input_dim, self.filter_size), name='{}_W_va'.format(self.name))
        self.U_va = self.inner_init((self.hidden_dim, self.filter_size), name='{}_U_va'.format(self.name))
        self.b_va = K.zeros((self.filter_size,), name='{}_b_va'.format(self.name))

        # read value transformation
        # self.W_vr = self.init((input_dim, self.filter_size), name='{}_W_vr'.format(self.name))
        # self.U_vr = self.inner_init((self.hidden_dim, self.filter_size), name='{}_U_vr'.format(self.name))
        # self.b_vr = K.zeros((self.filter_size,), name='{}_b_vr'.format(self.name))

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(K.concatenate([self.W_i, self.W_a, self.W_va]))
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(K.concatenate([self.U_i, self.U_a, self.U_va]))
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(K.concatenate([self.b_i, self.b_a, self.b_va]))
            self.regularizers.append(self.b_regularizer)

        self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                  self.W_a, self.U_a, self.b_a,
                                  self.W_va, self.U_va, self.b_va]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.filter_size)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
            # K.set_value(self.states[2],
            #             np.zeros((input_shape[0], self.hidden_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.filter_size)),
                           K.zeros((input_shape[0], self.output_dim))]

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
        reducer0 = K.zeros((self.input_dim, self.hidden_dim))
        reducer1 = K.zeros((self.input_dim, self.filter_size))
        #reducer2 = K.zeros((self.input_dim, self.output_dim))
        initial_states = [K.dot(initial_state, reducer0), K.dot(initial_state, reducer1)] # (samples, hidden_dim), (samples, filter_size)
        return initial_states

    def preprocess_input(self, x, train=False):
        if train and (0 < self.dropout_W < 1):
            dropout = self.dropout_W
        else:
            dropout = 0
        input_shape = self.input_shape
        input_dim = input_shape[2]
        timesteps = input_shape[1]

        x_i = time_distributed_dense(x, self.W_i, self.b_i, dropout, input_dim, self.hidden_dim, timesteps)
        x_a = time_distributed_dense(x, self.W_a, self.b_a, dropout, input_dim, self.output_dim, timesteps)
        x_va = time_distributed_dense(x, self.W_va, self.b_va, dropout, input_dim, self.filter_size, timesteps)
        #x_q = time_distributed_dense(x, self.W_q, self.b_q, dropout, input_dim, self.output_dim, timesteps)
        #x_vr = time_distributed_dense(x, self.W_va, self.b_va, dropout, input_dim, self.filter_size, timesteps)
        return K.concatenate([x_i, x_a, x_va], axis=2)

    def step(self, x, states):
        h_tm1 = states[0]
        f_tm1 = states[1]
        if len(states) == 3:
            B_U = states[2]
        else:
            B_U = [1. for _ in range(3)]

        x_i = x[:, :self.hidden_dim]
        x_a = x[:, self.hidden_dim : self.hidden_dim + self.output_dim]
        x_va = x[:, self.hidden_dim + self.output_dim : self.hidden_dim + self.output_dim + self.filter_size]
        #x_vr = x[:, self.hidden_dim + self.filter_size : self.hidden_dim + 2*self.filter_size]
        #x_q = x[:, self.hidden_dim + 2*self.filter_size + self.output_dim : self.hidden_dim + 2*self.filter_size + 2*self.output_dim]

        h_t = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
        a = self.inner_activation(x_a + K.dot(h_tm1 * B_U[1], self.U_a))
        va = self.inner_activation(x_va + K.dot(h_tm1 * B_U[2], self.U_va))
        #q = self.inner_activation(x_q + K.dot(h_tm1 * B_U[3], self.U_q))
        #vr = self.inner_activation(x_vr + K.dot(h_tm1 * B_U[4], self.U_vr))

        f_t = self.activation(f_tm1 + K.dot(va,a))
        #y = (1-q) #+ q*(self.activationK.dot(vr,f_tm1))

        return f_t, [f_t, h_t]

    def get_constants(self, x, train=False):
        if train and (0 < self.dropout_U < 1):
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.dropout(ones, self.dropout_U) for _ in range(5)]
            return [B_U]
        return []

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "hidden_dim": self.hidden_dim,
                  "filter_size": self.filter_size,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U}
        base_config = super(DeepSet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class dumbFilter(Recurrent):
    '''Deep Bloom Filter.

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    '''
    def __init__(self, hidden_dim, filter_size,
                 init='glorot_uniform', activation='sigmoid',
                 W_regularizer=None,b_regularizer=None,
                 dropout_W=0., **kwargs):
        self.output_dim = 1
        self.hidden_dim = hidden_dim
        self.filter_size = filter_size
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W = dropout_W
        kwargs['return_sequences'] = True
        super(dumbFilter, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        input_dim = input_shape[2]
        self.input_dim = input_dim

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensors of shape (filter_size)
            self.states = [None]

        # input-hidden transformation
        self.W_i = self.init((self.hidden_dim, self.output_dim), name='{}_W_i'.format(self.name))
        self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(K.concatenate([self.W_i]))
            self.regularizers.append(self.W_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(K.concatenate([self.b_i]))
            self.regularizers.append(self.b_regularizer)

        self.trainable_weights = [self.W_i, self.b_i]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0], np.zeros((input_shape[0], self.filter_size)))
        else:
            self.states = [K.zeros((input_shape[0], self.filter_size))]

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, filter_size)
        initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
        reducer = K.zeros((self.input_dim, self.filter_size))
        initial_states = K.dot(initial_state, reducer) # (samples, filter_size)
        return initial_states

    def preprocess_input(self, x, train=False):
        if train and (0 < self.dropout_W < 1):
            dropout = self.dropout_W
        else:
            dropout = 0
        input_shape = self.input_shape
        input_dim = input_shape[2]
        timesteps = input_shape[1]

        a = x[:, :, :self.hidden_dim]
        va = x[:, :, self.hidden_dim:]
        a_p = time_distributed_dense(a, self.W_i, self.b_i, dropout, input_dim, self.hidden_dim, timesteps)
        return K.concatenate([a_p, va], axis=2)

    def step(self, x, states):
        f_tm1 = states

        a = self.activation(x[:, :self.output_dim])
        va = x[:, self.output_dim:]

        f = f_tm1 + K.repeat_elements(a, rep=self.filter_size, axis=1) * f_tm1
        y = a + (1-a)*K.sum(va*f_tm1)

        return y, [f]

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "hidden_dim": self.hidden_dim,
                  "filter_size": self.filter_size,
                  "init": self.init.__name__,
                  "activation": self.activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U}
        base_config = super(dumbFilter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LSTM2(Recurrent):
    '''Deep Bloom Filter.

    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U
        kwargs['return_sequences']=True
        super(LSTM2, self).__init__(**kwargs)


    def build(self):
        input_shape = self.input_shape
        input_dim = input_shape[2]
        self.input_dim = input_dim

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensors of shape (output_dim)
            self.states = [None, None]

        self.W_i = self.init((input_dim, self.output_dim),
                             name='{}_W_i'.format(self.name))
        self.U_i = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_i'.format(self.name))
        self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))

        self.W_f = self.init((input_dim, self.output_dim),
                             name='{}_W_f'.format(self.name))
        self.U_f = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_f'.format(self.name))
        self.b_f = self.forget_bias_init((self.output_dim,),
                                         name='{}_b_f'.format(self.name))

        self.W_c = self.init((input_dim, self.output_dim),
                             name='{}_W_c'.format(self.name))
        self.U_c = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_c'.format(self.name))
        self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))

        self.W_o = self.init((input_dim, self.output_dim),
                             name='{}_W_o'.format(self.name))
        self.U_o = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_o'.format(self.name))
        self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(K.concatenate([self.W_i,
                                                        self.W_f,
                                                        self.W_c,
                                                        self.W_o]))
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(K.concatenate([self.U_i,
                                                        self.U_f,
                                                        self.U_c,
                                                        self.U_o]))
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(K.concatenate([self.b_i,
                                                        self.b_f,
                                                        self.b_c,
                                                        self.b_o]))
            self.regularizers.append(self.b_regularizer)

        self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                  self.W_c, self.U_c, self.b_c,
                                  self.W_f, self.U_f, self.b_f,
                                  self.W_o, self.U_o, self.b_o]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights


    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim))]


    def preprocess_input(self, x, train=False):
        if train and (0 < self.dropout_W < 1):
            dropout = self.dropout_W
        else:
            dropout = 0
        input_shape = self.input_shape
        input_dim = input_shape[2]
        timesteps = input_shape[1]

        x_i = time_distributed_dense(x, self.W_i, self.b_i, dropout,
                                     input_dim, self.output_dim, timesteps)
        x_f = time_distributed_dense(x, self.W_f, self.b_f, dropout,
                                     input_dim, self.output_dim, timesteps)
        x_c = time_distributed_dense(x, self.W_c, self.b_c, dropout,
                                     input_dim, self.output_dim, timesteps)
        x_o = time_distributed_dense(x, self.W_o, self.b_o, dropout,
                                     input_dim, self.output_dim, timesteps)
        return K.concatenate([x_i, x_f, x_c, x_o], axis=2)


    def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        if len(states) == 3:
            B_U = states[2]
        else:
            B_U = [1. for _ in range(4)]

        x_i = x[:, :self.output_dim]
        x_f = x[:, self.output_dim: 2 * self.output_dim]
        x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
        x_o = x[:, 3 * self.output_dim:]

        i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
        f = self.inner_activation(x_f + K.dot(h_tm1 * B_U[1], self.U_f))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c))
        o = self.inner_activation(x_o + K.dot(h_tm1 * B_U[3], self.U_o))

        h = o * self.activation(c)
        return h, [h, c]


    def get_constants(self, x, train=False):
        if train and (0 < self.dropout_U < 1):
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.dropout(ones, self.dropout_U) for _ in range(4)]
            return [B_U]
        return []


    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U}
        base_config = super(LSTM2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))