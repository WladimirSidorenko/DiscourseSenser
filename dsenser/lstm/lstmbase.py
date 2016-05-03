#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing abstract interface class for Wang sense calssification.

Attributes:
WangBaseSenser (class):
  abstract class defining interface for explicit and implicit classifier

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

from dsenser.nnbase import NNBaseSenser
from dsenser.theano_utils import floatX, theano, HE_UNIFORM, ORTHOGONAL, TT

import numpy as np

##################################################################
# Variables and Constants
TRUNCATE_GRADIENTS = 20


##################################################################
# Class
class LSTMBaseSenser(NNBaseSenser):
    """Abstract class for disambiguating relation senses.

    Attrs:
    n_y (int): number of distinct classes

    Methods:

    """

    def _init_nn(self):
        """Initialize neural network.

        Args:
        (void)

        Returns:
        (void)

        """
        self.intm_dim = max(100, self.ndim - (self.ndim - self.n_y) / 2)
        # indices of word embeddings
        self.W_INDICES_ARG1 = TT.ivector(name="W_INDICES_ARG1")
        self.W_INDICES_ARG2 = TT.ivector(name="W_INDICES_ARG2")
        # connective's index
        self.CONN_INDEX = TT.iscalar(name="CONN_INDEX")
        # initialize the matrix of word embeddings
        self.init_w_emb()
        # word embeddings of the arguments
        self.EMB_ARG1 = self.W_EMB[self.W_INDICES_ARG1]
        self.EMB_ARG2 = self.W_EMB[self.W_INDICES_ARG2]
        # connective's embedding
        self._init_conn_emb()
        self.EMB_CONN = self.CONN_EMB[self.CONN_INDEX]
        # initialize forward LSTM unit
        invars = ((self.EMB_ARG1, False), (self.EMB_ARG2, False))
        params, outvars = self._init_lstm(invars)
        self._params.extend(params)
        self.F_OUT_ARG1, self.F_OUT_ARG2 = outvars
        self.F_ARG1 = TT.mean(self.F_OUT_ARG1, axis=0)
        self.F_ARG2 = TT.mean(self.F_OUT_ARG2, axis=0)
        # define final units
        self.I = TT.concatenate((self.F_ARG1, self.F_ARG2,
                                 self.EMB_CONN))
        self.I2Y = theano.shared(value=HE_UNIFORM((self.n_y,
                                                   self.intm_dim * 3)),
                                 name="I2Y")
        self.y_bias = theano.shared(value=HE_UNIFORM((1, self.n_y)),
                                    name="y_bias")
        self._params.extend([self.I2Y, self.y_bias])
        self.Y_pred = TT.nnet.softmax(TT.dot(self.I2Y, self.I).T + self.y_bias)
        # initialize cost and optimization functions
        self.Y_gold = TT.vector(name="Y_gold")
        self._cost = TT.sum((self.Y_pred - self.Y_gold) ** 2)
        self._dev_cost = (TT.argmax(self.Y_pred) != TT.argmax(self.Y_gold))
        grads = TT.grad(self._cost, wrt=self._params)
        self._init_funcs(grads)

    def _init_lstm(self, a_invars, a_sfx="-forward"):
        """Initialize LSTM layer.

        Args:
        a_invars (list(theano.shared)):
        list of input parameters as symbolic theano variable
        a_sfx (str):
        suffix to use for function and parameter names

        Returns:
        (2-tuple)
        parameters to be optimized and list of symbolic outputs from the
        function

        """
        intm_dim = self.intm_dim
        # initialize transformation matrices and bias term
        W_dim = (intm_dim, self.ndim)
        W = np.concatenate([ORTHOGONAL(W_dim), ORTHOGONAL(W_dim),
                            ORTHOGONAL(W_dim), ORTHOGONAL(W_dim)],
                           axis=0)
        W = theano.shared(value=W, name="W" + a_sfx)

        U_dim = (intm_dim, intm_dim)
        U = np.concatenate([ORTHOGONAL(U_dim), ORTHOGONAL(U_dim),
                            ORTHOGONAL(U_dim), ORTHOGONAL(U_dim)],
                           axis=0)
        U = theano.shared(value=U, name="U" + a_sfx)

        V = ORTHOGONAL(U_dim)   # V for vendetta
        V = theano.shared(value=V, name="V" + a_sfx)

        b_dim = (1, intm_dim * 4)
        b = theano.shared(value=HE_UNIFORM(b_dim), name="b" + a_sfx)

        params = [W, U, V, b]

        # initialize dropout units
        w_do = theano.shared(value=floatX(np.ones((4 * intm_dim,))),
                             name="w_do")
        w_do = self._init_dropout(w_do)
        u_do = theano.shared(value=floatX(np.ones((4 * intm_dim,))),
                             name="u_do")
        u_do = self._init_dropout(u_do)

        # custom function for splitting up matrix parts
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        # define recurrent LSTM unit
        def _step(x_, h_, c_,
                  W, U, V, b, w_do, u_do):
            """Recurrent LSTM unit.

            Note:
            The general order of function parameters to fn is:
            sequences (if any), prior result(s) (if needed),
            non-sequences (if any)

            Args:
            x_ (theano.shared): input vector
            h_ (theano.shared): output vector
            c_ (theano.shared): memory state
            W (theano.shared): input transform matrix
            U (theano.shared): inner-state transform matrix
            V (theano.shared): output transform matrix
            b (theano.shared): bias vector
            w_do (TT.col): dropout unit for the W matrix
            u_do (TT.col): dropout unit for the U matrix

            Returns:
            (2-tuple(h, c))
            new hidden and memory states

            """
            # pre-compute common terms:
            # W \in R^{236 x 100}
            # x \in R^{1 x 100}
            # U \in R^{236 x 59}
            # h \in R^{1 x 59}
            # b \in R^{1 x 236}
            # w_do \in R^{236 x 1}
            # u_do \in R^{236 x 1}

            # xhb \in R^{1 x 236}
            xhb = (TT.dot(W * w_do.dimshuffle((0, 'x')), x_.T) +
                   TT.dot(U * u_do.dimshuffle((0, 'x')), h_.T)).T + b
            # i \in R^{1 x 59}
            i = TT.nnet.sigmoid(_slice(xhb, 0, intm_dim))
            # f \in R^{1 x 59}
            f = TT.nnet.sigmoid(_slice(xhb, 1, intm_dim))
            # c \in R^{1 x 59}
            c = TT.tanh(_slice(xhb, 2, intm_dim))
            c = i * c + f * c_
            # V \in R^{59 x 59}
            # o \in R^{1 x 59}
            o = TT.nnet.sigmoid(_slice(xhb, 3, intm_dim) +
                                TT.dot(V, c.T).T)
            # h \in R^{1 x 59}
            h = o * TT.tanh(c)
            # return current output and memory state
            return h.flatten(), c.flatten()

        m = 0
        n = intm_dim
        ov = None
        outvars = []
        for iv, igbw in a_invars:
            m = iv.shape[0]
            ret, _ = theano.scan(_step,
                                 sequences=[iv],
                                 outputs_info=[floatX(np.zeros((n,))),
                                               floatX(np.zeros((n,)))],
                                 non_sequences=[W, U, V, b, w_do, u_do],
                                 name="LSTM" + str(iv) + a_sfx,
                                 n_steps=m,
                                 truncate_gradients=TRUNCATE_GRADIENTS,
                                 go_backwards=igbw)
            ov = ret[0]
            outvars.append(ov)
        return params, outvars
