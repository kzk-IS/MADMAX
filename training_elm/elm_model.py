import numpy as np
import json
import math
import sys
import argparse
import re


class ExtremeLearningMachine(object):
    def __init__(self, n_unit, activation=None):
        self._activation = self._sig if activation is None else self._relu
        self._n_unit = n_unit

    @staticmethod
    def _relu(x):
        return x * (x > 0.0)

    @staticmethod
    def _sig(x):
        return 1. / (1 + np.exp(-x))

    @staticmethod
    def _add_bias_hstack(x):
        return np.hstack((x, np.ones((x.shape[0], 1))))


    @staticmethod
    def _add_bias_vstack(x):
        return np.vstack((x, np.ones((x.shape[1]))))


    def get_W0(self):
        return self.W0


    def get_W1(self):
        return self.W1
        
    
    def save_weights(self,file_name):
        W0 = self.get_W0()
        W1 = self.get_W1()
        np.savez(file_name,w0=W0,w1=W1)


    def load_weights(self,file_name):
        npz_file = np.load(file_name)
        W0 = npz_file["w0"]
        W1 = npz_file["w1"]
        self.W0, self.W1 = W0,W1
    

    def fit(self, X, y):
        self.W0 = np.random.random((X.shape[1], self._n_unit))
        X_add_bias = self._add_bias_hstack(X)
        w0_add_bias = self._add_bias_vstack(self.W0)

        z = self._activation(X_add_bias.dot(w0_add_bias))
        self.W1 = np.linalg.lstsq(z, y)[0]

    def predict(self, X):
        if not hasattr(self, 'W0'):
            raise UnboundLocalError('must fit before transform')
        X_add_bias = self._add_bias_hstack(X)
        w0_add_bias = self._add_bias_vstack(self.W0)
        z = self._activation(X_add_bias.dot(w0_add_bias))
        return np.argmax(z.dot(self.W1), axis=1)


    def transform(self, X):
        if not hasattr(self, 'W0'):
            raise UnboundLocalError('must fit before transform')
        X_add_bias = self._add_bias_hstack(X)
        w0_add_bias = self._add_bias_vstack(self.W0)
        z = self._activation(X_add_bias.dot(w0_add_bias))
        return z.dot(self.W1)

    def fit_transform(self, X, y):
        self.W0 = np.random.random((X.shape[1], self._n_unit))
        z = self._add_bias(self._activation(X.dot(self.W0)))
        self.W1 = np.linalg.lstsq(z, y)[0]
        return z.dot(self.W1)