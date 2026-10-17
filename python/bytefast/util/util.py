#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: The purpose of this file is not to accumulate all useful utility
# functions. This file should contain very commonly used and requested functions
# (such as test). If you think you have a function at that level, please create
# an issue and we will happily review your suggestion. This file is also not supposed
# to pull in dependencies outside of numpy/scipy without very good reasons. For
# example, this file should not use sklearn and matplotlib to produce a t-sne
# plot of word embeddings or such.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen


def test(predictions, labels, k=1):
    """
    Return precision and recall modeled after fasttext's test
    """
    precision = 0.0
    nexamples = 0
    nlabels = 0
    for prediction, labels in zip(predictions, labels):
        for p in prediction:
            if p in labels:
                precision += 1
        nexamples += 1
        nlabels += len(labels)
    return (precision / (k * nexamples), precision / nlabels)


def find_nearest_neighbor(query, vectors, ban_set, cossims=None):
    """
    query is a 1d numpy array corresponding to the vector to which you want to
    find the closest vector
    vectors is a 2d numpy array corresponding to the vectors you want to consider
    ban_set is a set of indicies within vectors you want to ignore for nearest match
    cossims is a 1d numpy array of size len(vectors), which can be passed for efficiency

    returns the index of the closest match to query within vectors

    """
    if cossims is None:
        cossims = np.matmul(vectors, query, out=cossims)
    else:
        np.matmul(vectors, query, out=cossims)
    rank = len(cossims) - 1
    result_i = np.argpartition(cossims, rank)[rank]
    while result_i in ban_set:
        rank -= 1
        result_i = np.argpartition(cossims, rank)[rank]
    return result_i


def _reduce_matrix(X_orig, dim, eigv):
    """
    Reduces the dimension of a (m × n)   matrix `X_orig` to
                          to a (m × dim) matrix `X_reduced`
    It uses only the first 100000 rows of `X_orig` to do the mapping.
    Matrix types are all `np.float32` in order to avoid unncessary copies.
    """
    if eigv is None:
        mapping_size = 100000
        X = X_orig[:mapping_size]
        X = X - X.mean(axis=0, dtype=np.float32)
        C = np.divide(np.matmul(X.T, X), X.shape[0] - 1, dtype=np.float32)
        _, U = np.linalg.eig(C)
        eigv = U[:, :dim]

    X_reduced = np.matmul(X_orig, eigv)

    return (X_reduced, eigv)


def reduce_model(ft_model, target_dim):
    """
    ft_model is an instance of `_ByteFast` class
    This function computes the PCA of the input and the output matrices
    and sets the reduced ones.
    """
    inp_reduced, proj = _reduce_matrix(
        ft_model.get_input_matrix(), target_dim, None)
    out_reduced, _ = _reduce_matrix(
        ft_model.get_output_matrix(), target_dim, proj)

    ft_model.set_matrices(inp_reduced, out_reduced)

    return ft_model
