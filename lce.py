"""
Python Implementation of Local Collective Embeddings

__author__ : Abhishek Thakur
__original__ : https://github.com/msaveski/LCE

"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import sparse

eps = 7. / 3 - 4. / 3 - 1


def L2_norm_row(X):
    return sparse.spdiags(1. / (np.sqrt(sum(X * X, 2)) + eps), 0, len(X), len(X)) * X


def tr(A, B):
    x = np.multiply(A, B)
    return (x.sum(axis=0)).sum(axis=0)


def construct_A(X, k, binary=False):

    nbrs = NearestNeighbors(n_neighbors=1 + k).fit(X)
    if binary:
        return nbrs.kneighbors_graph(X)
    else:
        return nbrs.kneighbors_graph(X, mode='distance')


def LCE(Xs, Xu, A, k=15, alpha=0.1, beta=0.05, lamb=0.001, epsilon=0.01, maxiter=150, verbose=True):

    n = Xs.shape[0]
    v1 = Xs.shape[1]
    v2 = Xu.shape[1]

    W = abs(np.random.rand(n, k))
    Hs = abs(np.random.rand(k, v1))
    Hu = abs(np.random.rand(k, v2))

    D = sparse.dia_matrix((A.sum(axis=0), 0), A.shape)

    gamma = 1. - alpha
    trXstXs = tr(Xs, Xs)
    trXutXu = tr(Xu, Xu)

    WtW = W.T.dot(W)
    WtXs = W.T.dot(Xs)
    WtXu = W.T.dot(Xu)
    WtWHs = WtW.dot(Hs)
    WtWHu = WtW.dot(Hu)
    DW = D.dot(W)
    AW = A.dot(W)

    itNum = 1
    delta = 2.0 * epsilon

    ObjHist = []

    while True:

        # update H
        Hs_1 = np.divide(
            (alpha * WtXs), np.maximum(alpha * WtWHs + lamb * Hs, 1e-10))
        Hs = np.multiply(Hs, Hs_1)

        Hu_1 = np.divide(
            (gamma * WtXu), np.maximum(gamma * WtWHu + lamb * Hu, 1e-10))
        Hu = np.multiply(Hu, Hu_1)

        # update W
        W_t1 = alpha * Xs.dot(Hs.T) + gamma * Xu.dot(Hu.T) + beta * AW
        W_t2 = alpha * W.dot(Hs.dot(Hs.T)) + gamma * \
            W.dot(Hu.dot(Hu.T)) + beta * DW + lamb * W
        W_t3 = np.divide(W_t1, np.maximum(W_t2, 1e-10))
        W = np.multiply(W, W_t3)

        # calculate objective function
        WtW = W.T.dot(W)
        WtXs = W.T.dot(Xs)
        WtXu = W.T.dot(Xu)
        WtWHs = WtW.dot(Hs)
        WtWHu = WtW.dot(Hu)
        DW = D.dot(W)
        AW = A.dot(W)

        tr1 = alpha * (trXstXs - 2. * tr(Hs, WtXs) + tr(Hs, WtWHs))
        tr2 = gamma * (trXutXu - 2. * tr(Hu, WtXu) + tr(Hu, WtWHu))
        tr3 = beta * (tr(W, DW) - tr(W, AW))
        tr4 = lamb * (np.trace(WtW) + tr(Hs, Hs) + tr(Hu, Hu))

        Obj = tr1 + tr2 + tr3 + tr4
        ObjHist.append(Obj)

        if itNum > 1:
            delta = abs(ObjHist[-1] - ObjHist[-2])
            if verbose:
                print "Iteration: ", itNum, "Objective: ", Obj, "Delta: ", delta
            if itNum > maxiter or delta < epsilon:
                break

        itNum += 1

    return W, Hu, Hs

