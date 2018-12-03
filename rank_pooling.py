# -*- coding: utf-8 -*-

'''
@project: Rank pooling
@author: MRzzm
@E-mail: zhangzhimeng1@gmail.com
@github: https://github.com/MRzzm/rank-pooling-python.git
'''
import numpy as np
import scipy.sparse
from sklearn import svm

def smoothSeq(seq):

    res = np.cumsum(seq, axis=1)
    seq_len = np.size(res, 1)
    res = res / np.expand_dims(np.linspace(1, seq_len, seq_len), 0)
    return res

def rootExpandKernelMap(data):

    element_sign=np.sign(data)
    nonlinear_value=np.sqrt(np.fabs(data))
    return np.vstack((nonlinear_value*(element_sign>0),nonlinear_value*(element_sign<0)))

def getNonLinearity(data,nonLin='ref'):

    # we don't provide the Chi2 kernel in our code
    if nonLin=='none':
        return data
    if nonLin=='ref':
        return rootExpandKernelMap(data)
    elif nonLin=='tanh':
        return np.tanh(data)
    elif nonLin=='ssr':
        return np.sign(data)*np.sqrt(np.fabs(data))
    else:
        raise("We don't provide {} non-linear transformation".format(nonLin))

def normalize(seq,norm='l2'):

    if norm=='l2':
        seq_norm = np.linalg.norm(seq, ord=2, axis=0)
        seq_norm[seq_norm == 0] = 1
        seq_norm = seq / np.expand_dims(seq_norm, 0)
        return seq_norm
    elif norm=='l1':
        seq_norm=np.linalg.norm(seq,ord=1,axis=0)
        seq_norm[seq_norm==0]=1
        seq_norm=seq/np.expand_dims(seq_norm,0)
        return seq_norm
    else:
        raise("We only provide l1 and l2 normalization methods")



def rank_pooling(time_seq,C = 1,NLStyle = 'ssr'):
    '''
    This function only calculate the positive direction of rank pooling.
    :param time_seq: D x T
    :param C: hyperparameter
    :param NLStyle: Nonlinear transformation.Including: 'ref', 'tanh', 'ssr'.
    :return: Result of rank pooling
    '''

    seq_smooth=smoothSeq(time_seq)
    seq_nonlinear=getNonLinearity(seq_smooth,NLStyle)
    seq_norm=normalize(seq_nonlinear)
    seq_len=np.size(seq_norm, 1)
    Labels=np.array(range(1,seq_len+1))
    seq_svr=scipy.sparse.csr_matrix(np.transpose(seq_norm))
    svr_model = svm.LinearSVR(epsilon=0.1, tol=0.001, C=C, loss='squared_epsilon_insensitive', fit_intercept=False, dual=False)
    svr_model.fit(seq_svr,Labels)
    return svr_model.coef_