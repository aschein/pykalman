import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import sys

rn.seed(1234)
from pykalman import KalmanFilter

if __name__ == '__main__':
    data_file = np.load('../../pykalman/datasets/icews/data.npz')
    N_TV = data_file['Y']  # TxV count matrix

    K = 10  # number of components
    V = 50  # number of columns to subset out
    N_TV = N_TV[:, :V]

    S = 10   # number of forecasting timesteps
    N_SV = N_TV[-S:, :]  # test set
    N_TV = N_TV[:-S, :]  # train set
    T = N_TV.shape[0]

    E_V = N_TV.mean(axis=0)
    Y_TV = N_TV - E_V  # TxV zero-mean real matrix
    Lambda_KK = np.diag(rn.uniform(-0.9,0.9,size=(K)))
    #Lambda_KK = rn.uniform(-0.75, 0.75, size=(K, K))  # initialize KxK state transition matrix
    Phi_VK = rn.normal(0, 0.25, size=(V, K))     # initialize VxK observation matrix
    #Phi_VK = rn.uniform(-0.75, 0.75, size=(V, K))     # initialize VxK observation matrix

    learning_config = {'stabilize':False,'diagonal': False,'compute_likelihood': True}
    kf = KalmanFilter(observation_matrices=Phi_VK,
                      transition_matrices=Lambda_KK,
                      learning_config=learning_config)  # stabilize=True uses the SVD trick to make sure eigenvalues<1

    em_vars = ['transition_covariance',
               'transition_matrices',
               'observation_matrices',
               'observation_covariance',
               'initial_state_covariance',
               'initial_state_mean']

    kf = kf.em(Y_TV, n_iter=25, em_vars=em_vars)  # fit the model

    Lambda_KK = kf.transition_matrices
    assert (np.abs(np.linalg.eigvals(Lambda_KK)) <= 1.).all()
    Phi_VK = kf.observation_matrices
    Sigma_KK = kf.transition_covariance
    D_VV = kf.observation_covariance

    # predict the test set (Y_SV)
    pred_Y_SV = np.zeros((S, V))
    z_K = kf.filter(Y_TV)[0][-1]
    for s in xrange(S):
        z_K = np.dot(Lambda_KK, z_K)
        pred_Y_SV[s] = np.dot(Phi_VK, z_K)

    print 'lds MAE: %.5f' % np.abs(N_SV - pred_Y_SV + E_V).mean()
    print 'lds RMSE: %.5f' % np.sqrt(((N_SV - pred_Y_SV + E_V)**2).mean())

    # comment in to plot the forecasts for each dimension
    # for v in xrange(V):
    #     plt.plot(np.arange(T+S), np.append(N_TV[:, v], N_SV[:, v]), 'o-')
    #     plt.plot(np.arange(T, T+S), pred_Y_SV[:, v] + E_V[v], 'o-', color='r')
    #     plt.axvline(T-1, color='g', linestyle='--')
    #     plt.show()
