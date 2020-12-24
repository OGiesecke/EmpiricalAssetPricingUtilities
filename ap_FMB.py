import pandas as pd
import numpy as np
from python_nw import newey



def ap_FMB(mr,mf):
    # Initialize

    dT, dN = mr.shape
    dT, dK = mf.shape

    valpha = np.empty((dN,1))
    mbeta = np.empty((dN,dK))
    mresid =np.empty((dT,dN))
    mlambda_t = np.empty((dT,dK))
    malpha_t = np.empty((dT,dN))

    # Pass 1: Time-series regressions

    vones = np.ones((dT,1))
    for i in range(0,dN):
        vres = newey(mr[:,i],np.hstack((vones, mf)).reshape(dT,dK+1),0)
        valpha[i] = vres.beta[0]
        mbeta[i,:] = vres.beta[1:].transpose()
        mresid[:,i] = vres.resid


    # Pass 2: Time-series regressions
    for t in range(0,dT):
        vres = newey(mr[t,:].transpose(),mbeta,0)
        mlambda_t[t,:] = vres.beta.transpose()
        malpha_t[t,:] = vres.resid.transpose()

    valpha = np.mean(malpha_t,0).transpose()
    vlambda = np.mean(mlambda_t,0).transpose()

    # Compute standard errors

    mcov_alpha =1/dT*np.cov(malpha_t,rowvar=0)
    mcov_lambda = 1/dT*np.cov(mlambda_t,rowvar=0)
    valpha_t = valpha / np.sqrt(np.diag(mcov_alpha))
    vlambda_t = vlambda / np.sqrt(np.diag(mcov_lambda))

    # Asset pricing test

    dmodel_test_stat = valpha.transpose()@np.linalg.pinv(mcov_alpha)@valpha
    dmodel_p = 1-chi2.cdf(dmodel_test_stat,dN-dK)

    return vlambda, vlambda_t, valpha, valpha_t, dmodel_test_stat, dmodel_p
