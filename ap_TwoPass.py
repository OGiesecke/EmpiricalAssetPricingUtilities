import pandas as pd
import numpy as np
from python_nw import newey


def ap_TwoPass(mr,mf,ishanken_correction):
    dT, dN = mr.shape
    dT, dK = mf.shape

    valpha = np.empty((dN,1))
    mbeta = np.empty((dN,dK))
    valpha_t = np.empty((dN,1))
    mresid = np.empty((dT,dN))

    # Pass 1: Time-series regressions

    vones = np.ones((dT,1))
    for i in range(0,dN):
        vres = newey(mr[:,i],np.hstack((vones, mf)).reshape(dT,dK+1),0)
        valpha[i] = vres.beta[0]
        mbeta[i,:] = vres.beta[1:].transpose()
        valpha_t[i] = vres.tstat[0]
        mresid[:,i] = vres.resid


    # Pass 2: Time-series regressions

    vres = newey(np.mean(mr,0).transpose(),mbeta,0);
    vlambda = vres.beta;
    valpha = vres.resid;

    # Compute standard errors

    msigma = np.cov(mresid,rowvar=0)
    msigma_f = np.cov(mf,rowvar=0)
    meye_N = np.eye(dN)

    dcorrection = 1
    if ishanken_correction == 1:
        dcorrection = 1 + vlambda.transpose()@np.linalg.inv(msigma_f)@vlambda

    mcov_alpha = (1/dT) * (meye_N - mbeta@np.linalg.inv(mbeta.transpose()@mbeta)@mbeta.transpose()) @ msigma @ (meye_N - mbeta@np.linalg.inv(mbeta.transpose()@mbeta)@mbeta.transpose()) * dcorrection
    mcov_lambda = (1/dT) * (np.linalg.inv(mbeta.transpose()@mbeta)@mbeta.transpose()@msigma@mbeta@np.linalg.inv(mbeta.transpose()@mbeta) * dcorrection + msigma_f)
    valpha_t = valpha / np.sqrt(np.diag(mcov_alpha))
    vlambda_t = vlambda / np.sqrt(np.diag(mcov_lambda))

    # Asset pricing test

    dmodel_test_stat = valpha.transpose()@np.linalg.pinv(mcov_alpha)@valpha
    dmodel_p = 1-chi2.cdf(dmodel_test_stat,dN-dK)

    return vlambda, vlambda_t, valpha, valpha_t, dmodel_test_stat, dmodel_p, dcorrection
