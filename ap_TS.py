import pandas as pd
import numpy as np
from python_nw import newey
from scipy.stats import f
from scipy.stats import chi2

def ap_TS(mr,mf):

    dT, dN = mr.shape
    dT, dK = mf.shape

    vbeta = np.empty((dN,dK))
    valpha = np.empty((dN,1))
    valpha_t = np.empty((dN,1))
    mresid = np.empty((dT,dN))

    # Time-series regressions

    vones = np.ones((dT,1))
    for i in range(0,dN):
        vres = newey(mr[:,i],np.hstack((vones, mf)).reshape(dT,dK+1),0)
        valpha[i] = vres.beta[0]
        vbeta[i,:] = vres.beta[1:]
        valpha_t[i] = vres.tstat[0]
        mresid[:,i] = vres.resid

    ## Properties of risk premia

    vlambda = np.mean(mf,0).transpose();
    if dK == 1:
        vlambda_t = vlambda / np.sqrt(np.var(mf)/dT)
    else:
        vlambda_t = vlambda / np.sqrt(np.diag(np.cov(mf,rowvar=0))/dT)

    ## GRS test
    if dK == 1:
        dGRS = ((dT-dN-dK)/dN)*1/(1+np.mean(mf,0) / np.var(mf) * np.mean(mf,0)) * valpha.transpose()@np.linalg.inv(np.cov(mresid,rowvar=0,bias=True))@valpha
    else:
        dGRS = ((dT-dN-dK)/dN)*1/(1+np.mean(mf,0)@np.linalg.inv(np.cov(mf,rowvar=0,bias=True))@np.mean(mf,0).transpose()) * valpha.transpose()@np.linalg.inv(np.cov(mresid,rowvar=0,bias=True))@valpha
    dGRS_p = 1-f.cdf(dGRS,dN,dT-dN-dK);

    return valpha, valpha_t, vlambda, vlambda_t, dGRS, dGRS_p, vbeta
