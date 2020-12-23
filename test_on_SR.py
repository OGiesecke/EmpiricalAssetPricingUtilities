import numpy as np

def test_on_SRs(vR1,vR2,ik):
    
    vR1 = vR1.to_numpy()
    vR2 = vR2.to_numpy()
    
    # Size of the sample
    dT      = len(vR1);
    # GMM estimator of mu
    vmu     = np.array([np.mean(vR1) ,np.mean(vR1**2),np.mean(vR2), np.mean(vR2**2)]).T
    
    # Moment errors
    
    mf      = np.array([vR1, (vR1**2), vR2 ,(vR2**2)]).T - np.ones((dT,1)) * vmu
    
    # Compute the estimator of V
    
    mV      = np.zeros((4,4))
    for j in range(-ik,ik+1):
        
        maux    = np.zeros((4,4))
        
        for t in range(np.max((0,j)) , dT+np.min((0,j)) ) :
            #print(t)
            maux = maux + mf[t,:].reshape(4,1) @ mf[t-j,:].reshape(1,4) / dT

        mV      = mV + (ik+1-np.abs(j))/(ik+1) * maux

    
    # Compute G
    
    vG      = np.array([ (vmu[1]-vmu[0]**2)**(-1/2)+vmu[0]**2*(vmu[1]-vmu[0]**2)**(-3/2),
                        -0.5*vmu[0]*(vmu[1]-vmu[0]**2)**(-3/2),
                        -(vmu[3]-vmu[2]**2)**(-1/2)-vmu[2]**2*(vmu[3]-vmu[2]**2)**(-3/2),
                        0.5*vmu[2]*(vmu[3]-vmu[2]**2)**(-3/2)])
            
    # Compute cal(T) and the asymptotic variance of cal(T) (scaled by T)
    
    dcalT   = vmu[0]/np.sqrt(vmu[1]-vmu[0]**2)-vmu[2]/np.sqrt(vmu[3]-vmu[2]**2)
    dvarT   = vG.T @ mV @ vG / dT
    
    # Compute the p-value
    
    dt_stat = dcalT / np.sqrt(dvarT)
    return dt_stat 