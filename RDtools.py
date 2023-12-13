import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.linalg import null_space,pinv

def binent(p,bits=False):
    return phi(p,bits) + phi(1-p,bits)

# this will return 0 if p < 0 or p > 1, and will print a warning if the value exceeds a tolerance
def phi(p,bits=False):
    tol = 1e-14
    if p <= 0:
        if np.abs(p) > tol:
            warnings.warn('in phi: Exceeds input tolerance (' + str(np.abs(p)) + ' vs. ' + str(tol) +')')
        return 0
    if p >= 1:
        if p-1 > tol:
            warnings.warn('in phi: Exceeds input tolerance (' + str(p-1) + ' vs. ' + str(tol) +')')
        return 0
    if bits is True:
        return p*np.log2(1/p)
    else:
        return p*np.log(1/p)

# pygx is a 1x2 array where element 0 is p(y=1|x=0), and element 1 is p(y=1|x=1)
def binMI(px,pygx,bits=False):
    py = px*pygx[1] + (1-px)*pygx[0]
    result = binent(py,bits) # H(Y)
    result = result - px*binent(pygx[1],bits) - (1-px)*binent(pygx[0],bits) # H(Y|X)
    return result

# for the joint distribution, expressed as p(x,y): [p(0,0),p(0,1),p(1,0),p(1,1)]
def binMI_joint(pyx,bits=False):
    tol = 1e-14
    px = [pyx[0]+pyx[1],pyx[2]+pyx[3]]
    py = [pyx[0]+pyx[2],pyx[1]+pyx[3]]
    hxy = np.sum([phi(i,bits) for i in pyx])
    result = binent(px[1],bits) + binent(py[1],bits) - hxy
    if (result < 0):
        if (np.abs(result) > tol):
            warnings.warn('in binMI_joint: Exceeds output tolerance (' + str(np.abs(result)) + ' vs. ' + str(tol) +')')
        return 0
    return result

# given the joint distribution as specified above, returns the entropy H(X|Y)
def h_x_given_y(pyx):
    py0 = pyx[0] + pyx[2]
    hy = binent(py0)
    hxy = 0
    for p in pyx:
        hxy += phi(p)
    return hxy - hy

# here pxgy is in the same form as the strategy, while px is a scalar (i.e. ps, probability that x = 1)
def joint_p_from_pxgy_and_px(pxgy,px):
    X = np.array([[1-px,px]])
    V = np.array([[pxgy[0],pxgy[2]],[pxgy[1],pxgy[3]]])
    Y = X @ np.linalg.inv(V)
    joint = np.diag(Y[0,:]) @ V # Y needs to be diagonal, but on which side?
    return np.array([joint[0,0],joint[1,0],joint[0,1],joint[1,1]])

def getConstantDistortionDistributions(ps,d,Dmax):
    # we need to find solutions to the equation M p = d
    # for p, order of entries: [p(0,0),p(0,1),p(1,0),p(1,1)]
    # with p(s,\hat{s})

    # enforces distortion constraint - solution is Dmax
    M = np.array([d])

    # enforces probability constraint: sum of p = 1 - solution is 1
    M = np.vstack((M,[1,1,1,1]))

    # enforces marginal constraint: last 2 entries sum to ps - solution is ps
    M = np.vstack((M,[0,0,1,1]))

    # solutions in the vector d should be given in the same order as the above
    d = np.array([np.array([Dmax,1,ps])]).T

    # one solution to the problem is given by the pseudoinverse
    p0 = pinv(M) @ d

    # the remainder are given by the matrix kernel
    Mk = null_space(M)

    # because the null space is likely one-dimensional, solve each coordinate for min and max
    b_all = np.zeros((4,2))
    for i in range(0,4):
        # solve Mk * c + p0 = 0 and Mk * c + p0 = 1
        r = [-1*p0[i][0] / Mk[i][0] , (1 - p0[i][0]) / Mk[i][0]]
        if r[0] < r[1]:
            # minimum element should be first
            b_all[i,0] = r[0]
            b_all[i,1] = r[1]
        else:
            b_all[i,0] = r[1]
            b_all[i,1] = r[0]

    # intersection of all ranges: max of minima, min of maxima
    bounds = [np.max(b_all[:,0]),np.min(b_all[:,1])]

    params = {}
    params['p0'] = p0
    params['Mk'] = Mk
    params['bounds'] = bounds

    return params

def getMinRate(params,numPoints=1000):
    p0 = params['p0']
    Mk = params['Mk']
    bounds = params['bounds']

    if bounds[0] > bounds[1]:
        return None

    q_v = np.linspace(bounds[0],bounds[1],1000)
    p_m = np.zeros((4,len(q_v)))
    mi_v = np.zeros(len(q_v))
    for i,q in enumerate(q_v):
        p_m[:,i] = (p0 + q*Mk)[:,0]
        # this shouldn't happen but due to numerical issues there might be a tiny negative component,
        # or a tiny component exceeding 1,
        # which messes up the mutual information calculation
        if np.any(p_m[:,i] < 0) or np.any(p_m[:,i] > 1):
            # just throw out this value - setting MI to infinity guarantees that the minimum won't pick it
            mi_v[i] = np.inf
        else:
            mi_v[i] = binMI_joint(p_m[:,i])

    rateResult = {}
    rateResult['p'] = p_m[:,np.argmin(mi_v)]
    rateResult['mi'] = mi_v[np.argmin(mi_v)]

    return rateResult

def getRD(ps,d,numPoints=100):
    r_v = np.array([])
    Dmax_v = np.array([])
    p_v = []
    Dmax_try_v = np.linspace(np.min(d),np.max(d),numPoints)
    for Dmax in Dmax_try_v:
        params = getConstantDistortionDistributions(ps,d,Dmax)
        r = getMinRate(params)
        if r is not None:
            if (len(r_v) == 0) or (r['mi'] <= r_v[-1]):
                r_v = np.append(r_v,r['mi'])
                Dmax_v = np.append(Dmax_v,Dmax)
                p_v.append(r['p'])

    result = {}
    result['r_v'] = r_v
    result['Dmax_v'] = Dmax_v
    result['p'] = p_v
    return result

# assume 2x2 for now
# strat order: (x,y) = [(0,0),(0,1),(1,0),(1,1)]
# note this means the first and third, and second and fourth, elements should sum to 1# returns a vector of -\log t_x^{(y)}
# given R and s
def getDistortionFunction(R,s):
    return -1*np.log(getT(R,s))

def getT(R,s):
    rows,cols = np.shape(R)
    Qinv = np.diag((np.linalg.inv(R) @ (np.array([np.ones(cols)]).T))[:,0])
    B = R @ Qinv

    t = np.zeros(4)
    foo = np.array([[s[0],s[2]],[s[1],s[3]]]) @ B
    t[0] = foo[0,0]
    t[2] = foo[0,1]
    t[1] = foo[1,0]
    t[3] = foo[1,1]
    return t

def getSfromT(R,t):
    rows,cols = np.shape(R)
    tm = np.array([[t[0],t[2]],[t[1],t[3]]])
    Qinv = np.diag((np.linalg.inv(R) @ (np.array([np.ones(cols)]).T))[:,0])
    B = R @ Qinv
    Binv = np.linalg.inv(B)
    s = np.zeros(4)
    sm = tm @ Binv
    s[0] = sm[0,0]
    s[2] = sm[0,1]
    s[1] = sm[1,0]
    s[3] = sm[1,1]
    return s

# returns the xth diagonal element of Q
def getLambdaStarX(R,x=None):
    cols = np.shape(R)[1]
    Qinv = np.diag((np.linalg.inv(R) @ (np.array([np.ones(cols)]).T))[:,0])
    Q = np.linalg.inv(Qinv)
    if x is None:
        return np.log(np.diag(Q))
    return np.log(Q[x,x])

def getFullDistortionFunction(R,s):
    d = getDistortionFunction(R,s)
    d[0] -= getLambdaStarX(R,0)
    d[1] -= getLambdaStarX(R,0)
    d[2] -= getLambdaStarX(R,1)
    d[3] -= getLambdaStarX(R,1)
    return d

def getAvgLambdaStar(R,px):
    return np.sum(getLambdaStarX(R)*np.array([1-px,px]))

def minimumPracticalDistortion(R,px):
    # for now, assume the minimum distortion elements are on the diagonal
    L = -1*np.log(np.diag(R))
    return np.sum(L * np.array([1-px,px]))