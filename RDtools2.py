#!/usr/bin/env python
# coding: utf-8

import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
import warnings
from scipy.linalg import null_space,pinv,inv
from scipy.optimize import linprog
from ProgressBar import ProgressBar


# # Overview
#
# Here we will have $p(x,y)$, $p(x|y)$, and $d(x,y)$ as matrices, of the form
# $$
# P_{xy} =
# \left[\begin{array}{cccc}
# p(x=0,y=0) & p(x=1,y=0) & \cdots & p(x=n_x-1,y=0) \\
# p(x=0,y=1) & p(x=1,y=1) & \cdots & p(x=n_x-1,y=1) \\
# \vdots & \vdots & \ddots & \vdots \\
# p(x=0,y=n_y-1) & p(x=1,y=n_y-1) & \cdots & p(x=n_x-1,y=n_y-1)
# \end{array}\right]
# $$
# where $n_x$ and $n_y$ are, respectively, the alphabet sizes of $x$ and $y$. We will specify $p(x|y)\rightarrow P_{x|y}$ and $d(x,y)\rightarrow D_{xy}$ in the same order as above.
#
# This will require matrix stacking and destacking, and methods will be written to accomplish this.


# this will return 0 if p < 0 or p > 1, and will print a warning if the value exceeds a tolerance
def phi(p,bits=False,debug=False):
    tol = 1e-14
    if p <= 0:
        if (np.abs(p) > tol) and (debug is True):
            warnings.warn('in phi: Exceeds input tolerance (' + str(np.abs(p)) + ' vs. ' + str(tol) +')')
        return 0
    if p >= 1:
        if (p-1 > tol) and (debug is True):
            warnings.warn('in phi: Exceeds input tolerance (' + str(p-1) + ' vs. ' + str(tol) +')')
        return 0
    if bits is True:
        return p*np.log2(1/p)
    else:
        return p*np.log(1/p)


def stack(M):
    (ny,nx) = np.shape(M)
    result = np.zeros(nx*ny)
    for i in range(ny):
        result[(i*nx):((i+1)*nx)] = M[i,:]
    return result

def unstack(v,ny,nx):
    M = np.zeros((ny,nx))
    for i in range(ny):
        M[i,:] = v[(i*nx):((i+1)*nx)]
    return M


# Calculates the mutual information in the joint distribution Pxy
def MI(Pxy,bits=False):
    (ny,nx) = np.shape(Pxy)
    px = np.ones(ny) @ Pxy
    Hx = np.sum([phi(i,bits=bits) for i in px])
    py = Pxy @ np.ones(nx)
    Hy = np.sum([phi(i,bits=bits) for i in py])
    Hxy = np.sum([phi(float(i),bits=bits) for i in np.nditer(Pxy)])
    return Hx + Hy - Hxy

def h_x_given_y(Pxy):
    (ny,nx) = np.shape(Pxy)
    py = Pxy @ np.ones(nx)
    result = np.zeros((ny,nx))
    for i in range(ny):
        result[i,:] = Pxy[i,:] / py[i]
    return result

# px is a vector of the form: [p(x=0),p(x=1),...,p(x=nx-1)]
def getConstantDistortionDistributions(px,Dxy,Dmax):
    (ny,nx) = np.shape(Dxy)

    stackDxy = stack(Dxy)

    # enforces distortion constraint - solution is Dmax
    M = np.array(stackDxy)

    # enforces probability constraint - total probability is 1
    # M = np.vstack((M,np.ones(len(M))))

    # enforces marginal constraint: marginal distribution Pr(x=1) must equal px
    # by doing it for all x, this also enforces the overall probability constraint (sum=1)
    for i in range(nx):
        foo = np.zeros(len(stackDxy))
        foo[i::nx] = np.ones(ny)
        M = np.vstack((M,foo))

    # right hand side of the equation
    r = np.append(np.array([Dmax]),px)

    # matrix kernel gives the space of solutions
    Mk = null_space(M)

    # need a starting point - given by the pseudoinverse
    p0 = feasiblePoint(M,r,Mk)

    if p0 is None:
        return None

    params = {}
    params['p0'] = p0
    params['Mk'] = Mk
    params['ny'] = ny
    params['nx'] = nx

    return params

# finds one feasible point in the set of constant distortion distributions,
# or returns None if it can't find any
# Uses the same matrices as provided by getConstantDistortionDistributions, and
# is intended for use by that function
def feasiblePoint(M,r,Mk,tol=1e-2,doWarnings=False,useRandom=False):
    result = pinv(M) @ r

    # if the pseudoinverse gives a good solution, return it
    if (np.all(result > 0)):
        return result

    # otherwise use linear programming to find a solution
    if doWarnings is True:
        print('Using linear programming!')
    (rows,cols) = np.shape(M)
    c = np.ones(cols) # objective doesn't matter as long as we can find a single feasible point
    c[0] = 0 # however, can't sum to 1 as that is guaranteed to be 1 always

    # we don't want a solution of either 0 or 1 because it causes problems for gradient calculation
    # so we stand off from 0 and 1 by tol
    bounds = []
    for i in range(cols):
        bounds.append((tol,1-tol))

    result = linprog(c,A_eq=M,b_eq=r,bounds=bounds)
    if result.success:
        return result.x

    # if nothing worked so far, try a random search
    if useRandom is True:
        if doWarnings is True:
            print('Using random!')

        (l,dim) = np.shape(Mk)
        absmins = np.array([np.min(np.abs(Mk[:,i])) for i in range(dim)])
        test_p0 = pinv(M) @ r
        maxIter = 1000000

        count = 0
        while ((np.any(test_p0 < 0)) & (count < maxIter)):
            bump = np.random.rand(dim) * (2/absmins) - 1/absmins
            test_p0 = np.array(pinv(M) @ r)
            for i in range(dim):
                test_p0 += bump[i] * Mk[:,i]
            count += 1

        if count < maxIter:
            return test_p0

    # nothing worked, probably the problem is infeasible
    if doWarnings is True:
        print('No solution!')
    return None


# get the gradient at point p with respect to kernel Mk
# p0 should be in the form given by getConstantDistortionDistributions, i.e., the vectorized (stacked) form
def getGradient(p0,Mk,ny,nx):
    # will need to check if we are within delta of the boundary
    # actually we can solve it explicitly, add a standoff to ensure we don't touch the boundary
    (l,dim) = np.shape(Mk)
    result = np.zeros(dim)

    # component of derivative for H(X,Y)
    for k in range(dim):
        # we are subtracting this so the negative sign disappears (subtracting a negative quantity)
        result[k] = np.sum((1 + np.log(p0))*Mk[:,k])

    # component of derivative for H(Y)
    # marginalization matrix
    Z = np.zeros((ny,len(p0)))
    for i in range(ny):
        Z[i,(i*nx):((i+1)*nx)] = np.ones(nx)

    for k in range(dim):
        # negative sign because entropy is negative
        result[k] -= np.sum((1+np.log(Z @ p0))*(Z @ Mk[:,k]))

    return result

# gets the solution using simple gradient descent with a fixed step
def gdRate(params,step=0.01,numIter=100000):
    p0 = params['p0']
    Mk = params['Mk']
    ny = params['ny']
    nx = params['nx']

    p0_current = np.array(p0) # copy
    (l,dim) = np.shape(Mk)
    lastMI = np.inf
    cgrad = np.zeros(dim)

    i = 0
    descending = True # true if we're still descending the gradient

    while (i < numIter) and descending is True:
        grad = getGradient(p0_current,Mk,ny,nx)
        for j in range(dim):
            p0_current = p0_current - step*grad[j]*Mk[:,j]

        if (MI(unstack(p0_current,ny,nx)) > lastMI):
            descending = False
        lastMI = MI(unstack(p0_current,ny,nx))
        i += 1
        cgrad += step*grad

    return (lastMI,p0_current,i,cgrad)


def getRD(px,Dxy,numPoints=100,show_pb=False):
    if show_pb == True:
        pb = ProgressBar(numPoints,40)

    (ny,nx) = np.shape(Dxy)
    d = stack(Dxy)

    r_v = np.array([])
    Dmax_v = np.array([])
    invalid_Dmax_v = np.array([])
    nan_Dmax_v = np.array([])
    p_v = []
    Dmax_try_v = np.linspace(np.min(d),np.max(d),numPoints)

    for Dmax in Dmax_try_v:
        params = getConstantDistortionDistributions(px,Dxy,Dmax)
        if params is not None:
            r = gdRate(params)
            if np.isnan(r[0]) == True:
                # if it's NaN, add it to the NaN pile
                nan_Dmax_v = np.append(nan_Dmax_v,Dmax)
            elif (len(r_v) == 0) or (r[0] <= r_v[-1]):
                r_v = np.append(r_v,r[0])
                Dmax_v = np.append(Dmax_v,Dmax)
                p_v.append(r[1])
        else:
            # if the response is None, add it to the None pile
            invalid_Dmax_v = np.append(invalid_Dmax_v,Dmax)
        if show_pb == True:
            pb.iterate()

    result = {}
    result['r_v'] = r_v
    result['Dmax_v'] = Dmax_v
    result['p'] = p_v
    result['invalid_Dmax_v'] = invalid_Dmax_v
    result['nan_Dmax_v'] = nan_Dmax_v

    if show_pb == True:
        pb.hide()

    return result

# gets the equivalent strategy matrix t from the reward matrix R and the actual strategy s
# R is a (number of phenotypes) x (number of outcomes) matrix
# s is a (cardinality of side information) x (number of phenotypes) matrix
# Dxy = s @ R gives the growth rate (actual, not log), where D[i,j] = reward for side info i and outcome j
# note: this is the same order as given at the top (but CHECK TO MAKE SURE!!)
def getT(R,s):
    (rows,cols) = np.shape(R)
    q = pinv(R) @ np.ones(rows)
    Qinv = np.diag(q)
    B = R @ Qinv
    return s @ B


def getDistortionFunction(R,s):
    return -1*np.log(getT(R,s))

# returns the xth diagonal element of Q
def getLambdaStarX(R,x=None):
    (rows,cols) = np.shape(R)
    if rows == cols:
      q = inv(R) @ np.ones(rows)
    else:
      q = pinv(R) @ np.ones(rows)
    Qinv = np.diag(q)
    Q = lg.inv(Qinv)
    if x is None:
        return np.log(np.diag(Q))
    return np.log(Q[x,x])
    #Qinv = np.diag((np.linalg.inv(R) @ (np.array([np.ones(cols)]).T))[:,0])
    #Q = np.linalg.inv(Qinv)
    #if x is None:
    #  return np.log(np.diag(Q))
    #return np.log(Q[x,x])

def getFullDistortionFunction(R,s):
    (rows,cols) = np.shape(R)
    Dxy = getDistortionFunction(R,s)
    for i in range(cols):
        Dxy[:,i] -= getLambdaStarX(R,i)
    return Dxy
