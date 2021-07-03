"""
The `samplers` module contains a set of functions for generating samples from
a *random i.i.d sample*. The output random variates should have similar 
distribution to the input sample.

The aim is different from bootstrapping, where we resample from the same
values. Here, we will obtain from a smoother distribution.

These are the functions that are present in this module:

rand_bartlett():
    Generate random variates from Bartlett kernel. (Helper)

rand_from_density():
    Generate new sample from a kernel density estimate.

rand_from_hist():
    Generate a new sample from a histogram estimate.

rand_from_Finv():
    Generate a new sample from an estimate of F-inverse.
"""

import numpy as np
from scipy.interpolate import Akima1DInterpolator
from statsmodels.nonparametric import bandwidths
from statsmodels.distributions.empirical_distribution import ECDF

def rand_bartlett(rng, size=(1,10)):
    """
    Returns random variates from the Bartlett kernel, which is given by 
    f(x) = 3/4(1 - x^2) for -1 < x < 1, and 0 otherwise. In other literature,
    this is also referred to as the Epanechnikov kernel.

    In this function, generation is carried out using the rejection method. The 
    acceptance rate is about 2/3 - that's why we see a 1.5 in the algo.

    Parameters
    ----------
    rng  : random number generator. This should be the one from numpy.
    size : The array of random number variates to return.

    >>> rng = np.random.default_rng(123)
    >>> rand_bartlett(rng, size=1)
    array([0.36470373])

    """
    rvs_needed = np.array(size).prod()
    rvs_generated = 0
    out = np.empty(0)
    
    while rvs_generated < rvs_needed:
        X = rng.uniform(-1.0, 1.0, size=round(rvs_needed*1.5, None))
        U = rng.uniform(0.0, 1.0, size = X.shape)
        out = np.hstack((out, X[U <= 1 - X**2]))
        
        rvs_generated = out.size
        
    return out[:rvs_needed].reshape(size)   

def rand_from_density(X, rng, size=(1,10), lower=-np.inf, 
    upper=np.inf, batch_size=100):
    """
    Returns a sample using the density estimate fitted to the original sample
    of values.

    The kernel used is Bartlett's kernel. The latter is sampled from using 
    `rand_bartlett()`. Scott's rule of thumb is used for the bandwidth.

    Density estimates sometimes do not respect the support of the original
    dataset. This is the reason for the two arguments to truncate the density
    estimate. If one of them is used, then the density is sampled from until
    the all values fall within the boundary.

    Parameters
    ----------
    X       : The original dataset, as a 1-D numpy array.
    rng     : random number generator. This should be the one from numpy.
    size    : The array of random number variates to return.
    lower   : The lower limit of the original dataset.
    upper   : The upper limit of the original dataset.
    batch_size: If sampling has to be carried out until the limits are
                respected, then this is the batch size of samples that is 
                used. This is probably more efficient than simulating one 
                at a time.

    >>> rng = np.random.default_rng(123)
    >>> X = rng.chisquare(1, 100)
    >>> Xp = rand_from_density(X, rng, size=100, lower=0.0)
    >>> Xpp = rand_from_density(X, rng, size=100)
    >>> Xp.min()  # positive
    0.002818632591977721
    >>> Xpp.min() # violates 0 lower limit
    -0.24301177299527474

    """

    rvs_needed = np.array(size).prod()
    bw0 = bandwidths.bw_scott(X)
    
    if (lower is not -np.inf) or (upper is not np.inf):
        rvs_generated = 0
        out = np.empty(0)
        
        while rvs_generated < rvs_needed:
            Y0 = rng.choice(X, size=batch_size) + \
            rand_bartlett(rng, batch_size)*bw0

            out = np.hstack((out, Y0[np.column_stack((Y0 >= lower,  \
            Y0 <= upper)).all(axis=1)])) 

            rvs_generated = out.size
        
        out = out[:rvs_needed].reshape(size)
    else:
        out = rng.choice(X, size=rvs_needed) + \
        rand_bartlett(rng, rvs_needed)*bw0

    return out


def rand_from_hist(X, rng, size=(1,10)):
    """
    Returns a sample using the histogram fitted to the original sample
    of values.

    Freedman-Diaconis rule is used to estimate the bin widths.

    This is a straightforward method, using the uniform distribution within
    bins to simulate random variates. There is no possibility of support 
    violation.

    Parameters
    ----------
    X       : The original dataset, as a 1-D numpy array.
    rng     : random number generator. This should be the one from numpy.
    size    : The array of random number variates to return.

    >>> rng = np.random.default_rng(123)
    >>> X = rng.exponential(2.0, size=100)
    >>> X_hist = rand_from_hist(X, rng, size=100)

    """
    rvs_needed = np.array(size).prod()
    hist_details = np.histogram(X, bins='fd')
    nbins = len(hist_details[0])
    
    bin_choices = rng.choice(nbins, size=rvs_needed, 
    p=hist_details[0]/len(X))
    out = rng.uniform(hist_details[1][bin_choices],
    hist_details[1][bin_choices+1])
    
    return out

def rand_from_Finv(X, rng, size=(1,10), Xmin=None, Xmax = None, return_fn=False):
    """
    Returns a sample using the interpolated F-inverse function.

    The Akima algorithm is used for interpolating the F-inverse function.

    If `return_fn` is True, then the F-inverse function is returned, so that it
    can be re-used.

    Parameters
    ----------
    X       : The original dataset, as a 1-D numpy array.
    rng     : random number generator. This should be the one from numpy.
    size    : The array of random number variates to return.
    Xmin    : The minimum of the support of X. If not provided, the minimum
              observed X is used.
    Xmax    : The minimum of the support of X. If not provided, the maximum 
              observed X is used.
    return_fn: A logical value. It determines if the F-inverse function is
               returned. If True, then size argument is ignored. If False,
               A numpy array is returned using the interpolated F-inverse.

    >>> rng = np.random.default_rng(123)
    >>> X = rng.exponential(2.2, size=100)
    >>> X_from_F = rand_from_Finv(X, rng, size=200)
    >>> X_from_F[:5]
    array([3.55652952, 0.31075643, 0.87066671, 1.87087861, 5.23183298])
    >>> Finv = rand_from_Finv(X, rng, return_fn=True)
    >>> Finv(rng.uniform(size=3))
    array([1.52988127, 1.0962427 , 0.86032202])

    """
    rvs_needed = np.array(size).prod()
    ecdf1 = ECDF(X)
    U = np.hstack((0.0, ecdf1.y[1:-1], 1.0))
    
    if Xmin is None:
        Xmin = X.min()
    if Xmax is None:
        Xmax = X.max()
        
    Finv = np.hstack((Xmin, ecdf1.x[1:-1], Xmax))
    ak2 = Akima1DInterpolator(U, Finv)
    if return_fn:
        return ak2

    U_rand = rng.uniform(size=rvs_needed)
    out = ak2(U_rand).reshape(size)
    return out
