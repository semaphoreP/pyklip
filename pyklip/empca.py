#!/usr/bin/env python

import numpy as np
import numpy.fft as fft
import scipy.linalg as la
import scipy.ndimage as ndimage
import scipy.interpolate as sinterp
import scipy.signal as signal
from scipy.stats import t
import sys
import multiprocessing
import time

"""
Weighted Principal Component Analysis using Expectation Maximization

Original: Stephen Bailey, Spring 2012
Rewritten by Timothy Brandt, Spring 2016
"""

def np_calc_chisq(data, b, w, coef):
    """
    Calculate chi squared

    Args:
        im: nim x npix, single-precision numpy.ndarray. Data to be fit by the basis images
        b: nvec x npts, double precision numpy.ndarray. The nvec basis images.
        w: nim x npts, single-precision numpy.ndarray. Weights (inverse variances) of the data.
        coef: nvec x npts, double precision numpy.ndarray. The coefficients of the basis image fits.

    Returns:
        chisq, the total chi squared summed over all points and all images
    """

    chisq = 0
    nim = data.shape[0]
    for i in range(nim):
        chisq += np.sum((data[i] - np.sum(coef[i] * b.T, axis=1)) ** 2 * w[i])

    return chisq

def set_pixel_weights(imflat, rflat, ivar=None, mode='standard', inner_sup=17, outer_sup=66):
    '''
    Args:
        imflat: array of flattend images, shape (N, number of section indices)
        rflat: radial component of the polar coordinates flattened to 1D, length = number of section indices
        mode:
            'standard': assume poission statistics to calculate variance as sqrt(photon count)
                        use inverse sqrt(variance) as pixel weights and multiply by a radial weighting
        inner_sup: radius within which to supress weights
        outer_sup: radius beyond which to supress weights

    Returns:
        pixel weights for empca

    '''

    #default weights are ones
    weights = np.ones(imflat.shape)

    if mode.lower() == 'standard':
        # this is simply using sqrt(pixel value) as the standard deviation, hence inverse weights
        weights = 1. / (np.sqrt(np.abs(imflat)) + 10)
        weights *= imflat != 0
        # suppress contribution from pixels beyond inner and outer working angles
        weights *= 1 / (1 + np.exp((inner_sup - rflat) / 1.))
        weights *= 1 / (1 + np.exp((rflat - outer_sup) / 1.))

    if mode.lower() == 'standard_ivar':
        #TODO: add ivar to CHARISData class and implement this method
        pass

    return weights

def _random_orthonormal(nvec, nvar, seed=1):
    '''
    Generate random orthonormal vectors as initial guess model
    Doesn't protect against rare duplicate vectors leading to 0s

    Args:
        nvec: rank of model
        nvar: number of parameters (e.g. number of pixels in an image for psf fitting)
        seed:

    Returns:
        array of random orthonormal vectors A[nvec, nvar]
    '''

    if seed is not None:
        np.random.seed(seed)
        
    A = np.random.normal(size=(nvec, nvar))
    for i in range(nvec):
        A[i] /= np.linalg.norm(A[i])

    for i in range(1, nvec):
        for j in range(0, i):
            A[i] -= np.dot(A[j], A[i])*A[j]
            A[i] /= np.linalg.norm(A[i])

    if np.any(np.isnan(A)):
        raise ValueError("random orthonormal is nan")
    return A

def weighted_empca(data, weights=None, niter=25, nvec=5, randseed=1, maxcpus=1, silent=True):
    '''
    Perform iterative low-rank matrix approximation of data using weights.

    Generated model vectors are not orthonormal and are not
    rotated/ranked by ability to model the data, but as a set
    they are good at describing the data.

    Args:
        data: images to model
        weights: weights for every pixel
        niter: maximum number of iterations to perform
        nvec: number of vectors to solve (rank of the approximation)
        randseed: rand num generator seed; if None, don't re-initialize
        maxcpus: maximum cpus to use for parallel programming
        silent: bool, whether to show chi_squared for each iteration

    Returns:
        returns the best low-rank approximation to the data in a weighted
        least-squares sense (dot product of coefficients and basis vectors).
    '''

    if weights is None:
        weights = np.ones(data.shape, float)

    ##################################################################
    # The following code makes sure that there are two copies each 
    # of data and weights, one in C format (last axis fast) and one 
    # in Fortran format (first axis fast).  This costs a factor of
    # two in memory usage but speeds up access later.
    ##################################################################

    if not (isinstance(data, np.ndarray) and isinstance(weights, np.ndarray)):
        raise TypeError("'data' and 'weights' must be numpy ndarrays.")
    if not (data.shape == weights.shape and len(data.shape) == 2):
        raise ValueError("'data' and 'weights' must be 2D arrays of the same shape.")
   
    if data.flags['C']:
        dataC = data.astype(float)
        dataF = (dataC.T).copy(order='C')
    elif data.flags['F']:
        dataC = data.copy(order='C').astype(float)
        dataF = data.T.astype(float)
    else:
        raise AttributeError("Attribute 'flags' missing from data.")
    
    if weights.flags['C']:
        weightsC = weights.astype(float)
        weightsF = (weightsC.T).copy(order='C')
    elif weights.flags['F']:
        weightsC = weights.copy(order='C').astype(float)
        weightsF = weights.T.astype(float)
    else:
        raise AttributeError("Attribute 'flags' missing from weights.")

    ##################################################################
    # Random initial guess for the low-rank approximation, zero
    # for the initial fit/approximation coefficients.
    ##################################################################

    nobs, nvar = data.shape
    P = _random_orthonormal(nvec, nvar, seed=randseed)
    C = np.zeros((nobs, nvec))

    if not silent:
        print('iter     dchi2      R2          time (s)')

    ncpus = multiprocessing.cpu_count()
    if maxcpus is not None:
        ncpus = min(ncpus, maxcpus)

    chisq_orig = np_calc_chisq(dataC, P*0, weightsC, C)
    chisq_last = chisq_orig
    datwgt = dataC*weightsC

    singular_matrix = 0
    for itr in range(1, niter + 1):

        tstart = time.time()
        ##############################################################
        # Solve for best-fit coefficients with the previous/first
        # low-rank approximation.
        ##############################################################

        P3D = np.empty((P.shape[0], P.shape[0], P.shape[1]))
        for i in range(P.shape[0]):
            P3D[i] = P*P[i]
        A = np.tensordot(weights, P3D.T, axes=1)
        b = np.dot(datwgt, P.T)
        
        try:
            C = np.linalg.solve(A, b).T
        except:
            singular_matrix += 1
            Ainv = np.linalg.pinv(A)
            C = np.einsum('nmp,np->nm', Ainv, b).T
            
        ##############################################################
        # Compute the weighted residual (chi squared) value from the
        # previous fit.
        ##############################################################

        if not silent:

            chisq = np_calc_chisq(dataC, P, weightsC, C.T)
            print('%3d  %9.3g  %12.6f %11.3f' % (itr, chisq - chisq_last, 1 - chisq / chisq_orig, time.time() - tstart))
            chisq_last = chisq

        if itr == niter:

            ##########################################################
            # Compute the low-rank approximation to the data.
            ##########################################################

            model = np.dot(C.T, P)

        else:

            ##########################################################
            # Update the low-rank approximation.
            ##########################################################
            C3D = np.empty((C.shape[0], C.shape[0], C.shape[1]))
            for i in range(C.shape[0]):
                C3D[i] = C*C[i]
            A = np.tensordot(weights.T, C3D.T, axes=1)
            b = np.dot(datwgt.T, C.T)

            try:
                P = np.linalg.solve(A, b).T
            except:
                singular_matrix += 1
                Ainv = np.linalg.pinv(A)
                P = np.einsum('nmp,np->nm', Ainv, b).T

    ##################################################################
    # Normalize the low-rank approximation.
    ##################################################################

    for k in range(nvec):
        P[k] /= np.linalg.norm(P[k])

    if singular_matrix > 0:
        print('number of singular matrices encountered:{}'.format(singular_matrix))

    return model