#!/usr/bin/env python
from astropy.io import fits
import numpy as np
from scipy import ndimage, signal, optimize, linalg, stats
import multiprocessing
import re
import glob
import os
import sys
import time


class Consumer(multiprocessing.Process):

    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means we should exit
                break
            self.result_queue.put(next_task())
        return


class Task(object):
    def __init__(self, index, func, args):
        self.index = index
        self.func = func
        self.args = args

    def __call__(self):
        return self.index, self.func(*self.args)


def _smooth(im, ivar, sig=1, spline_filter=False):
    '''
    Private function _smooth smooths an image accounting for the
    inverse variance.  It optionally spline filters the result to save
    time in later calls to ndimage.map_coordinates.

    Args:
        im: ndarray, 2D image to smooth
        ivar: ndarray, 2D inverse variance, shape should match im
        sig: float, standard deviation of Gaussian smoothing kernel, default 1
        spline_filter: boolean, spline filter the result?  Default False.

    Returns:
        imsmooth : ndarray, smoothed image of the same size as im
    '''

    if not isinstance(im, np.ndarray) or not isinstance(ivar, np.ndarray):
        raise TypeError("image and ivar passed to _smooth must be ndarrays")
    if im.shape != ivar.shape or len(im.shape) != 2:
        raise ValueError("image and ivar must be 2D ndarrays of the same shape")

    nx = int(sig * 4 + 1) * 2 + 1
    x = np.arange(nx) - nx // 2
    x, y = np.meshgrid(x, x)

    window = np.exp(-(x ** 2 + y ** 2) / (2 * sig ** 2))
    imsmooth = signal.convolve2d(im * ivar, window, mode='same')
    imsmooth /= signal.convolve2d(ivar, window, mode='same') + 1e-35
    imsmooth *= im != 0

    if spline_filter:
        imsmooth = ndimage.spline_filter(imsmooth)

    return imsmooth


def _spotloc(phi, sep, pitch=15, D=8.2, astrogrid='XYdiag'):
    '''
    Private function _spotloc computes the location of four
    satellite spots in units of lambda

    Args:
        phi: float, the angle of spots in radians
        sep: float, the separation of the spots in units of lambda/D
        pitch: float, lenslet pitch in units of milliarcseconds. Default 15
        D: float, telescope effective aperture in meters. Default 8.2
        astrogrid: astrogrid status read from the header, determines the pattern of the diffraction spots


    Returns:
        r : ndarray, array of (identical) separations in units of lenslets/microns
        phi : ndarray, array of spot angles in radians
    '''

    phi = np.arange(4) * np.pi / 2 + phi
    if astrogrid == 'Xdiag' or astrogrid == 'X':
        phi = np.take(phi, [1, 3])
    elif astrogrid == 'Ydiag' or astrogrid == 'Y':
        phi = np.take(phi, [0, 2])
    r = sep * 1e-6 / D * 3600 * 180 / np.pi / (pitch * 1e-3)
    r = r * np.ones(phi.shape)

    return r, phi


def _par_to_dx_dy(p, lam):
    '''
    Private function _par_to_dx_dy.  Use the parameters given by
    argument p (as a list), together with the wavelength array, to
    compute the offsets as a function of wavelength given as a
    second-order polynomial in wavelength.

    Args:
        p: list of floats, the coefficients of the polynomial fit to the centroid
        lam: wavelength(s), either as a number or as an array

    Returns:
        dx, dy : tuple of 1D arrays, each the same shape as lam, with the wavelength-dependent offsets

    '''

    order = (len(p) - 2) // 2
    dx = (lam * 0 + 1.) * p[0]
    dy = (lam * 0 + 1.) * p[1 + order]

    for i in range(1, order + 1):
        dx += p[i] * lam ** i
        dy += p[1 + i + order] * lam ** i

    return dx, dy


def _spotintens(p, cube, lam, astrogrid='XYdiag'):
    '''
    Private function _spotintens computes the negative of the sum of
    the intensity of the four satellite spots induced by the SCExAO DM
    in XYdiag mode.  It is intended to be passed to a minimization
    routine (to maximize sum of the intensities).

    Args:
        p: list of floats
           p[0] is the angle of spots in radians
           p[1] is the separation in lambda/D
           p[2] - p[-1] are the coefficients of the polynomial fit to the centroid
        cube: 3D ndarray, input data cube, assumed to be smoothed and spline filtered
        lam: 1D ndarray, wavelength array in microns corresponding to the first axis of cube
        astrogrid: astrogrid status read from the header, determines the pattern of the diffraction spots

    Returns:
        sumval : float, the negative of the sum of the spot intensities at the four locations
                 given by p, with integer multiples of pi/2 added to p[0].

    '''

    if not isinstance(cube, np.ndarray):
        raise TypeError("cube must be a 3-dimensional ndarray")
    if len(cube.shape) != 3:
        raise TypeError("cube must be a 3-dimensional ndarray")
    if len(cube) != len(lam):
        raise ValueError("Dimensions of lam and cube must match")

    r, phi = _spotloc(p[0], p[1], astrogrid=astrogrid)
    rcosphi = r * np.cos(phi)
    rsinphi = r * np.sin(phi)
    sumval = 0

    dx, dy = _par_to_dx_dy(p[2:], lam)

    for i in range(cube.shape[0]):
        x = rcosphi * lam[i] + cube.shape[2] // 2 + dx[i]
        y = rsinphi * lam[i] + cube.shape[1] // 2 + dy[i]

        sumval -= np.sum(ndimage.map_coordinates(cube[i], [y, x], prefilter=False))

    return sumval


def _resid(im1, im2):
    '''
    Private function _resid takes two images, and subtracts a constant
    and a scaled version of the second image from the first.  It
    returns the sum of the squared residuals.

    Args:
        im1: ndarray, image
        im2: ndarray, image of the same dimensions as im1

    Returns:
        chisq : float, sum of the squared residuals after subtracting the
                best fit constant+im2 model from im1

    '''

    if not isinstance(im1, np.ndarray) or not isinstance(im2, np.ndarray):
        raise TypeError("Images passed to _resid must be ndarrays.")
    if im1.shape != im2.shape:
        raise ValueError("Dimensions of images in _resid must match.")

    Sxx = np.sum(im2 ** 2)
    Sx = np.sum(im2)
    Sy = np.sum(im1)
    Sxy = np.sum(im1 * im2)
    S = im2.shape[0]
    norm = (S * Sxy - Sx * Sy) / (S * Sxx - Sx ** 2)
    zeropt = (Sxx * Sy - Sx * Sxy) / (S * Sxx - Sx ** 2)

    return np.sum((im1 - norm * im2 - zeropt) ** 2)


def _resid_xy(p, im1, im2, x, y):
    '''
    Private function _resid_xy computes the residual between im1 and
    im2, but with an offset applied to the positions of im2 before
    finding the residual.  The residual is only computed at certain
    points given by ndarrays x and y.  This function is intended to be
    passed to a minimization routine.

    The function calls map_coordinates to compute the reference and
    comparison images at the appropriate points, then calls _resid to
    fit the model and compute the residual.

    Args:
        p: list of two floats
           p[0] is the x offset between im1 and im2
           p[1] is the y offset
        im1: 2D ndarray, reference image, assumed to be spline_filtered
        im2: 2D ndarray, comparison image, assumed to be spline_filtered
        x: 2D ndarray, x coordinates of the points at which to fit im2 to im1
        y: 2D ndarray, y coordinates of the points at which to fit im2 to im1

    Returns:
        chisq : float, square of the residuals between im1 and a model made by
                translating im2 and sampling at x and y.

    '''

    dx, dy = [p[0], p[1]]
    ref_im = ndimage.map_coordinates(im1, [y, x], prefilter=False)
    compare_im = ndimage.map_coordinates(im2, [y + dy, x + dx], prefilter=False)

    return _resid(ref_im, compare_im)


def _cc_resid(p, pp, cube, lam, x, y, retarr=False):
    '''
    Private function _cc_resid cross-correlates wavelength slices of a
    data cube to a reference wavelength.  The output is the sum of the
    squares of the lenslet-by-lenslet differences between the
    reference and scaled/shifted slices (after optimizing the relative
    scale), or the scaled/shifted slices themselves (as a template).

    Args:
        p: list of floats, the coefficients of the polynomial fit to the centroid
        pp: list of floats, the coefficients of the reference polynomial fit to the
            centroid (used to keep the reference center in place when optimizing p)
        cube: 3D ndarray, data cube
        lam: 1D ndarray, wavelengths
        x: ndarray with x lenslet coordinates of the cube to use in the calculation
        y: ndarray with y lenslet coordinates of the cube to use in the calculation
        retarr: return the scaled/shifted template rather than the
                           cross-correlation score? Default False.

    Returns:
        If retarr is False (default), return the sum of the squared differences
        of the various wavelength channels with the reference channel.  If
        retarr is True, return the template image, the average of all of the
        scaled and shifted images at the input x and y coordinates.

    '''

    dx1, dy1 = _par_to_dx_dy(p, lam)
    ny, nx = [cube.shape[1], cube.shape[2]]

    # Centroid of the reference wavelength slice

    iref = cube.shape[0] // 2
    dx2, dy2 = _par_to_dx_dy(pp, lam[iref])
    xx = x + dx2
    yy = y + dy2

    if retarr:
        interparr = np.zeros(tuple([cube.shape[0]] + list(x.shape)))
    else:
        ref = ndimage.map_coordinates(cube[iref], [yy + ny // 2, xx + nx // 2], prefilter=False)
        chisq = 0

    for i in range(cube.shape[0]):

        _x = (xx - dx1[i]) * lam[i] / lam[iref] + dx1[i] + nx // 2
        _y = (yy - dy1[i]) * lam[i] / lam[iref] + dy1[i] + ny // 2

        compare = ndimage.map_coordinates(cube[i], [_y, _x], prefilter=False)
        if retarr:
            interparr[i] = compare
        else:
            chisq += _resid(ref, compare)

    if retarr:
        return np.mean(interparr, axis=0)
    else:
        return chisq


def _get_fids(prihdrs):
    '''
    Read the observation times to use as the independent variable for the polynomial fit.

    Args:
        prihdrs: primary headers of the dataset

    Returns:
        fids: the observation time of each cube in integer seconds, offset by the first exposure.
    '''

    fids = []
    mjd_found = True
    for prihdr in prihdrs:
        try:
            mjd = prihdr['mjd']
            # convert unit of days to unit of seconds.
            # and truncate to integer (not mathematically necesary)
            mjd = int(mjd * 24 * 3600)
            fids.append(mjd)
        except:
            print('mjd keyword not found in the header')
            mjd_found = False
            break

    if not mjd_found:
        # if coundn't find mjd keyword in all headers, use arange() to generate indices for the polynomial fit
        fids = np.arange(len(prihdrs))

    fids = np.array(fids)
    fids -= fids[0]

    return fids

def get_sats_satf(p, cube, lam, astrogrid='XYdiag'):
    '''
    retrieves the pixel locations of all four satellite spots at each wavelength,
    and the negative sum of the four spot intensities at each wavelength

    Args:
        p: list of floats, the coefficients of the polynomial fit to the centroid
        cube: ndarray, data cube for which the centroid is fitted
        lam: ndarray, wavelengths for the datacube
        astrogrid: astrogrid status read from the header, determines the pattern of the diffraction spots

    Returns:
        sats: pixel locations (in [x,y] format) of all four satellite spots at each wavelength, shape (wvs, 4, 2)
        satf: float, peak fluxes (interpolated pixel value) at the fitted spot locations
    '''

    if not isinstance(cube, np.ndarray):
        raise TypeError("data cube must be a 3-dimensional ndarray")
    if len(cube.shape) != 3:
        raise TypeError("data cube must be a 3-dimensional ndarray")
    if len(cube) != len(lam):
        raise ValueError("Dimensions of lam and cube must match")

    r, phi = _spotloc(p[0], p[1], astrogrid=astrogrid)
    rcosphi = r * np.cos(phi)
    rsinphi = r * np.sin(phi)

    dx, dy = _par_to_dx_dy(p[2:], lam)

    if astrogrid == 'Xdiag' or astrogrid == 'Ydiag' or astrogrid == 'X' or astrogrid == 'Y':
        spot_num = 2
    else:
        spot_num = 4

    sats = np.zeros((len(lam), spot_num, 2))
    satf = np.zeros((len(lam), spot_num))

    for i in range(cube.shape[0]):
        x = rcosphi * lam[i] + cube.shape[2] // 2 + dx[i]
        y = rsinphi * lam[i] + cube.shape[1] // 2 + dy[i]

        # note that for sats, indices are recorded as [x,y], not [y,x], to be consistent with pyklip's convention
        sats[i] = np.stack((x, y), axis=1)
        satf[i] = ndimage.map_coordinates(cube[i], [y, x], prefilter=False)

    return sats, satf


def recen(p, cube, lam, sats, satf, n=None, scale=False, head=None, outfile=None,
          mask=None, data_HDU1=True):
    '''
    Function recen recenters the input data according to the offset
    parameters given in the argument p.  Optionally scale the data by
    wavelength to undo the scaling of diffraction.  Return the
    recentered cube.

    Args:
        p: list of floats, the coefficients of the polynomial fit to the centroid
        cube: 3D ndarray, data cube
        lam: 1D array of wavelengths
        sats: fitted satellite spot indices before recentering
        satf: fitted satellite spot fluxes
        n : integer, spatial dimension of recentered cube.  Default original cube size
        scale: boolean, rescale by wavelength?  Default False
        head: fits header for output file.  Default None
        outfile: string or None, name of output file.  Default None
        mask: boolean lenslet mask
        data_HDU1: boolean, write data to HDU1 and leave HDU0 with no data?  Default True.

    Returns:
        3D ndarray, nlam x n x n, recentered (optionally scaled by wavelength) data cube.

    '''

    if n is None:
        n = cube.shape[1]

    ny, nx = [cube.shape[1], cube.shape[2]]
    dx, dy = _par_to_dx_dy(p[2:], lam)

    x = np.arange(n) - n // 2
    x, y = np.meshgrid(x, x)

    cencube = np.zeros((lam.shape[0], x.shape[0], x.shape[1]))
    for i in range(lam.shape[0]):
        if scale:
            _y = y * lam[i] / lam[len(lam) / 2] + dy[i] + ny // 2
            _x = x * lam[i] / lam[len(lam) / 2] + dx[i] + nx // 2
            print('fitted spots for scaled recen cubes not tested for correctness yet')
            sats[i, :] -= [nx // 2, ny // 2]
            sats[i, :, 1] = sats[i, :, 1] * lam[i] / lam[len(lam) / 2] + ny // 2 - dy[i]
            sats[i, :, 0] = sats[i, :, 0] * lam[i] / lam[len(lam) / 2] + nx // 2 - dx[i]
        else:
            _y = y + dy[i] + ny // 2
            _x = x + dx[i] + nx // 2
            sats[i, :, 1] -= dy[i]
            sats[i, :, 0] -= dx[i]

        cencube[i] = ndimage.map_coordinates(cube[i], [_y, _x], prefilter=False)

    if mask is not None:
        k = (mask.shape[0] - n) // 2
        for i in range(cencube.shape[0]):
            cencube[i] *= mask[k:-k, k:-k]

    if data_HDU1:
        out = fits.HDUList(fits.PrimaryHDU(None, head))
        out.append(fits.PrimaryHDU(cencube))
        out.append(fits.PrimaryHDU(np.ones(cencube.shape)))
    else:
        out = fits.HDUList(fits.PrimaryHDU(cencube, head))

    if outfile is not None:
        out.writeto(outfile, overwrite=True)
        savespot_hdr(sats, satf, outfile, hdu_num=int(data_HDU1))
    else:
        print('output directory not specified, recentered cubes not saved...')

    return cencube


def fitrelcen(image1, image2, x, y, method='Powell'):
    '''
    Function fitrelcen fits for the offset between two images without
    any wavelength dependence by calling _resid_xy, minimizing the sum
    of the squared differences between the images (after optimizing
    over the wavelength-dependent relative normalization).

    Args:
        image1: 2D ndarray, first image
        image2:2D ndarray, second image
        x: ndarray, x coordinates of pixels/lenslets to use
        y: ndarray, y coordinates of pixels/lenslets to use
        method : method passed to scipy.optimize.minimize. Default 'Powell'.

    Returns:
        xc, yc : two floating point numbers giving the best-fit offset between
                 image1 and image2

    '''

    xc, yc = optimize.minimize(_resid_xy, [0, 0],
                               (image1, image2, x, y), method=method).x

    return [xc, yc]


def fitcen(cube, ivar, lam, spotsep=None, guess_center_loc=None, i1=1, i2=-1, r1=15, r2=35, spot_dx=4,
           astrogrid='XYdiag', smooth=True):
    '''
    Function fitcen.  Fit for the center of a CHARIS data cube using
    the satellite spots by maximizing the agreement between scaled
    cube slices around the spot locations.  If no spot locations are
    provided, use only the diffraction pattern itself in an annulus
    around the image center.

    Args:
        cube: 3D ndarray, CHARIS data cube
        ivar: 3D ndarray, inverse variance of the CHARIS data cube
        lam: 1D ndarray, wavelengths in microns
        spotsep: float or None.  If float, separation of the satellite spots in units
                 of lambda/D.  If None, only use the diffraction pattern in an annulus
                 between r1 and r2.
        guess_center_loc: manually specify initial location of image center if necessary, in [x, y] format
        i1: int, first slice of the data cube to use.  Default 1 (skip slice 0)
        i2: int, high limit of slices to use.  Default -1 (skip last slice)
        r1: float, minimum separation from approximate center for the annulus of the
            diffraction pattern to use in centroiding.  Default 15
        r2: float, maximum separation from approximate center for the annulus of the
            diffraction pattern to use in centroiding.  Default 35
        spot_dx: float, radius around spot location to cut out in order to match the
                 spot location as a function of wavelength.  Default 4
        astrogrid: astrogrid status read from the header, determines the pattern of the diffraction spots
        smooth: whether to smooth image before fitting

    Returns:
        p : list of floats
            p[0] is the angle of spots in radians
            p[1] is the separation in lambda/D
            p[2] - p[-1] are the coefficients of the polynomial fit to the centroid

    '''

    ####################################################################
    # Lightly smooth the cube before starting.
    ####################################################################

    cubesmooth = np.copy(cube)
    mask = np.any(cubesmooth, axis=0) != 0
    iref = i1 + len(lam[i1:i2]) // 2

    for i in range(cubesmooth.shape[0]):
        if smooth:
            cubesmooth[i] = _smooth(cubesmooth[i], ivar[i], lam[i] / 3., spline_filter=False)
            cubesmooth[i] *= mask
        cubesmooth[i] = ndimage.spline_filter(cubesmooth[i])

    if spotsep is not None:
        if np.abs(spotsep - 15.9) < 1:
            phi = -18 * np.pi / 180
        elif np.abs(spotsep - 10) < 2:
            phi = 27 * np.pi / 180
        else:
            print("Must call fitcen with a valid separation for the satellite spots.")
            return None

    ####################################################################
    # Initial center using inner region of the cube (not the spots)
    ####################################################################


    x = np.arange(cube.shape[1]) - cube.shape[1] // 2
    x, y = np.meshgrid(x, x)
    indx = np.where((x ** 2 + y ** 2 > r1 ** 2) * (x ** 2 + y ** 2 < r2 ** 2))

    if guess_center_loc is None:
        xc0, yc0 = optimize.minimize(_cc_resid, [0, 0], ([0, 0], cubesmooth[i1:i2], lam[i1:i2], x[indx], y[indx], False),
                                     method='Powell').x
    else:
        xc0 = guess_center_loc[0] - cube.shape[2] // 2 # xc0 is 0th order dx
        yc0 = guess_center_loc[1] - cube.shape[1] // 2 # yc0 is 0th order dx

    if spotsep is None:
        return [xc0, yc0]

    p0 = [phi, spotsep, xc0, 0, yc0, 0]

    ####################################################################
    # Now estimate the center by fitting for the spot locations at
    # each wavelength
    ####################################################################

    p1 = optimize.minimize(_spotintens, p0, (cubesmooth[i1:i2], lam[i1:i2], astrogrid), method='Powell').x
    phi, spotsep = [p1[0], p1[1]]

    ####################################################################
    # Use the areas immediately around the spots to get a center, now
    # by cross-correlation.
    ####################################################################

    _r, _phi = _spotloc(phi, spotsep, astrogrid=astrogrid)
    rcosphi = _r * np.cos(_phi) * lam[iref]
    rsinphi = _r * np.sin(_phi) * lam[iref]

    ok = np.zeros(x.shape)
    for i in range(len(_phi)):
        ok += (rcosphi[i] - x) ** 2 + (rsinphi[i] - y) ** 2 < spot_dx ** 2
    indx = np.where(ok != 0)

    p2 = optimize.minimize(_cc_resid, p0[2:], (p1[2:], cubesmooth[i1:i2], lam[i1:i2], x[indx], y[indx]),
                           method='Powell').x

    ####################################################################
    # Pull out the region of the image corresponding to the average spots.
    ####################################################################

    avgspots = _cc_resid(p2, p2, cubesmooth[i1:i2], lam[i1:i2], x, y, retarr=True)

    lam0 = lam[iref]

    ####################################################################
    # Revise the center location one more time, use this new center
    # (derived from the spots) to shift the coefficients of the fit
    ####################################################################

    p3 = optimize.minimize(_spotintens, [phi, spotsep, 0, 0], (np.asarray([avgspots]), np.asarray([lam0]), astrogrid),
                           method='Powell').x

    ####################################################################
    # Use the center derived from the spots to shift the coefficients
    # of the fit
    ####################################################################

    # center:

    # x: (p2[0] + p3[2] + p2[1]*lam0 - p2[0] - p2[1]*lam)*lam/lam0 + p2[0] + p2[1]*lam
    # y: (p2[2] + p3[3] + p2[3]*lam0 - p2[2] - p2[3]*lam)*lam/lam0 + p2[2] + p2[3]*lam

    # x: p2[0] + (p3[2]/lam0 + 2*p2[1])*lam - (p2[1]/lam0)*lam**2
    # y: p2[2] + (p3[3]/lam0 + 2*p2[3])*lam - (p2[3]/lam0)*lam**2

    phi, spotsep = [p3[0], p3[1]]
    cen_coef = [p2[0], p3[2] / lam0 + 2 * p2[1], -p2[1] / lam0,
                p2[2], p3[3] / lam0 + 2 * p2[3], -p2[3] / lam0]

    return list([phi, spotsep]) + cen_coef


def fitcen_parallel(infiles, cubes, ivars, prihdrs, astrogrid_status=None, astrogrid_sep=None, smooth_coef=True,
                    guess_center_loc=None, smooth_cubes=True, maxcpus=multiprocessing.cpu_count() // 2):
    '''
    Function fitcen_parallel.  Centroid a series of CHARIS data cubes
    in parallel using fitcen.  By default, get the wavelengths and
    astrogrid parameters from the headers.  This might fail on early
    CHARIS data before the standardization of headers.

    Args:
        infiles: input files
        cubes: all image cubes in the data set corresponding to filenames, shape (ncube, nwv, ny, nx)
        ivars: inverse variance frames corresponding to cubes
        prihdrs: primary headers for the cubes
        astrogrid_status: None or list of astrogrid configurations for SCExAO.
                          If None, try to read the astrogrid configuration from the header.
                          If this fails, assume there is no astrogrid and centroid using the
                          general diffraction pattern.  Default None.
        astrogrid_sep: None or list of astrogrid spot separations in units of lambda/D.
                       If None, try to read from the header.
                       If that fails, centroid using the general diffraction pattern.
                       Default None.
        smooth_coef: boolean.  smooth the nonlinear coefficients of the centroid fit (the terms
                     proportional to lambda and lambda^2) over the sequence of cubes?
                     Default True.
        guess_center_loc: manually specify initial location of image center if necessary, in [x, y] format
        smooth_cubes: whether to smooth the data before fitting
        maxcpus: int, maximum number of CPUs to use in parallelization

    Returns:
        [centroid_params, x, y, mask]

        centroid_params : 2D array of centroid parameters, first dimension is the number of files.
                          Second dimension is the length of the wavelength-dependent model.
        x : x-coordinates of the centroid at the middle wavelength
        y : y-coordinates of the centroid at the middle wavelength
        mask : 1D boolean array, True if astrogrid was on, or None if the astrogrid was never on.

    '''

    ####################################################################
    # First try to load the astrogrid status and spot separations from
    # the FITS headers
    ####################################################################

    if astrogrid_status is None:
        astrogrid_status = []
        astrogrid_sep = []
        for i, head in enumerate(prihdrs):
            try:
                if head['X_GRDST'] != 'XYdiag' and head['X_GRDST'] != 'Xdiag' and head['X_GRDST'] != 'Ydiag'\
                   and head['X_GRDST'] != 'X' and head['X_GRDST'] != 'Y':
                    print('{}: astrogrid status {} is not recognized, default to XYdiag at 15.5 lambda/D spot '
                          'separation...'.format(os.path.basename(infiles[i]), head['X_GRDST']))
                    astrogrid_status += ['XYdiag']
                    astrogrid_sep += [15.5]
                else:
                    astrogrid_status += [head['X_GRDST']]
                    astrogrid_sep += [head['X_GRDSEP']]

            except:
                print('{}: error reading astrogrid status from header, default to XYdiag at 15.5 lambda/D spot '
                      'separation...'.format(os.path.basename(infiles[i])))
                astrogrid_status += ['XYdiag']
                astrogrid_sep += [15.5]

    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    ncpus = min(multiprocessing.cpu_count(), maxcpus)
    consumers = [Consumer(tasks, results)
                 for i in range(ncpus)]
    for w in consumers:
        w.start()

    fids = _get_fids(prihdrs)

    lamlist = []
    grid_on = np.zeros(len(cubes), int)

    ####################################################################
    # Load each file, get the wavelengths, and pass it to fitcen.
    ####################################################################

    for i in range(len(cubes)):

        cube = cubes[i]
        ivar = ivars[i]
        head = prihdrs[i]
        lam = head['lam_min'] * np.exp(np.arange(cube.shape[0]) * head['dloglam'])
        lam *= 1e-3 # in microns
        lamlist += [lam]

        if astrogrid_status[i] is None:
            spotsep = 15.5
        else:
            spotsep = astrogrid_sep[i]
            if not np.isscalar(spotsep):
                print('astrogrid information: header[X_GRDSEP] is not a scalar, default to using 15.5 lambda/D...')
                spotsep = 15.5
            grid_on[i] = 1

        tasks.put(Task(i, fitcen, (cube, ivar, lam, spotsep, guess_center_loc, 1, -1, 15, 35, 4, astrogrid_status[i],
                                   smooth_cubes)))

    for i in range(ncpus):
        tasks.put(None)

    centroid_params = None

    for fid in fids:
        index, result = results.get()
        if centroid_params is None:
            centroid_params = np.zeros((len(fids), len(result)))

        centroid_params[index] = result

    x = None
    y = None

    if np.sum(grid_on) == 0:
        mask = None
    else:
        mask = grid_on

    ####################################################################
    # If desired, smooth the nonlinear coefficients (in wavelength) of
    # the centroiding polynomial so that it varies smoothly with time.
    # Do not smooth the offsets, i.e., do not change the centers at a
    # given reference wavelength.  These can be refined later.
    ####################################################################

    if smooth_coef:

        x1 = polyfit(fids, centroid_params[:, 3], mask=mask, return_y=False)
        x2 = polyfit(fids, centroid_params[:, 4], mask=mask, return_y=False)
        y1 = polyfit(fids, centroid_params[:, 6], mask=mask, return_y=False)
        y2 = polyfit(fids, centroid_params[:, 7], mask=mask, return_y=False)
        x = np.zeros(centroid_params.shape[0])
        y = np.zeros(centroid_params.shape[0])

        for i in range(len(fids)):
            p = centroid_params[i]
            lamref = lamlist[i][len(lamlist[i]) // 2]
            dxref = p[3] * lamref + p[4] * lamref ** 2
            dyref = p[6] * lamref + p[7] * lamref ** 2
            dxnew = x1[i] * lamref + x2[i] * lamref ** 2
            dynew = y1[i] * lamref + y2[i] * lamref ** 2

            x[i] = dxref + p[2]
            y[i] = dyref + p[5]

            centroid_params[i, 2] += dxref - dxnew
            centroid_params[i, 5] += dyref - dynew

        centroid_params[:, 3] = x1
        centroid_params[:, 4] = x2
        centroid_params[:, 6] = y1
        centroid_params[:, 7] = y2

    return [centroid_params, x, y, mask]


def fitallrelcen(cubes, ivars, r1=15, r2=50, smooth=True, maxcpus=multiprocessing.cpu_count() // 2):
    '''
    Function fitallrelcen.  Fit for the relative centroids between all
    pairs of frames at the central wavelength using the PSF in an
    annulus around the center.  Return the best-fit relative offets.

    Args:
        cubes: all image cubes in the data set, shape (ncube, nwv, ny, nx)
        ivars: inverse variance frames corresponding to cubes
        r1: int, minimum separation in lenslets from the image center for annular reference region.  Default 15.
        r2: int, maximum separation in lenslets from the image center for annular reference region.  Default 50.
        smooth: whether to smooth the reference image before fitting
        maxcpus: int, maximum number of CPUs to allocate for parallelization. Default 1/2 of the available CPUs.

    Returns:
        xsol : 1D ndarray of the relative centers in x
        ysol : 1D ndarray of the relative centers in y

    '''

    ncpus = min(multiprocessing.cpu_count(), maxcpus)

    cubes = np.array(cubes)
    ivars = np.array(ivars)
    if len(cubes.shape) != 4:
        raise ValueError('input cubes have the wrong shape, expect (ncube, nwv, ny, nx), have {}'.format(cubes.shape))
    if cubes.shape != ivars.shape:
        raise ValueError('science data (cubes) and inverse variance data (ivars) have different shapes')
    ncube = cubes.shape[0]
    shape = cubes[0].shape
    iref = shape[0] // 2

    ####################################################################
    # Lightly smooth all images first.
    ####################################################################

    allims = np.zeros([cubes.shape[0], shape[1], shape[2]])

    for i in range(ncube):
        im = cubes[i, iref]
        ivar = ivars[i, iref]
        # TODO: smoothing has been moved to CHARIS.py._distortion_correction(), remove next line when things finalize
        if smooth:
            allims[i] = _smooth(im, ivar, 0.5, True)
        else:
            allims[i] = ndimage.spline_filter(im)

    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    consumers = [Consumer(tasks, results)
                 for k in range(ncpus)]
    for w in consumers:
        w.start()

    ####################################################################
    # Find the relative centers by cross-correlation, using only
    # lenslets >r1 and <r2 from the nominal center.
    ####################################################################

    ny, nx = [allims.shape[1], allims.shape[2]]
    x = np.arange(nx) - nx // 2
    x, y = np.meshgrid(x, x)
    r = np.sqrt(x ** 2 + y ** 2)
    x += nx // 2
    y += ny // 2

    ok = np.where((r > r1) * (r < r2))
    y = y[ok].copy()
    x = x[ok].copy()

    for i in range(ncube):
        for j in range(i + 1, ncube):
            index = i * (ncube) + j
            tasks.put(Task(index, fitrelcen, (allims[i], allims[j], x, y)))

    for k in range(ncpus):
        tasks.put(None)

    ####################################################################
    # Once we have all offsets between pairs of frames, solve for the
    # best-fit positions.  This is only defined up to a constant, so
    # also add the constraint that the mean position is zero.  We can
    # add an offset later based on the absolute centroiding routine.
    ####################################################################

    arr_x = np.zeros((ncube, ncube))
    arr_y = np.zeros((ncube, ncube))

    for i in range(ncube):
        for j in range(i + 1, ncube):
            index, result = results.get()
            fid2 = index % ncube
            fid1 = index // ncube

            arr_x[fid1, fid2] = result[0]
            arr_x[fid2, fid1] = -result[0]
            arr_y[fid1, fid2] = result[1]
            arr_y[fid2, fid1] = -result[1]

    b = np.zeros((ncube + 1))
    b[:ncube] = np.sum(arr_x, axis=0)
    arr = np.ones((ncube, ncube + 1))
    arr[:, :ncube] *= -1
    arr[:, :ncube] += ncube * np.identity(ncube)
    xsol = np.linalg.lstsq(arr.T, b, rcond=-1)[0]

    b[:ncube] = np.sum(arr_y, axis=0)
    ysol = np.linalg.lstsq(arr.T, b, rcond=-1)[0]

    return xsol, ysol


def polyfit(x, y, order=2, clip=2.5, niter=5, mask=None, return_y=True):
    '''
    Smooth a series of points with a polynomial, iteratively clipping
    outliers.

    Args:
        x: 1D ndarray of x coordinates for the polynomial fit
        y: 1D ndarray of y coordinates for the polynomial fit
        order: int, order of the polynomial to fit.  Default 2.
        clip: float, number of sigma outliers to clip.  Default 2.5.
        niter: int, number of iterations of sigma clipping.  Default 5.
        mask: boolean ndarray or None: mask each y value?  Default None
        return_y: boolean, return smoothed y values (as opposed to coefficients of the polynomial fit)?
                  Default True.

    Returns:
        y_smoothed : 1D array, if return_y=True
        coef : array of the polynomial coefficients if return_y=False

    '''

    if len(x) <= order and not return_y:
        return y

    arr = np.ones((len(x), order + 1))
    for i in range(1, order + 1):
        arr[:, i] = (x - np.median(x)) ** i

    ok = np.ones(y.shape)
    fit_vals = y.copy()

    for i in range(niter):

        _arr = arr.copy()
        if mask is not None:
            ok = ok * mask
        for j in range(order + 1):
            _arr[:, j] *= ok

        fitcoef = np.linalg.lstsq(_arr, y * ok, rcond=-1)[0]

        resid = y.copy()
        fit_vals[:] = 0
        for j in range(order + 1):
            resid -= fitcoef[j] * arr[:, j]
            fit_vals += fitcoef[j] * arr[:, j]

        std = np.sqrt(np.mean(resid[np.where(ok)] ** 2))
        ok = resid ** 2 < clip ** 2 * std ** 2

    if return_y:
        return fitcoef
    else:
        return fit_vals


def specphotcal(infiles, cubes, prihdrs, cencoef, aperture=1.):
    '''
    Function specphotcal.  Computes approximate photometry from the
    satellite spots (using aperture photometry) and scale each
    wavelength to this photometric value.  This should crudely put the
    cubes in units of contrast, though it omits the scaling of
    satellite spot intensity with 1/lambda^2.

    Args:
        infiles: list of file names with CHARIS data cubes
        cubes: all image cubes in the data set corresponding to filenames, shape (ncube, nwv, ny, nx)
        prihdrs: primary headers for the cubes
        cencoef: 2D ndarray with coefficeitns
        aperture: float, radius of aperture for photometry in units of lambda/D

    Returns:
        all_phot: photocalibration coefficients

    '''

    fids = _get_fids(prihdrs)

    phi = polyfit(fids, cencoef[:, 0], return_y=False)
    sep = polyfit(fids, cencoef[:, 1], return_y=False)

    all_phot = []
    all_x = []
    all_y = []

    for i in range(len(cubes)):

        im = cubes[i]
        head = prihdrs[i]
        astrogrid_status = head['X_GRDST']
        lam = head['lam_min'] * np.exp(np.arange(im.shape[0]) * head['dloglam'])
        lam *= 1e-3

        ny, nx = [im.shape[1], im.shape[2]]

        dx, dy = _par_to_dx_dy(cencoef[i, 2:], lam)
        phot = np.zeros(lam.shape)

        x = np.arange(nx)
        y = np.arange(ny)
        x, y = np.meshgrid(x, y)
        _r, _phi = _spotloc(phi[i], sep[i], astrogrid=astrogrid_status)

        # Manual offsets to account for the fact that two of the spots
        # lie atop spiders.  Check to see if the first spot is atop a
        # spider or not.
        # TODO: change these lines to accomadate different astrogrid setup
        if np.abs(_phi[0] % np.pi - 2.8) < np.pi / 4:
            _dphi = -np.pi / 2 + 12 * np.pi / 180 * np.asarray([1, -1, 1, -1])
        else:
            _dphi = -np.pi / 2 + 12 * np.pi / 180 * np.asarray([-1, 1, -1, 1])

        caltable = []

        for j in range(lam.shape[0]):

            xcoord = np.cos(_phi) * _r * lam[j] + dx[j] + nx // 2
            ycoord = np.sin(_phi) * _r * lam[j] + dy[j] + ny // 2
            xref = np.cos(_phi + _dphi) * _r * lam[j] + dx[j] + nx // 2
            yref = np.sin(_phi + _dphi) * _r * lam[j] + dy[j] + ny // 2

            # CHARIS is approximately critically sampled at 1.15 microns
            radius = 2 * aperture * (lam[j] / 1.15)


            for k in range(len(xcoord)):
                dist = np.sqrt((x - xcoord[k]) ** 2 + (y - ycoord[k]) ** 2)
                distref = np.sqrt((x - xref[k]) ** 2 + (y - yref[k]) ** 2)

                ref = np.sum(im[j][distref < radius])
                phot[j] += np.sum(im[j][dist < radius]) - ref

            caltable += [(dx[j] + nx // 2, dy[j] + ny // 2, phot[j])]
        all_phot += [phot]
        all_x += [dx + nx // 2]
        all_y += [dy + ny // 2]

    return all_phot