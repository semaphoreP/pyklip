#!/usr/bin/env python

import os
import glob
import warnings
import numpy as np
import astropy.io.fits as fits
import pyklip
import pyklip.klip as klip
import pyklip.empca as empca
import pytest
import sys
if sys.version_info < (3,3):
    import mock
    import unittest
else:
    import unittest
    import unittest.mock as mock

# this script contains unit tests for some functions in pyklip.klip and pyklip.empca

class klip_functions_TestCase(unittest.TestCase):

    '''
    tests for functions added to klip.py since the introduction of empca
    '''

    def test_make_polar_coordinates(self):

        x, y = np.meshgrid(np.arange(10), np.arange(10))
        x = np.reshape(x, (x.shape[0]*x.shape[1],), copy=False)
        y = np.reshape(y, (y.shape[0]*y.shape[1],), copy=False)

        # test for center at [0,0]
        center = [0, 0]
        r, phi = klip.make_polar_coordinates(x, y, center)
        ind = np.where((x==0) & (y==0))
        assert r[ind] == 0
        ind = np.where(y==0)
        testarray = np.zeros(10)
        testarray.fill(-np.pi)
        assert np.array_equal(phi[ind], testarray)

        # test for center at [5,5]
        center = [5, 5]
        r, phi = klip.make_polar_coordinates(x, y, center)
        ind = np.where((x==0) & (y==0))
        assert r[ind] == np.sqrt(50)
        ind = np.where((y>5) & (x==5))
        testarray = np.zeros(4)
        testarray.fill(-np.pi/2)
        assert np.array_equal(phi[ind], testarray)

    def test_median_collapse(self):
        test_cube = np.reshape(np.arange(9), (3, 3))
        weights = 2.
        ans = klip.collapse_data(test_cube, collapse_method='median')
        assert np.array_equal(ans, np.array([1., 4., 7.]))
        ans = klip.collapse_data(test_cube, weights, axis=0, collapse_method='median')
        assert np.array_equal(ans, np.array([3., 4., 5.]))

    def test_mean_collapse(self):
        test_cube = np.reshape(np.arange(9), (3, 3))
        weights = 2.
        ans = klip.collapse_data(test_cube, collapse_method='mean')
        assert np.array_equal(ans, np.array([1., 4., 7.]))
        ans = klip.collapse_data(test_cube, weights, axis=0, collapse_method='mean')
        assert np.array_equal(ans, np.array([3., 4., 5.]))

    def test_weighted_mean_collapse(self):
        test_cube = np.reshape(np.arange(9), (3, 3))
        weights = np.reshape(np.ones(9), (3, 3))
        ans = klip.collapse_data(test_cube, collapse_method='weighted_mean')
        assert np.array_equal(ans, np.array([1., 4., 7.]))
        ans = klip.collapse_data(test_cube, weights, axis=0, collapse_method='weighted-mean')
        assert np.array_equal(ans, np.array([3., 4., 5.]))
        weights = test_cube
        ans = klip.collapse_data(test_cube, weights, axis=1, collapse_method='weighted mean')
        assert np.array_equal(ans, (np.nanmean(test_cube * weights, axis=1) / np.nanmean(weights, axis=1)))

    def test_trimmed_mean_collapse(self):
        test_cube = np.array([[2., 1., 4., 0., 3.],
                              [5., 9., 7., 6., 8.]])
        ans = klip.collapse_data(test_cube, axis=1, collapse_method='Trimmed-mean')
        assert np.array_equal(ans, np.array([2, 7]))
        test_cube = np.array([[2., 1., 4., 0., 3., 5.],
                              [10., 9., 7., 6., 8., 11.]])
        ans = klip.collapse_data(test_cube, axis=1, collapse_method='trimmed_mean')
        assert np.array_equal(ans, np.array([2.5, 8.5]))

class empca_functions_TestCase(unittest.TestCase):

    '''
    tests for functions added to the original empca (mostly helper functions)
    '''

    def test_set_pixel_weights(self):
        # TODO: write test for this function after more weighting schemes are implemented
        pass