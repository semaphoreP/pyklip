import os
import glob
from time import time
import numpy as np
import pytest
import astropy.io.fits as fits

import pyklip.instruments.Instrument as Instrument
import pyklip.parallelized as parallelized
import pyklip.rdi as rdi

testdir = os.path.dirname(os.path.abspath(__file__)) + os.path.sep


def test_RDI():
    """
    Uses some ADI/SDI data to test RDI. Just makes sure it doesn't crash
    """

    filelist = glob.glob(testdir + os.path.join("data", "S20131210*distorcorr.fits"))
    filename = filelist[0]
    
    # just load in the first file
    numfiles = 1

    with fits.open(filename) as hdulist:
        inputdata = hdulist[1].data

    fakewvs = np.arange(37*numfiles, dtype=float) + 1
    fakepas = np.zeros(37*numfiles, dtype=float)
    fakecenters = np.array([[140,140] for _ in fakewvs])
    filenames = np.array([filename + str(i) for i in range(37*numfiles)])#np.repeat([filename], 37)

    dataset = Instrument.GenericData(inputdata[:1], fakecenters[:1], parangs=fakepas[:1], wvs=fakewvs[:1], filenames=filenames[:1])
    dataset.output_centers = dataset.centers
    dataset.output_wcs = dataset.wcs

    # psf library
    psflib = rdi.PSFLibrary(inputdata, fakecenters[1], filenames, compute_correlation=True)
    psflib.prepare_library(dataset)

    numbasis=[1,5,10,20,50] # number of KL basis vectors to use to model the PSF. We will try several different ones
    maxnumbasis=150 # maximum number of most correlated PSFs to do PCA reconstruction with
    annuli=3
    subsections=4 # break each annulus into 4 sectors
    parallelized.klip_dataset(dataset, outputdir=testdir, fileprefix="RDIonly-1file", annuli=annuli,
                            subsections=subsections, numbasis=numbasis, maxnumbasis=maxnumbasis, mode="RDI",
                            aligned_center=fakecenters[1], psf_library=psflib, movement=1)


def test_aligned_center_exception():
    """
    Passing in more than 1 center raises exception
    """
    filelist = glob.glob(testdir + os.path.join("data", "S20131210*distorcorr.fits"))
    filename = filelist[0]

    with fits.open(filename) as hdulist:
        inputdata = hdulist[1].data
    numfiles = 1
    fakecenters = np.array([[140,140] for _ in range(37*numfiles)])
    filenames = np.array([filename + str(i) for i in range(37*numfiles)])
    
    with pytest.raises(ValueError):
        psflib = rdi.PSFLibrary(inputdata, fakecenters, filenames, compute_correlation=True)