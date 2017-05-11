import os
import glob
import numpy as np
import astropy.io.fits as fits
import pyklip.instruments.Instrument as Instrument

testdir = os.path.dirname(os.path.abspath(__file__)) + os.path.sep

def test_generic_dataset():
    """
    Tests the generic dataset interface into pyklip using some GPI data

    Just makes sure it doesn't crash
    """

    filelist = glob.glob(testdir + os.path.join("data", "S20131210*distorcorr.fits"))
    filename = filelist[0]
    
    numfiles = 1

    hdulist = fits.open(filename)
    inputdata = hdulist[1].data

    fakewvs = np.arange(37*numfiles)
    fakepas = np.arange(37*numfiles)
    fakecenters = np.array([[140,140] for _ in fakewvs])
    filenames = np.repeat([filename], 37)

    dataset = Instrument.GenericData(inputdata, fakecenters, parangs=fakepas, wvs=fakewvs, filenames=filenames)

    dataset.savedata(os.path.join(testdir, "generic_dataset.fits"), dataset.input)
    # it didn't crash? Good enough

