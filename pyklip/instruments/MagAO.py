import os
import re
import subprocess
import glob
import astropy.io.fits as fits
import astropy.wcs as wcs
from astropy.modeling import models, fitting
import numpy as np
import scipy.ndimage as ndimage
import pyklip
from pyklip.parallelized import high_pass_filter_imgs
import scipy.stats
import multiprocessing as mp

import sys
from copy import copy

#different imports depending on if python2.7 or python3
if sys.version_info < (3,0):
    #python 2.7 behavior
    import ConfigParser
    from pyklip.instruments.Instrument import Data
    from pyklip.instruments.utils.nair import nMathar
else:
    import configparser as ConfigParser
    from pyklip.instruments.Instrument import Data
    from pyklip.instruments.utils.nair import nMathar


#from pyklip.instruments.P1640_support import P1640spots
#from pyklip.instruments.P1640_support import P1640utils
#from pyklip.instruments.P1640_support import P1640_cube_checker

from scipy.interpolate import interp1d

'''
#IMPORTANT NOTE:

Things that /must/ be in the headers of your sliced images:
- ROTOFF
- WLENGTH (wavelength band, in microns)
- GHSTPEAK or STARPEAK
- RA
- DEC


'''

class MagAOData(object):
    
    """
    A sequence of P1640 Data. Each P1640Data object has the following fields and functions 
    Args:
        filepaths: list of filepaths to files
        skipslices: a list of datacube slices to skip (supply index numbers e.g. [0,1,2,3])
        corefilepaths: a list of filepaths to core (i.e. unocculted) files, for contrast calc
        spot_directory: (None) path to the directory where the spot positions are stored. Defaults to P1640.ini val
    Attributes:
        input: Array of shape (N,y,x) for N images of shape (y,x)
        centers: Array of shape (N,2) for N centers in the format [x_cent, y_cent]
        filenums: Array of size N for the numerical index to map data to file that was passed in
        filenames: Array of size N for the actual filepath of the file that corresponds to the data
        PAs: Array of N for the parallactic angle rotation of the target (used for ADI) [in degrees]
        wvs: Array of N wavelengths of the images (used for SDI) [in microns]. For polarization data, defaults to "None"
        wcs: Array of N wcs astormetry headers for each image.
        IWA: a floating point scalar (not array). Specifies to inner working angle in pixels
        output: Array of shape (b, len(files), len(uniq_wvs), y, x) where b is the number of different KL basis cutoffs
        spot_flux: Array of N of average satellite spot flux for each frame
        contrast_scaling: Flux calibration factors (multiply by image to "calibrate" flux)
        flux_units: units of output data [DN, contrast]
        prihdrs: not used for P1640, set to None
        exthdrs: Array of N P1640 headers (these are written by the P1640 cube extraction pipeline)
    Methods:
        readdata(): reread in the data
        savedata(): save a specified data in the P1640 datacube format (in the 1st extension header)
        calibrate_output(): calibrates flux of self.output
    """

    ##########################
   ### Class Initialization ###
    ##########################
    #Some static variables to define the MagAO instrument
    centralwave = {} #in microns
    flux_zeropt = {}
    lenslet_scale = 1.0 #arcseconds per pixel (pixel scale)
    ifs_rotation = 0.0 #degrees CCW from +x axis to zenith
    observatory_latitude = 0.0
    ghstpeak_ratio = {}

    #read in MagAO configuration file and set these static variables
    package_directory = os.path.dirname(os.path.abspath(__file__))
    configfile = package_directory + "/" + "MagAO.ini"
    config = ConfigParser.ConfigParser()
    try:
        config.read(configfile)
        #get pixel scale
        lenslet_scale = float(config.get("instrument", "ifs_lenslet_scale")) #!
        #get IFS rotation
        ifs_rotation = float(config.get("instrument", "ifs_rotation"))
        bands = ['HA', 'CONT', 'z\'', 'r\'','i\'','Ys']
        for band in bands:
            centralwave[band] = float(config.get("instrument", "cen_wave_{0}".format(band)))
            flux_zeropt[band] = float(config.get("instrument", "zero_pt_flux_{0}".format(band)))
        observatory_latitude = float(config.get("observatory", "observatory_lat"))

        ghstpeak_ratio['line\''] = float(config.get("instrument", "ghost_ratio_HA"))
        ghstpeak_ratio['cont\''] = float(config.get("instrument", "ghost_ratio_CONT"))
        ghstpeak_ratio['z\''] = float(config.get("instrument", "ghost_ratio_z\'"))
        ghstpeak_ratio['i\''] = float(config.get("instrument", "ghost_ratio_i\'"))
    except ConfigParser.Error as e:
        print("Error reading MagAO configuration file: {0}".format(e.message))
        raise e
    
    #########################
   ###    Constructors     ###
    #########################
    def __init__(self, filepaths=None, highpass=False):
        """
        Initialization code for MagAOData
        """
        super(MagAOData, self).__init__()
        self._output = None
        if filepaths is None:
            #this won't get called if we run it normally so don't worry about it
            self._input = None
            self._centers = None
            self._filenums = None
            self._filenames = None
            self._PAs = None
            self._wvs = None
            self._wcs = None
            self._IWA = None
            self._OWA = None
            self.star_flux = None
            self.contrast_scaling = None
            self.prihdrs = None
            self._flipx = None
        else:
            self.readdata(filepaths, highpass=highpass)
    
    ##############################
   ### Instance Required Fields ###
    ##############################
    @property
    def input(self):
        return self._input
    @input.setter
    def input(self, newval):
        self._input = newval
    
    @property
    def centers(self):
        return self._centers
    @centers.setter
    def centers(self, newval):
        self._centers = newval

    @property
    def PAs(self):
        return self._PAs
    @PAs.setter
    def PAs(self, newval):
        self._PAs = newval
    
    @property
    def wvs(self):
        return self._wvs
    @wvs.setter
    def wvs(self, newval):
        self._wvs = newval
    
    @property
    def wcs(self):
        return self._wcs
    @wcs.setter
    def wcs(self, newval):
        self._wcs = newval

    @property
    def filenums(self):
        return self._filenums

    @property
    def IWA(self):
        return self._IWA
    @IWA.setter
    def IWA(self, newval):
        self._IWA = newval

    @property
    def OWA(self):
        return self._OWA
    @OWA.setter
    def OWA(self, newval):
        self._OWA = newval
    
    @property
    def output(self):
        return self._output
    @output.setter
    def output(self, newval):
        self._output = newval

    @property
    def flipx(self):
        return self._flipx
    @flipx.setter
    def flipx(self, newval):
        self._flipx = newval

    ###################
   ###    Methods    ###
    ###################
        
    def readdata(self, filepaths, highpass=False):
        """
        Method to open and read a list of MagAO data
        highpass: if True, run a Gaussian high pass filter (default size is sigma=imgsize/10)
                      can also be a number specifying FWHM of box in pixel units
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        print('reading data, num files: ',len(filepaths))
        data = []
        filenums = []
        filenames = []
        rot_angles = []
        wcs_hdrs = []
        wvs = []
        centers = []
        star_fluxes = []
        prihdrs = []

        # Create a threadpool for high pass filter
        pool = mp.Pool()

        for index, filepath in enumerate(filepaths):
            cube, center, pa, wv, astr_hdrs, prihdr, star_flux = _magao_process_file(self, filepath, highpass=highpass)
            data.append(cube)
            centers.append(center)
            star_fluxes.append(star_flux)
            rot_angles.append(pa)
            wvs.append(wv)
            wcs_hdrs.append(astr_hdrs)
            filenums.append([index])
            prihdrs.append(prihdr)
            filenames.append([filepath])

        # Close threadpool
        pool.close()
        pool.join()
            
        #FILENUMS IS 1D LIST
        data = np.array(data)
        dims = data.shape #should be 3D (#images by 451 by 451 for us, but depending on chosen size)
        filenums = np.array(filenums).reshape([dims[0]])
        filenames = np.array(filenames).reshape([dims[0]])
        rot_angles = np.array(rot_angles).reshape([dims[0]])
        wvs = np.array(wvs).reshape([dims[0]])
        wcs_hdrs = np.array(wcs_hdrs)
        dsize = dims[0]
        centers = np.zeros((dsize,2))
        for y in range(dsize):
            for x in range(2):
                centers[y][x] = (dims[1]-1)/2
        star_fluxes = np.array(star_fluxes)
 #       wcs_temp = [None]*dsize

        self._input = data
        self._centers = centers
        self._filenums = filenums
        self._filenames = filenames
        self._PAs = rot_angles
        self._wvs = wvs
        self._wcs = wcs_hdrs
        #IWA gets reset by GUI. This is the default value.
        self.IWA = 0
        # half the size of the array
        self.OWA = data.shape[1]/2
        #CHECK IWA AND OWA
        self.star_flux = star_fluxes
        self.contrast_scaling = 1./star_fluxes
        self.prihdrs = prihdrs
        self._flipx = False
        self.numwvs = np.size(np.unique(self.wvs))

    ##not currently used but will when we apply flux conversion
    def calibrate_output(self, img, spectral=False, units="contrast"):
        """
       Calibrates the flux of an output image. Can either be a broadband image or a spectral cube depending
        on if the spectral flag is set.

        Assumes the broadband flux calibration is just multiplication by a single scalar number whereas spectral
        datacubes may have a separate calibration value for each wavelength

        Args:
            img: unclaibrated image.
                 If spectral is not set, this can either be a 2-D or 3-D broadband image
                 where the last two dimensions are [y,x]
                 If specetral is True, this is a 3-D spectral cube with shape [wv,y,x]
            spectral: if True, this is a spectral datacube. Otherwise, it is a broadband image.
            units: currently only support "contrast" w.r.t central star

        Return:
            img: calibrated image of the same shape (this is the same object as the input!!!)
        """
        if units == "contrast":
            if spectral:
                # spectral cube, each slice needs it's own calibration
                numwvs = img.shape[0]
                img *= self.contrast_scaling[:numwvs, None, None]
            else:
                # broadband image
                img *= np.nanmean(self.contrast_scaling)

        return img
        
    def savedata(self, filepath, data, klipparams = None, filetype = None, zaxis = None, center=None, astr_hdr=None,
                 fakePlparams = None, more_keywords=None):
        """
        Save data in a GPI-like fashion. Aka, data and header are in the first extension header
        
        Inputs:
        filepath: path to file to output
        data: 2D or 3D data to save
        klipparams: a string of klip parameters
        filetype: filetype of the object (e.g. "KL Mode Cube", "PSF Subtracted Spectral Cube")
        zaxis: a list of values for the zaxis of the datacub (for KL mode cubes currently)
        astr_hdr: wcs astrometry header (None for NIRC2)
        center: center of the image to be saved in the header as the keywords PSFCENTX and PSFCENTY in pixels.
        The first pixel has coordinates (0,0)
        fakePlparams: fake planet params
        
        """
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(header=self.prihdrs[0]))
        hdulist.append(fits.ImageHDU(data=data, name="Sci"))
        
        # save all the files we used in the reduction
        # we'll assume you used all the input files
        # remove duplicates from list
        #print("filenames = " + self._filenames)
        filenames = np.unique(self._filenames)
        nfiles = np.size(filenames)
        hdulist[0].header["DRPNFILE"] = nfiles
        for i, thispath in enumerate(filenames):
            thispath = thispath.replace("\\", '/')
            splited = thispath.split("/")
            fname = splited[-1]
#            matches = re.search('S20[0-9]{6}[SE][0-9]{4}', fname)
            filename = fname#matches.group(0)
            hdulist[0].header["FILE{0}".format(i)] = filename

        # write out psf subtraction parameters
        # get pyKLIP revision number
        pykliproot = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        # the universal_newline argument is just so python3 returns a string instead of bytes
        # this will probably come to bite me later
        try:
            pyklipver = pyklip.__version__
        except:
            pyklipver = "unknown"
        hdulist[0].header['PSFSUB'] = "pyKLIP"
        hdulist[0].header.add_history("Reduced with pyKLIP using commit {0}".format(pyklipver))
        #if self.creator is None:
        #    hdulist[0].header['CREATOR'] = "pyKLIP-{0}".format(pyklipver)
        #else:
        #    hdulist[0].header['CREATOR'] = self.creator
        #    hdulist[0].header.add_history("Reduced by {0}".self.creator)

        # store commit number for pyklip
        hdulist[0].header['pyklipv'] = pyklipver

        if klipparams is not None:
            hdulist[0].header['PSFPARAM'] = klipparams
            hdulist[0].header.add_history("pyKLIP reduction with parameters {0}".format(klipparams))

        if fakePlparams is not None:
            hdulist[0].header['FAKPLPAR'] = fakePlparams
            hdulist[0].header.add_history("pyKLIP reduction with fake planet injection parameters {0}".format(fakePlparams))

        if filetype is not None:
            hdulist[0].header['FILETYPE'] = filetype

        if zaxis is not None:
            #Writing a KL mode Cube
            if "KL Mode" in filetype:
                hdulist[0].header['CTYPE3'] = 'KLMODES'
                #write them individually
                for i, klmode in enumerate(zaxis):
                    hdulist[0].header['KLMODE{0}'.format(i)] = klmode
                    
        #removed astr_hdr stuff            

        #use the dataset center if none was passed in
        if center is None:
            center = self.centers[0]
        if center is not None:
            hdulist[0].header.update({'PSFCENTX':center[0],'PSFCENTY':center[1]})
            hdulist[0].header.update({'CRPIX1':center[0],'CRPIX2':center[1]})
            hdulist[0].header.add_history("Image recentered to {0}".format(str(center)))

        hdulist.writeto(filepath, overwrite=True)
        hdulist.close()

        
def _magao_process_file(self, filepath, highpass=False, pool = None):

    try:
        hdulist = fits.open(filepath)
        header = hdulist[0].header
        cube = hdulist[0].data
        prihdr = hdulist[0].header
        
        wvs = header['WLENGTH'] 

        datasize = cube.shape[1] #ours will be 2D
        center = [[(datasize-1)/2, (datasize-1)/2]]
        
        dims = cube.shape
        x, y = np.meshgrid(np.arange(dims[1], dtype=float), np.arange(dims[0], dtype=float))

        #not really parang, but north up angle, which is the parallactic angle + the angle of the rotator+90
        parang = [(float(header['ROTOFF'])+90)]

        #WCS STUFF --------------

        w = wcs.WCS(header=header, naxis=[1,2])
        
        #define empty cd matrix to put values in later
        w.wcs.cd= np.array([[0,0],[0,0]])
    
        #add WCS info to headers:
        header['CDELT1'] = 2.2222e-6 #coordinate increment, calculated from plate scale
        header['CDELT2'] = 2.2222e-6 #coordinate increment, calculated from plate scale 
        header['CRPIX1'] = (datasize-1)/2.0#x-coordinate of ref pixel (center)
        header['CRPIX2'] = (datasize-1)/2.0#y-coordinate of ref pixel
        header['CRVAL1'] = header['RA'] #Right ascension at ref point , calculated from simbad location of eps eri
        header['CRVAL2'] = header['DEC'] #declination at ref point , calculated from simbad location of eps eri
        header['CTYPE1']  = 'RA---TAN'           #/ First axis is Right Ascension                  
        header['CTYPE2']  = 'DEC--TAN'          # / Second axis is Declination                     
        header['CUNIT1']  = 'deg     '         #  / Units of data                                  
        header['CUNIT2']  = 'deg     '          # / Units of data                                  
        header['RADESYS'] = 'FK5     '           #/ R.A DEC coordinate system reference
        #move data to wcs data format:
        w.wcs.crpix = [header['CRPIX1'], header['CRPIX2']]
        w.wcs.cdelt = np.array([header['CDELT1'], header['CDELT2']])
        w.wcs.crval = [header['CRVAL1'], header['CRVAL2']]
        w.wcs.ctype = [header['CTYPE1'], header['CTYPE2']]

        #turns out WCS data can be wrong. recalculating using rotoff
        rotoff = float(header['ROTOFF'])
        #changed the minus sign in front of vert_angle to fix direction of derotation 
        vert_angle = (rotoff+90)#(360-rotoff)
        vert_angle = np.radians(vert_angle)
        pc = np.array([[-np.cos(vert_angle), np.sin(vert_angle)],[np.sin(vert_angle), np.cos(vert_angle)]])
        pixel_scale = self.lenslet_scale #.008 arcsec/pixel (hard coded, defined in MagAO.ini)
        cdmatrix = pc * pixel_scale /3600.
        w.wcs.cd[0,0] = cdmatrix[0,0]
        w.wcs.cd[0,1] = cdmatrix[0,1]
        w.wcs.cd[1,0] = cdmatrix[1,0]
        w.wcs.cd[1,1] = cdmatrix[1,1]
        astr_hdrs = w

        #-----------------------

        if wvs == 0.6564:
            ghst_psf = self.ghstpeak_ratio['line\'']
        if wvs == 0.6428:
            ghst_psf = self.ghstpeak_ratio['cont\'']
        if wvs ==  0.90969793:
            ghst_psf = self.ghstpeak_ratio['z\'']
        if wvs == 0.76691462:
            ghst_psf = self.ghstpeak_ratio['i\'']
        if 'GHSTPEAK' in header:
            star_flux = [[header['GHSTPEAK']/ghst_psf]]
        if 'STARPEAK' in header:
            star_flux = [[header['STARPEAK']]]
    finally:
        hdulist.close()

    # reshape cube to work with highpass filter
    cube = cube.reshape([1, cube.shape[0], cube.shape[1]]) #makes a 2D y by x image into a 1 by y by x cube

    # high pass and remeasure the satellite spot fluxes if necessary
    highpassed = False
    if isinstance(highpass, bool):
        if highpass:
            cube = high_pass_filter_imgs(cube, pool=pool)
            highpassed = True
    else:
        # should be a number
        if isinstance(highpass, (float, int)):
            highpass = float(highpass)
            fourier_sigma_size = (cube.shape[1] / (highpass)) / (2 * np.sqrt(2 * np.log(2)))
            cube = high_pass_filter_imgs(cube, filtersize=fourier_sigma_size, pool=pool)
            highpassed = True

    #reshape cube because wants to be 2D elsewhere
    cube = cube[0,:,:]

    return cube, center, parang, wvs, astr_hdrs, prihdr, star_flux

#comes from NIRC2 or maybe P1640, but not GPI
#def calc_starflux(cube, center):
 #   dims = cube.shape
 #   y, x = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]))
 #   g_init = models.Gaussian2D(cube.max(), x_mean=center[0][0], y_mean=center[0][1], x_stddev=5, y_stddev=5, fixed={'x_mean':True,'y_mean':True,'theta':True})
 #   fit_g = fitting.LevMarLSQFitter()
 #   g = fit_g(g_init, y, x, cube)
 #   return [[g.amplitude]]
    
    
