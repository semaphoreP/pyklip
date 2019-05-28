import os
import re
import subprocess
import glob
import astropy.io.fits as fits

from astropy import wcs
from astropy.modeling import models, fitting
import numpy as np
import scipy.ndimage as ndimage
import scipy.stats

import sys
from copy import copy
import configparser as ConfigParser

#from pyklip.instruments.P1640_support import P1640spots
#from pyklip.instruments.P1640_support import P1640utils
#from pyklip.instruments.P1640_support import P1640_cube_checker

from scipy.interpolate import interp1d

class MAGAOData(object):
    
    """
    A sequence of MagAO Data. Should handle VisAO or Clio based on header keywords? Each MagAOData object has the following fields and functions 
    Args:
        filepaths: list of filepaths to occulted files
        skipslices: a list of datacube slices to skip (supply index numbers e.g. [0,1,2,3])
        corefilepaths: a list of filepaths to core (i.e. unocculted) files, for contrast calc
        spot_directory: (None)
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

    #I'm marking things that I'm not sure if we need with a "#!"

    ##########################
   ### Class Initialization ###
    ##########################
    #Some static variables to define the MAGAO instrument
    centralwave = {} #in microns
    fpm_diam = {} #in pixels
    flux_zeropt = {}
    spot_ratio = {} #w.r.t. central star
    lenslet_scale = 1.0 #arcseconds per pixel (pixel scale)
    ifs_rotation = 0.0 #degrees CCW from +x axis to zenith
    
    observatory_latitude = 0.0

    #read in MAGAO configuration file and set these static variables
    package_directory = os.path.dirname(os.path.abspath(__file__))
    configfile = package_directory + "/" + "MagAO_clio.ini"
    config = ConfigParser.ConfigParser()
    try:
        config.read(configfile)
        #get pixel scale
        pixel_scale = float(config.get("instrument", "pixel_scale")) #!
        NORTH_clio = float(config.get("instrument", "NORTH_clio")) #!
        #get IFS rotation
        rotation = float(config.get("instrument", "rotation"))
        bands = ['HA', 'CONT', "J", "H", "Ks", "3.1", "3.3", "Lp", "3.9", "Mp",]
        for band in bands:
            centralwave[band] = float(config.get("instrument", "cen_wave_{0}".format(band)))
            fpm_diam[band] = float(config.get("instrument", "fpm_diam_HA".format(band))) #!
            flux_zeropt[band] = float(config.get("instrument", "zero_pt_flux_HA".format(band))) #!
        observatory_latitude = float(config.get("observatory", "observatory_lat"))
    except ConfigParser.Error as e:
        print("Error reading MAGAO configuration file: {0}".format(e.message))
        raise e
    
    #########################
   ###    Constructors     ###
    #########################
    def __init__(self, filepaths=None):
        """
        Initialization code for MAGAOData
        """
        super(MAGAOData, self).__init__()
        self._output = None
        if filepaths is None:
            self._input = None
            self._centers = None
            self._filenums = None
            self._filenames = None
            self._PAs = None
            self._wvs = None
            self._wcs = None
            self._IWA = None
            #self.spot_flux = None #!
            #self.star_flux = None
            self.contrast_scaling = None
            self.prihdrs = None
            self.exthdrs = None
        else:
            self.readdata(filepaths)
    
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
    
    #@property
    #def wcs(self):
    #    return self._wcs
    #@wcs.setter
    #def wcs(self, newval):
    #    self._wcs = newval

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
    def output(self):
        return self._output
    @output.setter
    def output(self, newval):
        self._output = newval

    ###################
   ###    Methods    ###
    ###################
        
    def readdata(self, filepaths):
        """
        Method to open and read a list of MAGAO data
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        data = []
        filenums = []
        filenames = []
        rot_angles = []
        wvs = []
        centers = []
        wcs_hdrs = []
        star_fluxes = []
        #spot_fluxes = [] #!
        prihdrs = []
        self.wcs = []
        
        runningSum = 0
        for index, filepath in enumerate(filepaths):
            rcube, center, pa, wv, astr_hdrs, filt_band, prihdr = _magao_process_file(self, filepath, index)
            
            #runningSum = runningSum + 1
            #print("CUBE[0][0]: " + str(cube[0][0][0]))
            data.append(rcube)
            centers.append(center)
            #star_fluxes.append(star_flux)
            #spot_fluxes.append(spot_flux) #!
            rot_angles.append(pa)
            wvs.append(wv)
            filenums.append(np.ones(pa.shape[0]) * index)
            wcs_hdrs.append(astr_hdrs) #!
            prihdrs.append(prihdr)
            filenames.append([filepath for i in range(pa.shape[0])])
            print("read " + str(runningSum) + " files")
            
            
        centers = np.array(centers)
        #FILENUMS IS 1D LIST
        data = np.array(data)
        dims = data.shape
        print("DIMS = " + str(dims))
        print("LEN=" + str(len(filenums)))
        filenums = np.array(filenums).reshape([dims[0]])
        filenames = np.array(filenames).reshape([dims[0]])
        rot_angles = np.array(rot_angles).reshape([dims[0]])
        wvs = np.array(wvs).reshape([dims[0]])
        print("wvs is ",wvs)
        #wcs_hdrs = np.array(wcs_hdrs).reshape([dims[0] * dims[1]])
        #wcs_hdrs = np.array(wcs_hdrs)
        dsize = dims[0]
        #centers = np.zeros((dsize,2))
        #for y in range(dsize):
        #    for x in range(2):
        #        centers[y][x] = (dims[1]-1)/2
        #        #centers[y][x] = 224.5
        #star_fluxes = np.array(star_fluxes)

        self._input = data
        self._centers = centers
        self._filenums = filenums
        self._filenames = filenames
        print("Filenames associated with self")
        print(self._filenames)
        self._PAs = rot_angles
        self._wvs = wvs
        #self._wcs = None #wvs_hdrs
        #self._wcs.append(astr_hdrs)
        # Creating WCS info for MagAO
        self.wcs = np.array(wcs_hdrs)
        #self.wcs.append(astr_hdrs)
        #self._wcs = np.array(self._wcs)
        #self.wcs = np.array(self.wcs)
        #self.spot_flux = spot_fluxes
        #IWA gets reset by GUI. This is the default value.
        self.IWA = 15
        # half the size of the array
        self.OWA = data.shape[1]/2
        self.flipx = True
        #self.star_flux = star_fluxes
        #self.contrast_scaling = 1./star_fluxes
        self.prihdrs = prihdrs
        self.dn_per_contrast = np.ones(len(filenums))

    def calibrate_output(self, img, spectral=False, units="contrast"):
        """
        Calibrates the flux of the output of PSF subtracted data.

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
        print("Shape of img in calibrate_output: " + str(img.shape))
        if units == "contrast":
            if spectral:
                # spectral cube, each slice needs it's own calibration
                numwvs = img.shape[0]
                img /= self.dn_per_contrast[:numwvs, None, None]
            else:
                # broadband image
                img *= np.nanmean(self.dn_per_contrast)
            self.flux_units = "contrast"

        return img
        
    def savedata(self, filepath, data, klipparams = None, filetype = None, zaxis = None, center=None, astr_hdr=None,
                 fakePlparams = None,):
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
            hdulist[0].header["FILE_{0}".format(i)] = filename

        # write out psf subtraction parameters
        # get pyKLIP revision number
        pykliproot = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        # the universal_newline argument is just so python3 returns a string instead of bytes
        # this will probably come to bite me later
        try:
            pyklipver = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=pykliproot, universal_newlines=True).strip()
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

        #use the dataset astr hdr if none was passed in
        #if astr_hdr is None:
        #    print self.wcs[0]
        #    astr_hdr = self.wcs[0]
        # Removed astr_hdr for now, will try to put it back later
        '''
        if astr_hdr is not None:
            #update astro header
            #I don't have a better way doing this so we'll just inject all the values by hand
            astroheader = astr_hdr.to_header()
            exthdr = hdulist[0].header
            exthdr['PC1_1'] = astroheader['PC1_1']
            exthdr['PC2_2'] = astroheader['PC2_2']
            try:
                exthdr['PC1_2'] = astroheader['PC1_2']
                exthdr['PC2_1'] = astroheader['PC2_1']
            except KeyError:
                exthdr['PC1_2'] = 0.0
                exthdr['PC2_1'] = 0.0
            #remove CD values as those are confusing
            exthdr.remove('CD1_1')
            exthdr.remove('CD1_2')
            exthdr.remove('CD2_1')
            exthdr.remove('CD2_2')
            exthdr['CDELT1'] = 1
            exthdr['CDELT2'] = 1
        '''

        #use the dataset center if none was passed in
        if center is None:
            center = self.centers[0]
        if center is not None:
            hdulist[0].header.update({'PSFCENTX':center[0],'PSFCENTY':center[1]})
            hdulist[0].header.update({'CRPIX1':center[0],'CRPIX2':center[1]})
            hdulist[0].header.add_history("Image recentered to {0}".format(str(center)))

        hdulist.writeto(filepath, clobber=True)
        hdulist.close()

        
def _magao_process_file(self, filepath, filetype, quiet=False):
    #filetype == 0 --> HA
    #filetype == 1 --> CONT
    #print("Reading File: {0}".format(filepath))
    #hdulist = fits.open(filepath)
    #header = hdulist[0].header
    #angle = float(header['ROTOFF'])
    #angle = 90+angle
    #angles = [angle]
    #angles = np.array(angles)
    #hdulist = fits.open(os.getcwd()+"/../HD142527/HD142527/8Apr14/MERGED_long_sets/rotoff_preproc.fits")
    #rotangles = hdulist[0].data
    #rotangles = np.array(rotangles)
    #rotangles = np.zeros((0))
    #print("ROTANGLES = " + str(angle))
    #hdulist.close()
    #hdulist = fits.open(filepath)
    #read in MAGAO configuration file and set these static variables
    package_directory = os.path.dirname(os.path.abspath(__file__))
    configfile = package_directory + "/" + "MagAO_clio.ini"
    config = ConfigParser.ConfigParser()
    config.read(configfile)

    if not quiet:
        print("Reading File: {0}".format(filepath))

    try:
        hdulist = fits.open(filepath)
        header = hdulist[0].header
        angle = float(header['ROTOFF'])
        #angle = 90+angle
        angle = angle-180+float(config.get("instrument", "NORTH_clio"))
        angles = [angle]
        angles = np.array(angles)
        cube = hdulist[0].data
        #exthdr = None
        prihdr = hdulist[0].header
        
        # A.G. edits to get filter info from data header
        filt_band = prihdr['PASSBAND']
        if "MKO" in filt_band: filt_band = filt_band[4]
        wvs = self.centralwave[filt_band]
        #if filetype == 0:
        #    filt_band = "H-Alpha"
        #    wvs = self.centralwave["HA"]
        #else:
        #    filt_band = "Continuum"
        #    wvs = self.centralwave["CONT"]

        datasize = cube.shape[-2] #ours will be 2D

        # a list of centers from the header?
        center = np.array([header["CTRX"], header["CTRY"]])
        #center = [[(datasize-1)/2, (datasize-1)/2]]
        
        dims = cube.shape
        x, y = np.meshgrid(np.arange(dims[1], dtype=np.float32), np.arange(dims[0], dtype=np.float32))
        # From MagAO.py
        #nx = center[0][0] - (x - center[0][0])

        # ghost psf info?
        
        #star_flux = [[1]]
        #check later
       
        if len(dims)==2:
            cube.reshape([1, cube.shape[0], cube.shape[1]]) #makes a 2D y by x image into a 1 by y by x cube
        parang = angles
        #what is this for
        #astr_hdrs = np.repeat(None, 1)
        #or grab the astro header
        w = wcs.WCS(header=header, naxis=[1,2])
        #define empty cd matrix to put values in later
        w.wcs.cd= np.array([[0,0],[0,0]]) #???

        #add WCS info to headers:
        header['CDELT1'] = 2.2222e-6 #coordinate increment, calculated from plate scale
        header['CDELT2'] = 2.2222e-6 #coordinate increment, calculated from plate scale 
        header['CRPIX1'] = 512.0 #x-coordinate of ref pixel
        header['CRPIX2'] = 512.0 #y-coordinate of ref pixel
        header['CRVAL1'] = header['RA'] #Right ascension at ref point , calculated from simbad location of eps eri
        header['CRVAL2'] = header['DEC'] #declination at ref point , calculated from simbad location of eps eri
        header['CTYPE1']  = 'RA---TAN'           #/ First axis is Right Ascension                  
        header['CTYPE2']  = 'DEC--TAN'          # / Second axis is Declination                     
        header['CUNIT1']  = 'deg     '         #  / Units of data                                  
        header['CUNIT2']  = 'deg     '          # / Units of data                                  
        header['RADESYS'] = 'FK5     '           #/ R.A DEC coordinate system reference
        #print('header update check:', header['CDELT1'])
        #move data to wcs data format:
        w.wcs.crpix = [header['CRPIX1'], header['CRPIX2']]
        w.wcs.cdelt = np.array([header['CDELT1'], header['CDELT2']])
        w.wcs.crval = [header['CRVAL1'], header['CRVAL2']]
        w.wcs.ctype = [header['CTYPE1'], header['CTYPE2']]
        #w.wcs.set_pv([(2, 1, 45.0)])

        #turns out WCS data can be wrong. Let's recalculate it using avparang
        #parang = header['PARANG']
        parang = header["ROTOFF"]-180+float(config.get("instrument", "NORTH_Clio")) # The ROTOFF equation
        parang = np.array([parang])
        #changed the minus sign in front of vert_angle to fix direction of derotation 
        vert_angle = (360-parang) 
        vert_angle = np.radians(vert_angle)
        pc = np.array([[np.cos(vert_angle), np.sin(vert_angle)],[-np.sin(vert_angle), np.cos(vert_angle)]])
        pixel_scale = self.lenslet_scale #.008 arcsec/pixel (hard coded, defined in MagAO.ini)
        #print('pixel scale: ', pixel_scale)
        cdmatrix = pc * pixel_scale /3600.
        w.wcs.cd[0,0] = cdmatrix[0,0]
        w.wcs.cd[0,1] = cdmatrix[0,1]
        w.wcs.cd[1,0] = cdmatrix[1,0]
        w.wcs.cd[1,1] = cdmatrix[1,1]
        #print('cd: ',w.wcs.cd)
        #print('wcs: w', w)
        #astr_hdrs = [w.deepcopy() for i in range(channels)] #repeat astrom header for each wavelength slice
        #print(header)
        astr_hdrs = w
    except Exception as e: print('exception: ' +str(e))


    finally:
        hdulist.close()
        
    return cube, center, parang, wvs, astr_hdrs, filt_band, prihdr 


# Need to develop a routine like this:
#def calc_starflux(cube, center):
#    dims = cube.shape
#    y, x = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]))
#    g_init = models.Gaussian2D(cube.max(), x_mean=center[0][0], y_mean=center[0][1], x_stddev=5, y_stddev=5, fixed={'x_mean':True,'y_mean':True,'theta':True})
#    fit_g = fitting.LevMarLSQFitter()
#    g = fit_g(g_init, y, x, cube)
#    return [[g.amplitude]]
