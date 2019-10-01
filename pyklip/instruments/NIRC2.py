import os
import re
import subprocess

import astropy.io.fits as fits
from astropy import wcs
from astropy.modeling import models, fitting
from astropy import time as astrotime
from astropy.coordinates import Angle
import numpy as np
import scipy.ndimage as ndimage
import scipy.stats
from scipy.interpolate import interp1d
import time
from copy import copy
from scipy.integrate import romberg
from astropy.coordinates import SkyCoord, FK5
from astropy import units as u

import multiprocessing as mp

#different imports depending on if python2.7 or python3
import sys
if sys.version_info < (3,0):
    #python 2.7 behavior
    import ConfigParser
else:
    import configparser as ConfigParser
    
from pyklip.instruments.Instrument import Data
from pyklip.instruments.utils.nair import nMathar

from pyklip.parallelized import high_pass_filter_imgs
from pyklip.fakes import gaussfit2d
from pyklip.fakes import gaussfit2dLSQ

class NIRC2Data(Data):
    """
    A sequence of Keck NIRC2 ADI Data. Each NIRC2Data object has the following fields and functions

    Args:
        filepaths: list of filepaths to files
        highpass: if True, run a Gaussian high pass filter (default size is sigma=imgsize/10)
                  can also be a number specifying FWHM of box in pixel units
        find_star: (default) 'auto' will first try to get the star center coordinates from the FITS
                  header PSFCENTX & PSFCENTY keywords, and if that fails it will do a Radon transform to
                  locate the star via the diffraction spikes (and store the star center in the header for
                  future use). True will force the Radon transform; False will skip the Radon transform
                  even if no center is found in the header.

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
        creator: string for creator of the data (used to identify pipelines that call pyklip)
        klipparams: a string that saves the most recent KLIP parameters

    Methods:
        readdata(): reread in the dadta
        savedata(): save a specified data in the GPI datacube format (in the 1st extension header)
        calibrate_output(): flux calibrate the output data
    """
    ##########################
    ###Class Initilization ###
    ##########################
    #some static variables to define the GPI instrument
    lenslet_scales = {}  # in arcsec/pixel
    centralwave = {}  # in microns
    fpm_diam = {}  # in pixels
    fpm_yx = {}  # in pixels
    flux_zeropt = {}
    pupil_diam = {}  # in meters
    #spot_ratio = {} #w.r.t. central star
    #ifs_rotation = 0.0  # degrees CCW from +x axis to zenith

    observatory_latitude = None
    observatory_longitude = None

    ## read in GPI configuration file and set these static variables
    package_directory = os.path.dirname(os.path.abspath(__file__))
    configfile = package_directory + "/" + "NIRC2.ini"
    config = ConfigParser.ConfigParser()
    try:
        config.read(configfile)
        #get list of pixel scales for different cameras and time periods
        cameras = ['narrow_pre150413', 'narrow_post150413', 'medium', 'wide']
        for cam in cameras:
            lenslet_scales[cam] = float(config.get("pixel_scales", "pixel_scale_{0}".format(cam))) # arcsec/pix
        #get IFS rotation
        #ifs_rotation = float(config.get("instrument", "ifs_rotation")) #degrees
        #get some information specific to each band
        bands = ['J', 'H', 'K', 'Ks', 'Kp', 'Lp', 'Ms']
        for band in bands:
            centralwave[band] = float(config.get("instrument", "cen_wave_{0}".format(band)))
            flux_zeropt[band] = float(config.get("instrument", "zero_pt_flux_{0}".format(band)))
            #spot_ratio[band] = float(config.get("instrument", "APOD_{0}".format(band)))
        fpms = ['corona100', 'corona150', 'corona200', 'corona300', 'corona400',
                'corona600', 'corona800', 'corona1000', 'corona1500', 'corona2000']
        for fpm in fpms:
            fpm_diam[fpm] = float(config.get("instrument", "fpm_diam_{0}".format(fpm))) # arcsec
        for fpm in fpms:
            fpm_yx['narrow_' + fpm] = eval(config.get("instrument", "fpm_yx_narrow_{0}".format(fpm))) # (y,x) tuple [pixels]
            fpm_yx['medium_' + fpm] = eval(config.get("instrument", "fpm_yx_medium_{0}".format(fpm))) # (y,x) tuple [pixels]
            fpm_yx['wide_' + fpm] = eval(config.get("instrument", "fpm_yx_wide_{0}".format(fpm))) # (y,x) tuple [pixels]
        pupils = ['incircle', 'largehex', 'smallhex', 'open']
        for pupil in pupils:
            pupil_diam[pupil] = float(config.get("instrument", "pupil_diam_{0}".format(pupil))) # meters
        
        observatory_latitude = float(config.get("observatory", "observatory_lat"))
        observatory_longitude = float(config.get("observatory", "observatory_lon"))
    except ConfigParser.Error as e:
        print("Error reading GPI configuration file: {0}".format(e.message))
        raise e

    ####################
    ### Constructors ###
    ####################
    def __init__(self, filepaths=None, highpass=False, find_star='auto', meas_star_flux=False):
        """
        Initialization code for NIRC2Data

        Note:
            see class docstring for argument details
        """
        super(NIRC2Data, self).__init__()
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
            self.spot_flux = None
            self.star_flux = None
            self.contrast_scaling = None
            self.prihdrs = None
            self.exthdrs = None
            self.lenslet_scale = None
            self.pupil_diam = None
        else:
            self.readdata(filepaths, highpass=highpass, find_star=find_star, meas_star_flux=meas_star_flux)


    ################################
    ### Instance Required Fields ###
    ################################
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
    def filenums(self):
        return self._filenums
    @filenums.setter
    def filenums(self, newval):
        self._filenums = newval

    @property
    def filenames(self):
        return self._filenames
    @filenames.setter
    def filenames(self, newval):
        self._filenames = newval

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


    ###############
    ### Methods ###
    ###############
    def readdata(self, filepaths, highpass=False, find_star='auto', meas_star_flux=False):
        """
        Method to open and read a list of NIRC2 data

        Args:
            filespaths: a list of filepaths
            highpass: if True, run a Gaussian high pass filter (default size is sigma=imgsize/10)
                  can also be a number specifying FWHM of box in pixel units
            find_star: (default) 'auto' will first try to get the star center coordinates from the FITS
                  header PSFCENTX & PSFCENTY keywords, and if that fails it will do a Radon transform to
                  locate the star via the diffraction spikes (and store the star center in the header for
                  future use). True will force the Radon transform; False will skip the Radon transform
                  even if no center is found in the header.

        Returns:
            Technically none. It saves things to fields of the NIRC2Data object. See object doc string
        """
        #check to see if user just inputted a single filename string
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        #make some lists for quick appending
        data = []
        filenums = []
        filenames = []
        rot_angles = []
        wvs = []
        centers = []
        wcs_hdrs = []
        star_fluxes = []
        spot_fluxes = []
        prihdrs = []
        
        #Create a threadpool for high pass filter
        pool = None
        if highpass:
            pool = mp.Pool()

        #extract data from each file
        for index, filepath in enumerate(filepaths):
            cube, center, pa, wv, astr_hdrs, filt_band, fpm_band, pupil, star_flux, spot_flux, prihdr, exthdr, camera, obsdate =\
                _nirc2_process_file(filepath, highpass=highpass, find_star=find_star, meas_star_flux=meas_star_flux, pool=pool)
            
            data.append(cube)
            centers.append(center)
            star_fluxes.append(star_flux)
            spot_fluxes.append(spot_flux)
            rot_angles.append(pa)
            wvs.append(wv)
            filenums.append(np.ones(pa.shape[0]) * index)
            wcs_hdrs.append(astr_hdrs)
            prihdrs.append(prihdr)

            #filename = np.chararray(pa.shape[0])
            #filename[:] = filepath
            filenames.append([filepath for i in range(pa.shape[0])])
        
        # Close threadpool
        if highpass:
            pool.close()
            pool.join()
        
        #convert everything into numpy arrays
        #reshape arrays so that we collapse all the files together (i.e. don't care about distinguishing files)
        data = np.array(data)
        dims = data.shape
        data = data.reshape([dims[0] * dims[1], dims[2], dims[3]])
        filenums = np.array(filenums).reshape([dims[0] * dims[1]])
        filenames = np.array(filenames).reshape([dims[0] * dims[1]])
        rot_angles = np.array(rot_angles).reshape([dims[0] * dims[1]])  # want North Up
        wvs = np.array(wvs).reshape([dims[0] * dims[1]])
        wcs_hdrs = np.array(wcs_hdrs).reshape([dims[0] * dims[1]])
        centers = np.array(centers).reshape([dims[0] * dims[1], 2])
        star_fluxes = np.array(star_fluxes).reshape([dims[0] * dims[1]])
        spot_fluxes = np.array(spot_fluxes).reshape([dims[0] * dims[1]])
        
        #select correct pixel scale
        #narrow camera pixel scale changed after servicing on 2015-04-13
        date_2015_4_13 = time.strptime("2015-4-13", "%Y-%m-%d")
        if camera=='narrow':
            if obsdate < date_2015_4_13:
                lenslet_scale = NIRC2Data.lenslet_scales['narrow_pre150413']
            elif obsdate >= date_2015_4_13:
                lenslet_scale = NIRC2Data.lenslet_scales['narrow_post150413']
        else:
            lenslet_scale = NIRC2Data.lenslet_scales[camera]
        self.lenslet_scale = lenslet_scale
        
        #set these as the fields for the GPIData object
        self._input = data
        self._centers = centers
        self._filenums = filenums
        self._filenames = filenames
        self._PAs = rot_angles
        self._wvs = wvs
        self._wcs = [None] #wcs_hdrs
        self.spot_flux = spot_fluxes
        self._IWA = NIRC2Data.fpm_diam[fpm_band]/lenslet_scale/2.0
        self.star_flux = star_fluxes
        self.contrast_scaling = 1./star_fluxes #GPIData.spot_ratio[ppm_band]/np.tile(np.mean(spot_fluxes.reshape(dims[0], dims[1]), axis=0), dims[0])
        self.prihdrs = prihdrs
        self.pupil_diam = NIRC2Data.pupil_diam[pupil]
        
        #self.exthdrs = exthdrs

    def savedata(self, filepath, data, klipparams = None, filetype = None, zaxis = None, center=None, astr_hdr=None,
                 fakePlparams = None, more_keywords=None):
        """
        Save data in a GPI-like fashion. Aka, data and header are in the first extension header

        Note: In principle, the function only works inside klip_dataset(). In order to use it outside of klip_dataset,
            you need to define the following attribute:
                dataset.output_centers = dataset.centers

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
            more_keywords (dictionary) : a dictionary {key: value, key:value} of header keywords and values which will
                            written into the primary header

        """
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(header=self.prihdrs[0]))
        hdulist.append(fits.ImageHDU(data=data, name="Sci"))

        # save all the files we used in the reduction
        # we'll assume you used all the input files
        # remove duplicates from list
        filenames = np.unique(self.filenames)
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
        #    print(self.wcs[0])
        #    astr_hdr = self.wcs[0]
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

        #use the dataset center if none was passed in
        if center is None:
            center = self.output_centers[0]
        if center is not None:
            hdulist[0].header.update({'PSFCENTX':center[0],'PSFCENTY':center[1]})
            hdulist[0].header.update({'CRPIX1':center[0],'CRPIX2':center[1]})
            hdulist[0].header.add_history("Image recentered to {0}".format(str(center)))

        # store extra keywords in header
        if more_keywords is not None:
            for hdr_key in more_keywords:
                hdulist[0].header[hdr_key] = more_keywords[hdr_key]

        try:
            hdulist.writeto(filepath, overwrite=True)
        except TypeError:
            hdulist.writeto(filepath, clobber=True)
        hdulist.close()

    def calibrate_data(self, units="contrast"):
        """
        Calibrates the flux of the output of PSF subtracted data.

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
            for i in range(self.output.shape[0]):
                self.output[i] *= self.contrast_scaling[:, None, None]

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
        if units == "contrast":
            if spectral:
                # spectral cube, each slice needs it's own calibration
                numwvs = img.shape[0]
                img *= self.contrast_scaling[:numwvs, None, None]
            else:
                # broadband image
                img *= np.nanmean(self.contrast_scaling)

        return img

######################
## Static Functions ##
######################

def _nirc2_process_file(filepath, highpass=False, find_star='auto', meas_star_flux=False, pool=None):
    """
    Method to open and parse a NIRC2 file

    Args:
        filepath: the file to open
        highpass: if True, run a Gaussian high pass filter (default size is sigma=imgsize/10)
                  can also be a number specifying FWHM of box in pixel units
        find_star: (default) 'auto' will first try to get the star center coordinates from the FITS
                  header PSFCENTX & PSFCENTY keywords, and if that fails it will do a Radon transform to
                  locate the star via the diffraction spikes (and store the star center in the header for
                  future use). True will force the Radon transform; False will skip the Radon transform
                  even if no center is found in the header.
        meas_star_flux: if True, measures the stellar flux
        pool: optional to pass along a threadpool (mainly for highpass filtering multiple images)

    Returns: (using z as size of 3rd dimension, z=1 for NIRC2)
        cube: 3D data cube from the file. Shape is (z,256,256)
        center: array of shape (z,2) giving each datacube slice a [xcenter,ycenter] in that order
        parang: array of z of the parallactic angle of the target (same value just repeated z times)
        wvs: array of z of the wavelength of each datacube slice. (For NIRC2, wvs = [None])
        astr_hdrs: array of z of the WCS header for each datacube slice. (For NIRC2, wcs = [None])
        filt_band: the band (Ks, Lp, Ms) used in the data (string)
        fpm_band: For NIRC2, fpm_band = [None]
        ppm_band: For NIRC2, ppm_band = [None]
        spot_fluxes: For NIRC2, array of z containing 1.0 for each image
        prihdr: primary header of the FITS file
        exthdr: For NIRC2, None
    """
    print("Reading File: {0}".format(filepath))
    hdulist = fits.open(filepath, mode='update', ignore_missing_end=True) # some NIRC2 FITS files are missing END card
    try:
        #grab the data and headers
        cube = hdulist[0].data
        exthdr = None #hdulist[1].header
        prihdr = hdulist[0].header
        
        obsdate = time.strptime(prihdr['DATE-OBS'], "%Y-%m-%d") # observation date

        #get some instrument configuration from the primary header
        filt_band = prihdr['FILTER'].split('+')[0].strip()
        fpm_band = prihdr['SLITNAME']
        camera = prihdr['CAMNAME']
        pupil = prihdr['PMSNAME']
        rotmode = prihdr['ROTMODE'].lower()

        #for NIRC2, we only have broadband but want to keep the GPI array shape to make processing easier
        if prihdr['CURRINST'].strip() == 'NIRC2':
            wvs = [1.0]
            
            # Pull PA from header if already there, or try to calculate it from header.
            if 'ROTNORTH' in prihdr.keys():
                parang = prihdr['ROTNORTH']*np.ones(1)
            else:
                try:
                    parang = get_pa(hdulist, obsdate=obsdate, rotmode=rotmode, write_hdr=True)*np.ones(1)
                except:
                    parang = np.nan*np.ones(1)
            
            if find_star is True:
                # Ignore header and use Radon transform to find star center.
                try:
                    center = [get_star(hdulist, ctr=NIRC2Data.fpm_yx[('_').join([camera, fpm_band])], obsdate=obsdate,
                                    hp_size=0, im_smooth=0, sp_width=0, rad=100, radon_wdw=400, smooth=1,
                                    write_hdr=True, pool=pool, silent=True)]
                except:
                    center = [[np.nan, np.nan]]
            elif find_star is False:
                # Only try to get star center from headers (no Radon transform).
                try:
                    center = [[prihdr['PSFCENTX'], prihdr['PSFCENTY']]]
                except:
                    center = [[np.nan, np.nan]]
            elif find_star=='auto':
                # Pull star center from header if already there, or try to find it via Radon transform.
                # Radon assumes star is near center of FPM and may fail if otherwise.
                if 'PSFCENTX' and 'PSFCENTY' in prihdr.keys():
                    center = [[prihdr['PSFCENTX'], prihdr['PSFCENTY']]]
                else:
                    try:
                        center = [get_star(hdulist, ctr=NIRC2Data.fpm_yx[('_').join([camera, fpm_band])], obsdate=obsdate,
                                        hp_size=0, im_smooth=0, sp_width=0, rad=100, radon_wdw=400, smooth=1,
                                        write_hdr=True, pool=pool, silent=True)]
                    except:
                        center = [[np.nan, np.nan]]
            else:
                raise ValueError("Unsupported value for find_star; only 'auto', True, or False are accepted.")
            
            # Flipping x-axis to enable use of GPI data rotation code without modification
            dims = cube.shape
            x, y = np.meshgrid(np.arange(dims[1], dtype=np.float32), np.arange(dims[0], dtype=np.float32))
            nx = center[0][0] - (x - center[0][0])
            minval = np.min([np.nanmin(cube), 0.0])
            flipped_cube = ndimage.map_coordinates(np.copy(cube), [y, nx], cval=minval * 5.0)
            star_flux = np.nan
            if meas_star_flux:
                star_flux = calc_starflux(flipped_cube, center)
            cube = flipped_cube.reshape([1, flipped_cube.shape[0], flipped_cube.shape[1]])  #maintain 3d-ness
            astr_hdrs = np.repeat(None, 1)
            spot_fluxes = [[1]] #not suported currently
    finally:
        hdulist.close()
    
    #high pass filter
    highpassed = False
    if isinstance(highpass, bool):
        if highpass:
            cube = high_pass_filter_imgs(cube, pool=pool)
            highpassed = True
    else:
        # should be a number
        if isinstance(highpass, (float, int)):
            highpass = float(highpass)
            fourier_sigma_size = (cube.shape[1]/(highpass)) / (2*np.sqrt(2*np.log(2)))
            cube = high_pass_filter_imgs(cube, filtersize=fourier_sigma_size, pool=pool)
            highpassed = True

    return cube, center, parang, wvs, astr_hdrs, filt_band, fpm_band, pupil, star_flux, spot_fluxes, prihdr, exthdr, camera, obsdate

def calc_starflux(cube, center):
    """
    Fits a 2D Gaussian to an image to calculate the peak pixel value of
    the central star. The code assumes an unobscurated PSF.

    Args:
        cube: 2D image array. Shape is (256,256)
        center: star center in image in (x,y)

    Returns:
        Amplitude: Best fit amplitude of the 2D Gaussian.
    """

    dims = cube.shape
    y, x = np.meshgrid( np.arange(dims[0]), np.arange(dims[1]) )

    # Initializing Model. Fixing the rotation and the X, Y location of the star.
    g_init = models.Gaussian2D(cube.max(), x_mean=center[0][0], y_mean=center[0][1], x_stddev=5, y_stddev=5, \
        fixed={'x_mean':True,'y_mean':True,'theta':True})

    # Initializing Levenburg-Marquart Least-Squares fitting routine.
    fit_g = fitting.LevMarLSQFitter()

    # Fitting the amplitude, x_stddev and y_stddev
    g = fit_g(g_init, y, x, cube)

    return [[g.amplitude]]

def measure_star_flux(img, star_x, star_y):
    """
    Measure star peak fluxes using a Gaussian matched filter

    Args:
        img: 2D frame with unobscured, unsaturated PSF
        star_x, star_y: coordinates of the star
    Return:
        star_f: star flux
    """

    flux, fwhm, xfit, yfit = gaussfit2d(img, star_x, star_y, refinefit=False)
    if flux == np.inf: flux == np.nan
    print(flux, fwhm, xfit, yfit)

    return flux

def get_pa(hdulist, obsdate=None, rotmode=None, mean_PA=True, write_hdr=True, new_method=True):
    """
    Given a FITS data-header unit list (HDUList), returns the NIRC2 PA in [radians].
    PA is angle of detector relative to sky; ROTMODE is rotator tracking mode;
    PARANG is parallactic angle astrometric; INSTANGL is instrument angle;
    ROTPOSN is rotator physical position.
    Additional PA offset of -0.252 or -0.262 deg is applied for NIRC2 narrow cam
    depending on observation date.
    NOTE that the PA sign is flipped at the very end before output to conform to
    pyKLIP's rotation convention.
    
    Inputs:
        hdulist: a FITS HDUList (NOT a single HDU).
        obsdate: date of observation; will try to get from prihdr if not provided.
        rotmode: 'vertical angle' for ADI mode with PA rotating on detector, or
                 'position angle' for mode with PA orientation fixed on detector.
        mean_pa: if True (default), return the mean PA during the exposure.
                 If False, return the PA at the start of the exposure only.
                 Only applies to vertical angle mode.
        write_hdr: if True (default), writes keys to file header and saves them.
    """
    
    prihdr = hdulist[0].header
    
    # Date of NIRC2 servicing that changed PA offset.
    date_2015_4_13 = time.strptime("2015-4-13", "%Y-%m-%d")
    
    # If don't have it, get observation date (UT) from header and make a time object.
    if obsdate is None:
        obsdate = time.strptime(prihdr['DATE-OBS'], "%Y-%m-%d")
    
    # Additional offset to narrow cam PA not included in INSTANGL keyword.
    # This offset changed after instrument servicing on April 13, 2015.
    if prihdr['CAMNAME'].lower() == 'narrow':
        if obsdate < date_2015_4_13:
            zp_offset = -0.252 # [deg]; from Yelda et al. 2010
        elif obsdate >= date_2015_4_13:
            zp_offset = -0.262 # [deg]; from Service et al. 2016
    else:
        zp_offset = 0.
        print("WARNING: No PA offset applied.")
    
    if rotmode is None:
        global _last_rotmode
        try:
            rotmode = prihdr['ROTMODE'].strip().lower()
            _last_rotmode = rotmode
        except:
            rotmode = _last_rotmode.lower()
    rotposn = prihdr['ROTPOSN'] # [deg]
    instangl = prihdr['INSTANGL'] # [deg]

    if rotmode.lower() == 'vertical angle':
        parang = prihdr['PARANG'] # [deg]
        pa_deg = parang + rotposn - instangl + zp_offset # [deg]
    elif rotmode.lower() == 'position angle':
        pa_deg = rotposn - instangl + zp_offset # [deg]
    else:
        raise NotImplementedError
    
    if mean_PA and (rotmode.lower() == 'vertical angle'):
        if new_method is False:
            # Get info for PA smearing calculation.
            epochobj = prihdr['DATE-OBS']
            name = prihdr['targname']
            expref = prihdr['itime']
            coaddref = prihdr['coadds']
            sampref =  prihdr['sampmode']
            msrref =   prihdr['MULTISAM']
            xdimref =  prihdr['naxis1']
            ydimref =  prihdr['naxis2']
            tel = prihdr['TELESCOP']
            dec = prihdr['DEC'] + prihdr['DECOFF']
            if tel.lower() == 'keck ii':
                tel = 'keck2' # just cleaning up str
            
            # Calculate total time of exposure (integration + readout).
            if sampref == 2: totexp = ( expref + 0.18*(xdimref/1024.)**2) * coaddref
            if sampref == 3: totexp = ( expref + (msrref-1)*0.18*(xdimref/1024.)**2)*coaddref
            
            tinteg = totexp # [seconds]
            totexp = totexp/3600. # [hours]
            
            # Get hour angle at start of exposure.
            tmpahinit = prihdr['HA'] # [deg]
            ahobs = 24.*tmpahinit/360. # [hours]
            
            # Estimate vertical position angle at each second of the exposure.
            vp = [] #fltarr(round(3600.*totexp))
            for j in range(0, int(round(3600.*totexp*100.))-1):
                ahtmp = ahobs + (j*1.+0.001)/(3600.*100.) # [hours]
                vp.append(par_angle(ahtmp, dec, NIRC2Data.observatory_latitude))
                if j == 0: vpref = vp[0]
            vp = np.array(vp)
            
            # Handle case where PA crosses 0 <--> 360.
            vp[vp < 0.] += 360.
            vp[vp > 360.] -= 360.
            if vpref < 0.: vpref += 360.
            if vpref > 360.: vpref -= 360.
            
            # Check that images near PA=0 are handled correctly.
            if any(vp > 350) & any(vp < 10):
                vp[vp > 350] -= 360
            
            vpmean = np.nanmean(vp)
            
            if (vpmean < 0) & (vpref > 350):
                vpmean += 360.
            
            pa_deg_mean = pa_deg + (vpmean - vpref)
        else:

            expref = prihdr['itime']
            coaddref = prihdr['coadds']
            sampref =  prihdr['sampmode']
            msrref =   prihdr['MULTISAM']
            xdimref =  prihdr['naxis1']
            ydimref =  prihdr['naxis2']
            tel = prihdr['TELESCOP']
            dec = prihdr['DEC'] + prihdr['DECOFF']
            if tel.lower() == 'keck ii':
                tel = 'keck2' # just cleaning up str
            
            # Calculate total time of exposure (integration + readout).
            if sampref == 2: totexp = ( expref + 0.18*(xdimref/1024.)**2) * coaddref
            if sampref == 3: totexp = ( expref + (msrref-1)*0.18*(xdimref/1024.)**2)*coaddref
            tinteg = totexp # [seconds]

            ## date-obs saved when command issued, need to check if UT > 12 and EXPSTART < 12 (if so, add one to date)
            expstart = prihdr['EXPSTART']
            expend = prihdr['EXPSTOP']

            ut = Angle(prihdr['UTC']+' h').hour
            expstart_hr = Angle(expstart+' h').hour
            expend_hr = Angle(expend+' h').hour

            if (ut > 23) & (expstart_hr < 1):
                # Crossed the date line between UT and EXPSTART
                date_start = (astrotime.Time(prihdr['DATE-OBS']+'T12:00:00', format='isot', scale='ut1') + astrotime.TimeDelta(1.0, format='jd')).isot[0:10]
            else:
                date_start = prihdr['DATE-OBS']

            if expend_hr < expstart_hr:
                # Crossed date line between EXPSTART and EXPSTOP
                date_end = (astrotime.Time(date_start+'T12:00:00', format='isot', scale='ut1') + astrotime.TimeDelta(1.0, format='jd')).isot[0:10]
            else:
                date_end = date_start


            # Calculate LST for:
            #    -the time the header was written (assumed to be when PARANG was calculated)
            #    -the time at the start of the exposure
            #    -the time at the end

            lst_ref = astrotime.Time(prihdr['DATE-OBS']+'T'+prihdr['UTC'], format='isot', scale='ut1').sidereal_time('apparent', 'greenwich').degree
            lst0 = astrotime.Time(date_start+'T'+expstart, format='isot', scale='ut1').sidereal_time('apparent', 'greenwich').degree
            lst1 = (astrotime.Time(date_start+'T'+expstart, format='isot', scale='ut1') + astrotime.TimeDelta(tinteg, format='sec')).sidereal_time('apparent', 'greenwich').degree

            lst_ref = (lst_ref + NIRC2Data.observatory_longitude) % 360.
            lst0 = (lst0 + NIRC2Data.observatory_longitude) % 360.
            lst1 = (lst1 + NIRC2Data.observatory_longitude) % 360.

            r = prihdr['RA'] + prihdr['RAOFF'] # degrees
            d = (prihdr['DEC'] + prihdr['DECOFF']) * np.pi/180. # radians
            # Header information seems to have changed after 2017.8 (mid Oct 2017)
            obs_epoch = astrotime.Time(prihdr['DATE-OBS'], format='iso', scale='utc')
            if obs_epoch.decimalyear < 2017.8:
                coor = SkyCoord(ra=r, dec=d*180./np.pi, unit=(u.deg, u.deg), frame=FK5, equinox='J2000.0')
                coor_curr = coor.transform_to(FK5(equinox=obs_epoch))
                rp = coor_curr.ra.value # degrees
                dp = (coor_curr.dec.value) * np.pi/180. # radians
            else:
                rp = prihdr['RA'] + prihdr['RAOFF'] # degrees
                dp = (prihdr['DEC'] + prihdr['DECOFF']) * np.pi/180. # radians

            ha_ref = (lst_ref - rp)/15.
            ha0 = (lst0 - rp)/15.
            ha1 = (lst1 - rp)/15.          

            if ha0 <= -12.:
                ha0 += 24.
            if ha0 > 24.:
                ha0 -= 24.
            if ha1 <= -12.:
                ha1 += 24.
            if ha1 > 24.:
                ha1 -= 24.
            if ha_ref <= -12.:
                ha_ref += 24.
            if ha_ref > 24.:
                ha_ref -= 24.

            h_ref = ha_ref * 15. * np.pi/180.
            h0 = ha0 * 15. * np.pi/180.
            h1 = ha1 * 15. * np.pi/180.

            phi = NIRC2Data.observatory_latitude * np.pi/180.

            if ((h1 * h0) < 0) and (d > phi):
                wrap_flag = 1
            else:
                wrap_flag = 0

            result = (romberg(parang_eq, h0, h1, args=(d, phi, wrap_flag))/(h1-h0)) * (180./np.pi)

            pa_ref = -np.arctan2(-np.sin(h_ref), np.cos(dp)*np.tan(phi) - np.sin(dp)*np.cos(h_ref))*(180./np.pi)
            delta_pa = result-pa_ref

            pa_deg_mean = prihdr['PARANG'] + delta_pa + rotposn - instangl + zp_offset
            
    else:
        pa_deg_mean = pa_deg
        vpmean = np.nan
        vpref = np.nan
        totexp = np.nan
    
    if write_hdr:
        # Flip signs to conform to pyKLIP rotation convention.
        prihdr['TOTEXP'] = (totexp, 'Total exposure time [hours]')
        prihdr['PASTART'] = (-1*pa_deg, "Position angle at exposure start [deg]")
        prihdr['PASMEAR'] = (-1*(vpmean - vpref), "Exposure's weighted-mean PA minus PASTART [deg]")
        prihdr['ROTNORTH'] = (-1*pa_deg_mean, "Mean PA of North during exposure [deg]")
        
        hdulist.flush()
    
    # Flip sign to conform to pyKLIP rotation convention.
    return -1*pa_deg_mean

def parang_eq(H, d, phi, wrap_flag):
    paint_ineq = np.arctan2(np.sin(H)*np.cos(phi),np.sin(phi)*np.cos(d) - np.sin(d)*np.cos(phi)*np.cos(H))
    if wrap_flag and (H < 0.0):
        paint_ineq += 2.0*np.pi

    return paint_ineq

def par_angle(HA, dec, lat):
    """
    Compute the parallactic angle, given hour angle (HA [hours]),
    declination (dec [deg]), and latitude (lat [deg]).  Returns
    parallactic angle in [deg].
    """
    HA_rad = np.radians(HA*15.) # [hours] -> [rad]
    dec_rad = np.radians(dec)   # [deg] -> [rad]
    lat_rad = np.radians(lat)   # [deg] -> [rad]

    parallang = -np.arctan2(-np.sin(HA_rad),  # [rad]
                          np.cos(dec_rad)*np.tan(lat_rad) - np.sin(dec_rad)*np.cos(HA_rad))
    
    return np.degrees(parallang) # [deg]

def get_star(hdulist, ctr, obsdate, hp_size=0, im_smooth=0, sp_width=0, spike_angles=None,
              r_mask='all', rad=100, rad_out=np.inf, radon_wdw=400, smooth=1, PAadj=0.,
              write_hdr=True, pool=None, silent=True):
    """
    Runs Radon transform star-finding algorithm on image and (by default) saves the results
    in the original FITS header.
    
    Inputs:
        hdulist: a FITS HDUList (NOT a single HDU).
        ctr: (y,x) coordinate pair estimate for star position for image [pix].
        obsdate: date of observation; will try to get from prihdr if not provided.
        hp_size: size of high-pass filter box (via Fourier transform) in [pix].
        im_smooth: sigma of smoothing Gaussian function in [pix].
        sp_width: width of diffraction spike mask in [pix]; default is 0 (no masking).
        spike_angles: list of discrete angles from the assumed star positions along 
            which the radon transform will sum intensity to search for the true star
            position (it picks the maximum sum). These should match the angles
            of the strongest diffraction spikes [deg].
        r_mask: 'all' to mask out circle around ctr coords; anything else to do no radial masking.
        rad: r_mask=='all' will mask out all r <= rad [pix].
        rad_out: r_mask=='all' will mask out all r > rad_out [pix].
        radon_window: half width of the radon sampling region; size_window = image.shape[0]/2 is suggested.
            m & M:  The sampling region will be (-M*size_window, -m*size_window)U(m*size_window, M*size_window).
        smooth: smooth the radon cost function; for one pixel, replace it by the
            average of its +/- smooth neighbours; default = 2.
        PAadj: optional angle by which to rotate diffraction spike angles in [radians].
        write_hdr: (default) True will write the Radon transform star center to the original
            FITS header in PSFCENTX & PSFCENTY keywords.
        pool: multiprocessing pool for highpass filtering and other parallel uses.
        silent: (default) True to suppress additional output to stdout.
    
    Outputs:
        Returns [X,Y] list of Radon transform star center. Default is to also write
        the star coordinates to PSFCENTX & PSFCENTY in original FITS header.
    """
    from pyklip.instruments.utils.radonCenter import searchCenter
    from scipy.ndimage.filters import median_filter, gaussian_filter
    
    if not silent: print("Finding star...")
    data = hdulist[0].data.copy()
    hdr = hdulist[0].header
    
    # median filter data to replace NaN's with median values of neighbors.
    wh = np.where(np.isnan(data))
    data_medfilt = median_filter(np.nan_to_num(data), size=9)
    # replace all NaN in f with median values of neighbors.
    data[wh] = data_medfilt[wh]
    
    data = np.ma.masked_invalid(data)
    
    # Highpass filtering.
    if hp_size != 0:
        # should be a number
        fourier_sigma_size = (data.shape[1]/float(hp_size)) / (2*np.sqrt(2*np.log(2)))
        data_filt = high_pass_filter_imgs(np.array([data]), filtersize=fourier_sigma_size, pool=pool)[0]
    else:
        data_filt = data
    
    # Image smoothing.
    if im_smooth != 0:
        data_filt = gaussian_filter(data_filt, im_smooth)
    
    # Build cartesian coordinate grid and radius map.
    yy, xx = np.mgrid[0:data_filt.shape[0]:1, 0:data_filt.shape[1]:1]
    radii = np.sqrt((yy - ctr[0])**2 + (xx - ctr[1])**2)
    
    # Additional offset to narrow cam PA not included in INSTANGL keyword.
    # This offset changed after instrument servicing on April 13, 2015.
    date_2015_4_13 = time.strptime("2015-4-13", "%Y-%m-%d")
    if hdr['CAMNAME'].lower()=='narrow':
        if obsdate < date_2015_4_13:
            zp_offset = -0.252 # [deg]; from Yelda et al. 2010
        elif obsdate >= date_2015_4_13:
            zp_offset = -0.262 # [deg]; from Service et al. 2016
    elif hdr['CAMNAME'].lower()=='wide':
        zp_offset = 0.
    
    # Select angles along which to perform radon based on rotator mode.
    # Position angle (pa) of telescope optics only, from header info.
    if hdr['ROTMODE'].lower() == 'vertical angle':
        pa_tele = hdr['INSTANGL'] - zp_offset - hdr['ROTPOSN'] # [deg]
    # Other case is either 'position angle' mode or unidentified.
    else:
        pa_tele = hdr['PARANG'] # [deg]
    if spike_angles is None:
        # Angles at which diffraction spikes occur in NIRC2 data [deg].
        spike_angles = pa_tele + 30.0 + np.arange(3)*60.0 + PAadj # [deg]
    # Boolean mask excluding everything except diffraction spikes.
    spikemask = ~ make_spikemask(data_filt, hdr, ctr, spike_angles, yy, xx, sp_width) # ~ is inverse boolean
    
    mask_total = spikemask.copy()
    
    # Optionally mask stellar halo with additional circular mask centered on ctr.
    # Masked regions are r <= rad and r > rad_out.
    if r_mask=='all':
        mask_total[(spikemask==False) & (radii <= rad)] = True
        mask_total[(spikemask==False) & (radii > rad_out)] = True
    else:
        pass
    
    data_masked = np.ma.array(data_filt, mask=mask_total)
    
    # Perform radon transform search for star.
    if not silent: print("Performing radon transform search...")
    (x_radon, y_radon) = searchCenter(data_masked.filled(0.), ctr[1], ctr[0],
                            size_window=radon_wdw, size_cost=7, m=0.2, M=0.8,
                            smooth=smooth, theta=spike_angles)

    if not silent: print("radon y,x = {0}, {1}".format(y_radon, x_radon))
    
    if write_hdr:
        # Update the original FITS header with new star coordinates.
        hdr['PSFCENTX'] = (x_radon, 'Star X numpy coord')
        hdr['PSFCENTY'] = (y_radon, 'Star Y numpy coord')
        
        hdulist.flush()
        if not silent: print("Wrote new star coordinates to FITS header in PSFCENTX & PSFCENTY.")
    
    return [x_radon, y_radon]

def make_spikemask(data, hdr, ctr, spike_angles, yy, xx, width=31):
    """
    Construct diffraction spike mask from FITS header information.
    
    data: 2-D ndarray image to be masked (just to get size of array).
    hdr: FITS header for image constructing mask for.
    ctr: (y,x) coordinates for center of diffraction spike pattern
            (usually the estimated star location).
    spike_angles: position angles for diffraction spikes in image [radians].
    yy: mgrid or indices 2-D array of pixel y-coordinates.
    xx: mgrid or indices 2-D array of pixel x-coordinates.
    width: int or float width of spike mask in [pixels]; 0 for no mask.
    """
    
    mask = np.zeros(data.shape, dtype=np.bool)
    
    if (width > 0) & (not np.isnan(width)):
        yy_ctr = yy - ctr[0]
        xx_ctr = xx - ctr[1]
        
        for ang in np.radians(spike_angles):
            # Slope times x for each spike.
            mx = np.tan(ang)*xx_ctr
            # Mask with span 0.5*width above and below spike.
            band = np.abs(0.5*width/np.cos(ang))
            # Change spikemask to True anywhere inside mask.
            spikemask = (yy_ctr <= (mx + band)) & (yy_ctr >= (mx - band))
            # Replace False with True in mask wherever either spikemask or mask is True.
            mask = mask | spikemask
    else:
        mask = np.ones(data.shape, dtype=np.bool)
    
    return mask

