import os, subprocess
import astropy.io.fits as fits
from astropy import wcs
import numpy as np
import scipy.ndimage as ndimage

from pyklip.instruments.Instrument import Data

class Ifs(Data):
    """
    A sequence of SPHERE IFS Data.

    Args:
        data_cube: FITS file with a 4D-cube (Nfiles, Nwvs, Ny, Nx) with all IFS coronagraphic data
        psf_cube: FITS file with a 3-D (Nwvs, Ny, Nx) PSF cube
        info_fits: FITS file with a table in the 1st ext hdr with parallactic angle info
        wavelenegth_info: FITS file with a 1-D array (Nwvs) of the wavelength sol'n of a cube
        psf_cube_size: size of the psf cube to save (length along 1 dimension)
        nan_mask_boxsize: size of box centered around any pixel <= 0 to mask as NaNs
        IWA: inner working angle of the data in arcsecs

    Attributes:
        input: Array of shape (N,y,x) for N images of shape (y,x)
        centers: Array of shape (N,2) for N centers in the format [x_cent, y_cent]
        filenums: Array of size N for the numerical index to map data to file that was passed in
        filenames: Array of size N for the actual filepath of the file that corresponds to the data
        PAs: Array of N for the parallactic angle rotation of the target (used for ADI) [in degrees]
        wvs: Array of N wavelengths of the images (used for SDI) [in microns]. For polarization data, defaults to "None"
        IWA: a floating point scalar (not array). Specifies to inner working angle in pixels
        output: Array of shape (b, len(files), len(uniq_wvs), y, x) where b is the number of different KL basis cutoffs
        psfs: Spectral cube of size (Nwv, psfy, psfx) where psf_cube_size defines the size of psfy, psfx.
        psf_center: [x, y] location of the center of the PSF for a frame in self.psfs 
        flipx: True by default. Determines whether a relfection about the x axis is necessary to rotate image North-up East left
        nfiles: number of datacubes
        nwvs: number of wavelengths

    """
    # class initialization
    # Astrometric calibration: Maire et al. 2016
    north_offset = -102.18 # who knows on the sign on this angle
    platescale = 0.007462

    # Coonstructor
    def __init__(self, data_cube, psf_cube, info_fits, wavelength_info, psf_cube_size=21, nan_mask_boxsize=9,
                 IWA=0.15):
        super(Ifs, self).__init__()
        # read in the data
        with fits.open(data_cube) as hdulist:
            self._input = hdulist[0].data # 4D cube, Nfiles, Nwvs, Ny, Nx
            self._filenums = np.repeat(np.arange(self.input.shape[0]), self.input.shape[1])
            self.nfiles = self.input.shape[0]
            self.nwvs = self.input.shape[1]
            # collapse files with wavelengths
            self.input = self.input.reshape(self.nfiles*self.nwvs, self.input.shape[2],
                                            self.input.shape[3])
            # zeros are nans, and anything adjacient to a pixel less than zero is 0.
            input_minfilter = ndimage.minimum_filter(self.input, (0, nan_mask_boxsize, nan_mask_boxsize))
            self.input[np.where(input_minfilter <= 0)] = np.nan

            # centers are at dim/2
            self._centers = np.array([[img.shape[1]/2., img.shape[0]/2.] for img in self.input])

        # read in the psf cube
        with fits.open(psf_cube) as hdulist:
            self.psfs = hdulist[0].data # Nwvs, Ny, Nx
            self.psfs_center = [self.psfs.shape[2]//2, self.psfs.shape[1]//2] # (x,y)

            # trim the cube
            pixelsbefore = psf_cube_size//2
            pixelsafter = psf_cube_size - pixelsbefore
            self.psfs = np.copy(self.psfs[:, self.psfs_center[1]-pixelsbefore:self.psfs_center[1]+pixelsafter,
                                            self.psfs_center[0]-pixelsbefore:self.psfs_center[0]+pixelsafter])
            self.psfs_center = [psf_cube_size//2, psf_cube_size//2]

        # read in wavelength solution
        with fits.open(wavelength_info) as hdulist:
            self._wvs = hdulist[0].data
            # repeat for all Nfile cubes
            self._wvs = np.tile(self.wvs, self.nfiles)

        # read in PA info among other things
        with fits.open(info_fits) as hdulist:
            metadata = hdulist[1].data
            self._PAs = np.repeat(metadata["PA"] + metadata['PUPOFF'], self.nwvs)
            self._filenames = np.repeat(metadata["FILE"], self.nwvs)

        # we don't need to flip x for North Up East left
        self.flipx = False

        # I have no idea
        self.IWA = IWA / Ifs.platescale # 0.15" IWA

        # We aren't doing WCS info for SPHERE
        self.wcs = np.array([None for _ in range(self.nfiles * self.nwvs)])

        self._output = None


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

    def readdata(self, filepaths):
        """
        Reads in the data from the files in the filelist and writes them to fields
        """
        pass


    def savedata(self, filepath, data, klipparams=None, filetype="", zaxis=None , more_keywords=None):
        """
        Save SPHERE Data.

        Args:
filepath: path to file to output
            data: 2D or 3D data to save
            klipparams: a string of klip parameters
            filetype: filetype of the object (e.g. "KL Mode Cube", "PSF Subtracted Spectral Cube")
            zaxis: a list of values for the zaxis of the datacub (for KL mode cubes currently)
            more_keywords (dictionary) : a dictionary {key: value, key:value} of header keywords and values which will
                                         written into the primary header
        
        """
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(data=data))

        # save all the files we used in the reduction
        # we'll assume you used all the input files
        # remove duplicates from list
        filenames = np.unique(self.filenames)
        nfiles = np.size(filenames)
        hdulist[0].header["DRPNFILE"] = (nfiles, "Num raw files used in pyKLIP")
        for i, filename in enumerate(filenames):
            hdulist[0].header["FILE_{0}".format(i)] = filename + '.fits'



        # write out psf subtraction parameters
        # get pyKLIP revision number
        pykliproot = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        # the universal_newline argument is just so python3 returns a string instead of bytes
        # this will probably come to bite me later
        try:
            pyklipver = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=pykliproot, universal_newlines=True).strip()
        except:
            pyklipver = "unknown"
        hdulist[0].header['PSFSUB'] = ("pyKLIP", "PSF Subtraction Algo")
        hdulist[0].header.add_history("Reduced with pyKLIP using commit {0}".format(pyklipver))
        hdulist[0].header['CREATOR'] = "pyKLIP-{0}".format(pyklipver)

        # store commit number for pyklip
        hdulist[0].header['pyklipv'] = (pyklipver, "pyKLIP version that was used")

        if klipparams is not None:
            hdulist[0].header['PSFPARAM'] = (klipparams, "KLIP parameters")
            hdulist[0].header.add_history("pyKLIP reduction with parameters {0}".format(klipparams))


        # write z axis units if necessary
        if zaxis is not None:
            # Writing a KL mode Cube
            if "KL Mode" in filetype:
                hdulist[0].header['CTYPE3'] = 'KLMODES'
                # write them individually
                for i, klmode in enumerate(zaxis):
                    hdulist[0].header['KLMODE{0}'.format(i)] = (klmode, "KL Mode of slice {0}".format(i))
                hdulist[0].header['CUNIT3'] = "N/A"
                hdulist[0].header['CRVAL3'] = 1
                hdulist[0].header['CRPIX3'] = 1.
                hdulist[0].header['CD3_3'] = 1.

        if "Spectral" in filetype:
            uniquewvs = np.unique(self.wvs)
            # do spectral stuff instead
            # because wavelength solutoin is nonlinear, we're not going to store it here
            hdulist[0].header['CTYPE3'] = 'WAVE'
            hdulist[0].header['CUNIT3'] = "N/A"
            hdulist[0].header['CRPIX3'] = 1.
            hdulist[0].header['CRVAL3'] = 0
            hdulist[0].header['CD3_3'] = 1
            # write it out instead
            for i, wv in enumerate(uniquewvs):
                hdulist[0].header['WV{0}'.format(i)] = (wv, "Wavelength of slice {0}".format(i))

        center = self.centers[0]
        hdulist[0].header.update({'PSFCENTX': center[0], 'PSFCENTY': center[1]})
        hdulist[0].header.update({'CRPIX1': center[0], 'CRPIX2': center[1]})
        hdulist[0].header.add_history("Image recentered to {0}".format(str(center)))

        hdulist.writeto(filepath, overwrite=True)
        hdulist.close()

    def calibrate_output(self, img, spectral=False, units="contrast"):
        """
       Calibrates the flux of an output image. Can either be a broadband image or a spectral cube depending
        on if the spectral flag is set.

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
        return img


class Irdis(Data):
    """
    A sequence of SPHERE IRDIS Data.

    Args:
        data_cube: FITS file with a 4D-cube (Nfiles, Nwvs, Ny, Nx) with all IFS coronagraphic data
        psf_cube: FITS file with a 3-D (Nwvs, Ny, Nx) PSF cube
        info_fits: FITS file with a table in the 1st ext hdr with parallactic angle info
        wavelength_str: string to specifiy the band (e.g. "H2H3", "K1K2")
        psf_cube_size: size of the psf cube to save (length along 1 dimension)
        IWA: inner working angle of the data in arcsecs

    Attributes:
        input: Array of shape (N,y,x) for N images of shape (y,x)
        centers: Array of shape (N,2) for N centers in the format [x_cent, y_cent]
        filenums: Array of size N for the numerical index to map data to file that was passed in
        filenames: Array of size N for the actual filepath of the file that corresponds to the data
        PAs: Array of N for the parallactic angle rotation of the target (used for ADI) [in degrees]
        wvs: Array of N wavelengths of the images (used for SDI) [in microns]. For polarization data, defaults to "None"
        IWA: a floating point scalar (not array). Specifies to inner working angle in pixels
        output: Array of shape (b, len(files), len(uniq_wvs), y, x) where b is the number of different KL basis cutoffs
        psfs: Spectral cube of size (2, psfy, psfx) where psf_cube_size defines the size of psfy, psfx.
        psf_center: [x, y] location of the center of the PSF for a frame in self.psfs 
        flipx: True by default. Determines whether a relfection about the x axis is necessary to rotate image North-up East left
        nfiles: number of datacubes
        nwvs: number of wavelengths (i.e. 2 for dual band imaging)
    """
    # class initialization
    # Astrometric calibration: Maire et al. 2016
    north_offset = -1.75 # who knows on the sign on this angle
    platescale = 0.012255
    # dual band imaging central wavelengths
    wavelengths = {"Y2Y3" : (1.02, 1.073), "J2J3": (1.190, 1.270), "H2H3": (1.587, 1.667),
                   "H3H4": (1.667, 1.731), "K1K2": (2.1, 2.244)}

    # Coonstructor
    def __init__(self, data_cube, psf_cube, info_fits, wavelength_str, psf_cube_size=21, IWA=0.2):
        super(Irdis, self).__init__()
        # read in the data
        with fits.open(data_cube) as hdulist:
            self._input = hdulist[0].data # 4D cube, Nfiles, Nwvs, Ny, Nx
            self._filenums = np.repeat(np.arange(self.input.shape[0]), self.input.shape[1])
            self.nfiles = self.input.shape[0]
            self.nwvs = self.input.shape[1]
            # collapse files with wavelengths
            self.input = self.input.reshape(self.nfiles*self.nwvs, self.input.shape[2],
                                            self.input.shape[3])
            # centers are at dim/2
            self._centers = np.array([[img.shape[1]/2., img.shape[0]/2.] for img in self.input])

        # read in the psf cube
        with fits.open(psf_cube) as hdulist:
            self.psfs = hdulist[0].data # Nwvs, Ny, Nx
            if np.size(self.psfs.shape) == 4:
                # multiple PSF sequences were taken. Collpase them and take the average
                self.psfs = np.nanmean(self.psfs, axis=0)
            self.psfs_center = [self.psfs.shape[2]//2, self.psfs.shape[1]//2] # (x,y)

            # trim the cube
            pixelsbefore = psf_cube_size//2
            pixelsafter = psf_cube_size - pixelsbefore
            self.psfs = np.copy(self.psfs[:, self.psfs_center[1]-pixelsbefore:self.psfs_center[1]+pixelsafter,
                                            self.psfs_center[0]-pixelsbefore:self.psfs_center[0]+pixelsafter])
            self.psfs_center = [psf_cube_size//2, psf_cube_size//2]

        db_wvs = Irdis.wavelengths[wavelength_str]
        self._wvs = np.tile(db_wvs, self.nfiles)

        # read in PA info among other things
        with fits.open(info_fits) as hdulist:
            metadata = hdulist[1].data
            self._PAs = np.repeat(metadata["PA"] + metadata['PUPOFF'], self.nwvs)
            self._filenames = np.repeat(metadata["FILE"], self.nwvs)

        # we don't need to flip x for North Up East left
        self.flipx = False

        # I have no idea
        self.IWA = IWA / Ifs.platescale # 0.2" IWA

        # We aren't doing WCS info for SPHERE
        self.wcs = np.array([None for _ in range(self.nfiles * self.nwvs)])

        self._output = None


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

    def readdata(self, filepaths):
        """
        Reads in the data from the files in the filelist and writes them to fields
        """
        pass


    def savedata(self, filepath, data, klipparams=None, filetype="", zaxis=None , more_keywords=None):
        """
        Save SPHERE Data.

        Args:
filepath: path to file to output
            data: 2D or 3D data to save
            klipparams: a string of klip parameters
            filetype: filetype of the object (e.g. "KL Mode Cube", "PSF Subtracted Spectral Cube")
            zaxis: a list of values for the zaxis of the datacub (for KL mode cubes currently)
            more_keywords (dictionary) : a dictionary {key: value, key:value} of header keywords and values which will
                                         written into the primary header

        """
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(data=data))

        # save all the files we used in the reduction
        # we'll assume you used all the input files
        # remove duplicates from list
        filenames = np.unique(self.filenames)
        nfiles = np.size(filenames)
        hdulist[0].header["DRPNFILE"] = (nfiles, "Num raw files used in pyKLIP")
        for i, filename in enumerate(filenames):
            hdulist[0].header["FILE_{0}".format(i)] = filename + '.fits'



        # write out psf subtraction parameters
        # get pyKLIP revision number
        pykliproot = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        # the universal_newline argument is just so python3 returns a string instead of bytes
        # this will probably come to bite me later
        try:
            pyklipver = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=pykliproot, universal_newlines=True).strip()
        except:
            pyklipver = "unknown"
        hdulist[0].header['PSFSUB'] = ("pyKLIP", "PSF Subtraction Algo")
        hdulist[0].header.add_history("Reduced with pyKLIP using commit {0}".format(pyklipver))
        hdulist[0].header['CREATOR'] = "pyKLIP-{0}".format(pyklipver)

        # store commit number for pyklip
        hdulist[0].header['pyklipv'] = (pyklipver, "pyKLIP version that was used")

        if klipparams is not None:
            hdulist[0].header['PSFPARAM'] = (klipparams, "KLIP parameters")
            hdulist[0].header.add_history("pyKLIP reduction with parameters {0}".format(klipparams))


        # write z axis units if necessary
        if zaxis is not None:
            # Writing a KL mode Cube
            if "KL Mode" in filetype:
                hdulist[0].header['CTYPE3'] = 'KLMODES'
                # write them individually
                for i, klmode in enumerate(zaxis):
                    hdulist[0].header['KLMODE{0}'.format(i)] = (klmode, "KL Mode of slice {0}".format(i))
                hdulist[0].header['CUNIT3'] = "N/A"
                hdulist[0].header['CRVAL3'] = 1
                hdulist[0].header['CRPIX3'] = 1.
                hdulist[0].header['CD3_3'] = 1.

        if "Spectral" in filetype:
            uniquewvs = np.sort(np.unique(self.wvs))
            # do spectral stuff instead
            # because wavelength solutoin is nonlinear, we're not going to store it here
            hdulist[0].header['CTYPE3'] = 'WAVE'
            hdulist[0].header['CUNIT3'] = "microns"
            hdulist[0].header['CRPIX3'] = 1.
            # there's only 2 wvs in DBI
            hdulist[0].header['CRVAL3'] = uniquewvs[0]
            hdulist[0].header['CD3_3'] = uniquewvs[1] - uniquewvs[0]
            # write it out instead

        center = self.centers[0]
        hdulist[0].header.update({'PSFCENTX': center[0], 'PSFCENTY': center[1]})
        hdulist[0].header.update({'CRPIX1': center[0], 'CRPIX2': center[1]})
        hdulist[0].header.add_history("Image recentered to {0}".format(str(center)))

        hdulist.writeto(filepath, overwrite=True)
        hdulist.close()

    def calibrate_output(self, img, spectral=False, units="contrast"):
        """
       Calibrates the flux of an output image. Can either be a broadband image or a spectral cube depending
        on if the spectral flag is set.

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
        return img
