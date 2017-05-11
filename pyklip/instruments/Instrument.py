import abc
import os
import subprocess
import astropy.io.fits as fits
import numpy as np

class Data(object):
    """
    Abstract Class with the required fields and methods that need to be implemented

    Attributes:
        input: Array of shape (N,y,x) for N images of shape (y,x)
        centers: Array of shape (N,2) for N centers in the format [x_cent, y_cent]
        filenums: Array of size N for the numerical index to map data to file that was passed in
        filenames: Array of size N for the actual filepath of the file that corresponds to the data
        PAs: Array of N for the parallactic angle rotation of the target (used for ADI) [in degrees]
        wvs: Array of N wavelengths of the images (used for SDI) [in microns]. For polarization data, defaults to "None"
        wcs: Array of N wcs astormetry headers for each image.
        IWA: a floating point scalar (not array). Specifies to inner working angle in pixels
        OWA: (optional) specifies outer working angle in pixels
        output: Array of shape (b, len(files), len(uniq_wvs), y, x) where b is the number of different KL basis cutoffs
        creator: (optional) string for creator of the data (used to identify pipelines that call pyklip)
        klipparams: (optional) a string that saves the most recent KLIP parameters
        flipx: (optional) True by default. Determines whether a relfection about the x axis is necessary to rotate image North-up East left


    Methods:
        readdata(): reread in the dadta
        savedata(): save a specified data in the GPI datacube format (in the 1st extension header)
        calibrate_output(): flux calibrate the output data
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        # set field for the creator of the data (used for pipeline work)
        self.creator = None
        # set field for klip parameters
        self.klipparams = None
        # set the outer working angle (optional parameter)
        self.OWA = None
        # determine whether a reflection is needed for North-up East-left (optional)
        self.flipx = True


    ###################################
    ### Required Instance Variances ###
    ###################################

    #Note that each field has a getter and setter method so by default they are all read/write

    @abc.abstractproperty
    def input(self):
        """
        Input Data. Shape of (N, y, x)
        """
        return
    @input.setter
    def input(self, newval):
        return

    @abc.abstractproperty
    def centers(self):
        """
        Image centers. Shape of (N, 2) where the 2nd dimension is [x,y] pixel coordinate (in that order)
        """
        return
    @centers.setter
    def centers(self, newval):
        return

    @abc.abstractproperty
    def filenums(self):
        """
        Array of size N for the numerical index to map data to file that was passed in
        """
        return
    @filenums.setter
    def filenums(self, newval):
        return

    @abc.abstractproperty
    def filenames(self):
        """
        Array of size N for the actual filepath of the file that corresponds to the data
        """
        return
    @filenames.setter
    def filenames(self, newval):
        return


    @abc.abstractproperty
    def PAs(self):
        """
        Array of N for the parallactic angle rotation of the target (used for ADI) [in degrees]
        """
        return
    @PAs.setter
    def PAs(self, newval):
        return


    @abc.abstractproperty
    def wvs(self):
        """
        Array of N wavelengths (used for SDI) [in microns]. For polarization data, defaults to "None"
        """
        return
    @wvs.setter
    def wvs(self, newval):
        return


    @abc.abstractproperty
    def wcs(self):
        """
        Array of N wcs astormetry headers for each image.
        """
        return
    @wcs.setter
    def wcs(self, newval):
        return


    @abc.abstractproperty
    def IWA(self):
        """
        a floating point scalar (not array). Specifies to inner working angle in pixels
        """
        return
    @IWA.setter
    def IWA(self, newval):
        return


    @abc.abstractproperty
    def output(self):
        """
        Array of shape (b, len(files), len(uniq_wvs), y, x) where b is the number of different KL basis cutoffs
        """
        return
    @output.setter
    def output(self, newval):
        return



    ########################
    ### Required Methods ###
    ########################
    @abc.abstractmethod
    def readdata(self, filepaths):
        """
        Reads in the data from the files in the filelist and writes them to fields
        """
        return NotImplementedError("Subclass needs to implement this!")

    @staticmethod
    @abc.abstractmethod
    def savedata(self, filepath, data, klipparams=None, filetype="", zaxis=None, more_keywords=None):
        """
        Saves data for this instrument

        Args:
            filepath: filepath to save to
            data: data to save
            klipparams: a string of KLIP parameters. Write it to the 'PSFPARAM' keyword
            filtype: type of file (e.g. "KL Mode Cube", "PSF Subtracted Spectral Cube"). Wrriten to 'FILETYPE' keyword
            zaxis: a list of values for the zaxis of the datacub (for KL mode cubes currently)
            more_keywords (dictionary) : a dictionary {key: value, key:value} of header keywords and values which will
                                         written into the primary header
        """
        return NotImplementedError("Subclass needs to implement this!")

    @abc.abstractmethod
    def calibrate_output(self, img, spectral=False):
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

        Return:
            calib_img: calibrated image of the same shape
        """
        return NotImplementedError("Subclass needs to implement this!")


class GenericData(Data):
    """
    Basic class to interface with a basic direct imaging dataset

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


    Args:
        input_data: either a 1-D list of filenames to read in, or a 3-D cube of all data (N, y, x)
        centers: array of shape (N,2) for N centers in the format [x_cent, y_cent]
        parangs: Array of N for the parallactic angle rotation of the target (used for ADI) [in degrees]
        wvs: Array of N wavelengths of the images (used for SDI) [in microns]. For polarization data, defaults to "None"
        IWA: a floating point scalar (not array). Specifies to inner working angle in pixels
        filenames: Array of size N for the actual filepath of the file that corresponds to the data
    """
    # Coonstructor
    def __init__(self, input_data, centers, parangs=None, wvs=None, IWA=0, filenames=None):
        super(GenericData, self).__init__()
        # read in the data
        if np.array(input_data).ndim == 1:
            self._input = self.readdata(input_data)
        else:
            # assume this is a 3-D cube
            self._input = np.array(input_data)
        
        nfiles = self.input.shape[0]

        self.centers = np.array(centers)

        if self.centers.shape [0] != nfiles:
            raise ValueError("Input data has shape {0} but centers has shape {1}".format(self.input.shape,
                                                                                         self.centers.shape))

        if parangs is not None:
            self._PAs = parangs
        else:
            self._PAs = np.zeros(nfiles)

        if wvs is not None:
            self._wvs = wvs
        else:
            self._wvs = np.ones(nfiles)

        self.IWA = IWA

        if filenames is not None:
            self._filenames = filenames
            unique_filenames = np.unique(filenames)                                                                                 
            self._filenums = np.array([np.argwhere(filename == unique_filenames).ravel()[0] for filename in filenames])
        else:
            self._filenums = np.arange(nfiles)
            self._filenames = np.array(["{0}".format(i) for i in self.filenums])

        self._wcs = np.array([None for _ in range(nfiles)])

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
 

    def readdata(self, filepaths):
        """
        Reads in the data from the files in the filelist and writes them to fields.
        """
        input_data = []
        for filename in filepaths:
            with fits.open(filename) as hdulist:
                # assume the data is in the primary header
                data = hdulist[0].data
                # if this data has more than 2-D, collapse the Data
                dims = data.shape
                if np.size(dims) > 2:
                    nframes = np.prod(dims[:-2])
                    # collapse in all dimensions except y and x
                    data.shape = (nframes, dims[-2], dims[-1])

                input_data.append(data)

        # collapse data again
        input_data = np.array(input_data)
        dims = input_data.shape
        if np.szie(dims) > 3:
            nframes = np.prod(dims[:-2])
            # collapse in all dimensions except y and x
            input_data.shape = (nframes, dims[-2], dims[-1])


    def savedata(self, filepath, data, klipparams=None, filetype="", zaxis=None, more_keywords=None):
        """
        Saves data for this instrument

        Args:
            filepath: filepath to save to
            data: data to save
            klipparams: a string of KLIP parameters. Write it to the 'PSFPARAM' keyword
            filtype: type of file (e.g. "KL Mode Cube", "PSF Subtracted Spectral Cube"). Wrriten to 'FILETYPE' keyword
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

    def calibrate_output(self, img, spectral=False):
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

        Return:
            calib_img: calibrated image of the same shape
        """
        return img