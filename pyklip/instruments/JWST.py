from pyklip.instruments.Instrument import Data
import pyklip.rdi as rdi
import pyklip.klip

from astropy.io import fits
from astropy import wcs
from astroquery.svo_fps import SvoFps
from scipy.ndimage import fourier_shift, median_filter, shift
from scipy.optimize import leastsq
from skimage.registration import phase_cross_correlation

import numpy as np
import os, shutil
import matplotlib.pyplot as plt
import time
import copy 

import matplotlib
matplotlib.rc('font', serif='DejaVu Sans')
matplotlib.rcParams.update({'font.size': 14})
import matplotlib.patheffects as PathEffects

class JWSTData(Data):
    """
    Class to interpret JWST data using pyKLIP.
    """
    ############################
    ### Class Initialization ###
    ############################

    ####################
    ### Constructors ###
    ####################
    def __init__(self, filepaths=None, psflib_filepaths=None, centering='jwstpipe', badpix_threshold=0.2):

        # Initialize the super class
        super(JWSTData, self).__init__()

        # Mean wavelengths of the JWST filters from the SVO Filter Profile
        # Service
        self.wave = {}
        filter_list = SvoFps.get_filter_list(facility='JWST', instrument='NIRCAM')
        for i in range(len(filter_list)):
            name = filter_list['filterID'][i]
            name = name[name.rfind('.')+1:]
            self.wave[name] = filter_list['WavelengthMean'][i]/1e4 # micron
        filter_list = SvoFps.get_filter_list(facility='JWST', instrument='MIRI')
        for i in range(len(filter_list)):
            name = filter_list['filterID'][i]
            name = name[name.rfind('.')+1:]
            self.wave[name] = filter_list['WavelengthMean'][i]/1e4 # micron
        del filter_list

        # Get the target dataset
        reference = self.readdata(filepaths, centering, badpix_threshold)

        # If necessary, get the PSF library dataset for RDI procedures
        if psflib_filepaths != None:
            self.readpsflib(psflib_filepaths, centering, badpix_threshold, reference)
        else:
            self._psflib = None

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

    @property
    def psflib(self):
        return self._psflib
    @psflib.setter
    def psflib(self, newval):
        self._psflib = newval

    ###############
    ### Methods ###
    ###############

    def readdata(self, filepaths, centering='basic', badpix_threshold=0.2, verbose=False):
        """
        Method to open and read JWST data.

        Parameters
        ----------
        filepaths : list of str
            A list of file paths.
        centering : str
            String descriptor for method to estimate image centers.
        verbose : bool
            Boolean for terminal print statements.

        Returns
        -------
        reference : array
            A 2D image to be aligned to for readpsflib.
        """
        # Ensure we have a list of file paths
        if isinstance(filepaths, str):
            filepaths = [filepaths]
            if verbose:
                print('Only 1 file path was provided, are you sure you meant to do this?')

        # Check list is not empty
        if len(filepaths) == 0:
            raise ValueError('Empty filepath list provided to JWSTData!')

        # Intialize some arrays
        data = []
        pxdq = []
        centers = []
        filenames = []
        pas = []
        wvs = []
        wcs_hdrs = []

        # Go through files one by one
        fiducial_point_override = False
        for index, file in enumerate(filepaths):
            with fits.open(file) as f:

                # Let's figure out what instrument this is
                inst = f[0].header['INSTRUME']

                # Grab number of integrations, as this is how many images are
                # in the file
                nints = f[0].header['NINTS']
                pixel_scale = np.sqrt(f['SCI'].header['PIXAR_A2']) # arcsec; need this for later to calculate IWA

                sci_data = f['SCI'].data
                dq_data = f['DQ'].data

                # NIRCam specifics
                if inst == 'NIRCAM':
                    img_centers = [f['SCI'].header['CRPIX1']-1, f['SCI'].header['CRPIX2']-1]*nints
                    # Check for fiducial point override
                    if 'NARROW' in f[0].header['APERNAME'] :
                        fiducial_point_override = True
                # MIRI specifics
                elif inst == 'MIRI':
                    filt = f[0].header['FILTER']
                    # Cut out the unilluminated pixels
                    all_data, trim = trim_miri_data([sci_data, dq_data], filt)
                    sci_data, dq_data = all_data[0], all_data[1]

                    if filt == 'F1065C':
                        img_centers = [float(121-trim[0]), float(113-trim[1])]*nints
                    elif filt == 'F1140C':
                        img_centers = [float(114-trim[0]), float(116-trim[1])]*nints
                    elif filt == 'F1550C':
                        img_centers = [float(116-trim[0]), float(114-trim[1])]*nints
                    else:
                        raise ValueError('pyKLIP only currently supports F1065C/F1140C/F1550C MIRI data')

                    # plt.imshow(sci_data[0])
                    # plt.scatter([116-trim[0]], [114-trim[1]])
                    # plt.show()

                # Get the images
                data.append(sci_data)
                pxdq.append(dq_data)
                centers.append(img_centers) # header keywords are in 1-indexed coordinates

                # Assign filenames based on the file and the integration
                filenames += ['{}_INT{}'.format(file.split('/')[-1], i+1) for i in range(nints)]

                # Get PA for all frame withing the file
                pas.append([f['SCI'].header['ROLL_REF']]*nints)

                # Get the filter wavelength, should be the same for each file
                # though (for JWST)
                filt = f[0].header['FILTER']
                wave = self.wave[filt] # micron
                wvs.append([wave]*nints)

                # Get WCS information
                wcs_hdr = wcs.WCS(header=f['SCI'].header, naxis=f['SCI'].header['WCSAXES'])
                for i in range(nints):
                    wcs_hdrs.append(wcs_hdr.deepcopy())


        # Convert to numpy arrays and collapse integrations along a single axis
        data = np.concatenate(data)
        pxdq = np.concatenate(pxdq)

        centers = np.concatenate(centers).reshape(-1, 2)
        filenames = np.array(filenames)
        filenums = np.array(range(len(filenames)))
        pas = np.array(pas).flatten()
        wvs = np.array(wvs).flatten()

        # Fix bad pixels, reject frames with more than 1% bad pixels
        frac = np.sum(pxdq != 0, axis=(1, 2))/np.prod(pxdq.shape[1:])

        good = frac <= badpix_threshold
        data = data[good]
        pxdq = pxdq[good]
        centers = centers[good]
        filenames = filenames[good]
        filenums = filenums[good]
        pas = pas[good]
        wvs = wvs[good]
        data = self.fix_bad_pixels(data, pxdq)

        # f = plt.figure()
        # ax = plt.gca()
        # ax.plot(frac*100)
        # ax.axhline(badpix_threshold*100, color='red', label='threshold = %.0f%%' % (badpix_threshold*100.))
        # tt = ax.text(0.01, 0.99, 'Rejected %.0f of %.0f images' % (len(good)-np.sum(good), len(good)), ha='left', va='top', color='black', transform=ax.transAxes, size=12)
        # tt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
        # ax.set_ylim([0., 100.])
        # ax.set_xlabel('Image index')
        # ax.set_ylabel('Bad pixel fraction [%]')
        # plt.legend(loc='upper right')
        # plt.title('Science image rejection')
        # plt.tight_layout()
        # plt.savefig('bpfix_sci.pdf')
        # plt.show()

        print('--> Rejected %.0f of %.0f images due to too many bad pixels (threshold = %.0f%%)' % (len(good)-np.sum(good), len(good), badpix_threshold*100.))

        # Get image centers based on desired algorithm
        if centering == 'basic':
            reference = data[0].copy() # align to first science image
        else:
            # Need to subtract residual background so that image registration
            # does not fit to it
            data_medsub = data-np.median(data, axis=(1, 2), keepdims=True)
            # For MIRI, let's only look at the central region
            # if inst == 'MIRI':
            #     _, nx, ny = data_medsub.shape
            #     data_medsub = data_medsub[:,int(nx/4):int(3*nx/4),int(ny/4):int(3*ny/4)]
            reference = data_medsub[0].copy() # align to first science image

            tstart = time.time()
            if centering == 'jwstpipe':
                shifts, res_before, res_after = self.align_jwstpipe(reference, data_medsub[1:])
            elif centering == 'imageregis':
                shifts, res_before, res_after = self.align_imageregis(reference, data_medsub[1:])
            else:
                raise ValueError('Unknown centering algorithm')
            tend = time.time()
            centers[1:] -= shifts[:, :2]

            # f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
            # ax[0].plot(res_before, label='before align')
            # ax[0].plot(res_after, label='after align')
            # ax[0].grid(axis='y')
            # ax[0].set_xlabel('Image index')
            # ax[0].set_ylabel(r'$\Sigma$(residual${}^2$)')
            # ax[0].legend(loc='upper center')
            # ax[0].set_title('Residual reference-image', y=1., pad=10, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            # temp = np.unique(pas[1:])
            # for i in range(len(temp)):
            #     ww = pas[1:] == temp[i]
            #     ax[1].scatter(shifts[ww, 0]*pixel_scale*1000., shifts[ww, 1]*pixel_scale*1000., label='PA = %.0f deg' % temp[i])
            # ax[1].axis('square')
            # xlim = ax[1].get_xlim()
            # temp = xlim[1]-xlim[0]
            # ax[1].set_xlim([xlim[0]-0.35*temp, xlim[1]+0.35*temp])
            # ax[1].grid(axis='both')
            # ax[1].set_xlabel('Image x-shift [mas]')
            # ax[1].set_ylabel('Image y-shift [mas]')
            # ax[1].legend(loc='center right')
            # ax[1].set_title('Image shifts', y=1., pad=10, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            # plt.suptitle('Science image alignment -- '+centering+' method (%.0f s runtime)' % (tend-tstart))
            # plt.tight_layout()
            # plt.savefig(centering+'_sci.pdf')
            # # plt.savefig(centering+'_sci_bad.pdf')
            # plt.show()

        # Need to align the images so that they have the same centers
        image_center = np.array(data[0].shape)/2.
        for i, image in enumerate(data):
            recentered_image = pyklip.klip.align_and_scale(image, new_center=image_center, old_center=centers[i])
            data[i] = recentered_image
            centers[i] = image_center

        # Assume an inner working angle of 1 lambda/D
        if fiducial_point_override:
            IWA = 1. # pix
        else:
            lambda_d_arcsec = ((wvs[0]/1e6)/6.5)*(180./np.pi)*3600.
            IWA = lambda_d_arcsec/pixel_scale # pix

        # Assign all necessary properties
        self._input = data
        self._centers = centers
        self._filenums = filenums
        self._filenames = filenames
        self._PAs = pas
        self._wvs = wvs
        self._wcs = wcs_hdrs
        self._IWA = IWA
        return reference

    def readpsflib(self, psflib_filepaths, centering='basic', badpix_threshold=0.2, reference=None, verbose=False):
        """
        Method to open and read JWST data for use as part of a PSF library.

        Parameters
        ----------
        psflib_filepaths : list of str
            A list of file paths.
        centering : str
            String descriptor for method to estimate image centers.
        reference : array
            A 2D image to be aligned to.
        verbose : bool
            Boolean for terminal print statements.

        Returns
        -------
        None, data are saved to a JWSTData object.
        """
        # Ensure we have a list of file paths
        if isinstance(psflib_filepaths, str):
            psflib_filepaths = [psflib_filepaths]
            if verbose:
                print('Only 1 psflib filepath was provided, are you sure you meant to do this?')

        # Check list is not empty
        if len(psflib_filepaths) == 0:
            raise ValueError('Empty psflib filepath list provided to JWSTData!')

        # Intialize some arrays
        psflib_data = []
        psflib_pxdq = []
        psflib_offsets = []
        psflib_centers = []
        psflib_filenames = []

        # Prepare reference data for RDI subtractions
        for index, file in enumerate(psflib_filepaths):
            with fits.open(file) as f:

                inst = f[0].header['INSTRUME']

                # Grab number of integrations, as this is how many images are
                # in the file
                nints = f[0].header['NINTS']
                pixel_scale = np.sqrt(f['SCI'].header['PIXAR_A2']) # arcsec; need this for later to calculate IWA

                sci_data = f['SCI'].data
                dq_data = f['DQ'].data

                # NIRCam specifics
                if inst == 'NIRCAM':
                    center = [f['SCI'].header['CRPIX1']-1, f['SCI'].header['CRPIX2']-1]
                # MIRI specifics
                elif inst == 'MIRI':
                    filt = f[0].header['FILTER']
                    # Cut out the unilluminated pixels
                    all_data, trim = trim_miri_data([sci_data, dq_data], filt)
                    sci_data, dq_data = all_data[0], all_data[1]

                    if filt == 'F1065C':
                        center = [float(121-trim[0]), float(113-trim[1])]
                    elif filt == 'F1140C':
                        center = [float(114-trim[0]), float(116-trim[1])]
                    elif filt == 'F1550C':
                        center = [float(116-trim[0]), float(114-trim[1])]
                    else:
                        raise ValueError('pyKLIP only currently supports F1065C/F1140C/F1550C MIRI data')

                # Get the images
                psflib_data.append(sci_data)
                psflib_pxdq.append(dq_data)

                # Get the known offset between images
                offset = [f[0].header['XOFFSET'], f[0].header['YOFFSET']]/pixel_scale # pix
                # if inst == 'NIRCAM':
                #     offset = [f[0].header['XOFFSET'], f[0].header['YOFFSET']]/pixel_scale # pix
                # elif inst == 'MIRI':
                #     offset = get_miri_offset(file) / pixel_scale

                psflib_offsets.append(offset.tolist()*nints)
                if centering == 'basic':
                    psflib_centers.append([sum(x) for x in zip(center, offset)]*nints)
                else:
                    psflib_centers.append([sum(x) for x in zip(center, [0., 0.])]*nints) # dither will be detected using image registration

                # Assign filenames based on the file and the integration
                psflib_filenames += ['{}_INT{}'.format(file.split('/')[-1], i+1) for i in range(nints)]

        # Convert to numpy arrays and collapse along integration axis
        psflib_data = np.concatenate(psflib_data)
        psflib_pxdq = np.concatenate(psflib_pxdq)
        psflib_offsets = np.concatenate(psflib_offsets).reshape(-1, 2)
        psflib_centers = np.concatenate(psflib_centers).reshape(-1, 2)
        psflib_filenames = np.array(psflib_filenames)

        # Fix bad pixels, reject frames with more than 1% bad pixels
        frac = np.sum(psflib_pxdq != 0, axis=(1, 2))/np.prod(psflib_pxdq.shape[1:])
        good = frac <= badpix_threshold
        psflib_data = psflib_data[good]
        psflib_pxdq = psflib_pxdq[good]
        psflib_offsets = psflib_offsets[good]
        psflib_centers = psflib_centers[good]
        psflib_filenames = psflib_filenames[good]
        psflib_data = self.fix_bad_pixels(psflib_data, psflib_pxdq)

        # f = plt.figure()
        # ax = plt.gca()
        # ax.plot(frac*100)
        # ax.axhline(badpix_threshold*100, color='red', label='threshold = %.0f%%' % (badpix_threshold*100.))
        # tt = ax.text(0.01, 0.99, 'Rejected %.0f of %.0f images' % (len(good)-np.sum(good), len(good)), ha='left', va='top', color='black', transform=ax.transAxes, size=12)
        # tt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
        # ax.set_ylim([0., 100.])
        # ax.set_xlabel('Image index')
        # ax.set_ylabel('Bad pixel fraction [%]')
        # plt.legend(loc='upper right')
        # plt.title('Reference image rejection')
        # plt.tight_layout()
        # plt.savefig('bpfix_ref.pdf')
        # plt.show()

        print('--> Rejected %.0f of %.0f images due to too many bad pixels (threshold = %.0f%%)' % (len(good)-np.sum(good), len(good), badpix_threshold*100.))

        # Get image centers based on desired algorithm
        if centering == 'basic':
            pass
        else:
            if reference is None:
                raise UserWarning('Need reference for this centering algorithm')
            # Need to subtract residual background so that image registration
            # does not fit to it
            data_medsub = psflib_data-np.median(psflib_data, axis=(1, 2), keepdims=True)
            # if inst == 'MIRI':
            #     _, nx, ny = data_medsub.shape
            #     data_medsub = data_medsub[:,int(nx/4):int(3*nx/4),int(ny/4):int(3*ny/4)]
            tstart = time.time()
            if centering == 'jwstpipe':
                shifts, res_before, res_after = self.align_jwstpipe(reference, data_medsub)
            elif centering == 'imageregis':
                shifts, res_before, res_after = self.align_imageregis(reference, data_medsub)
            else:
                raise ValueError('Unknown centering algorithm')
            tend = time.time()
            psflib_centers -= shifts[:, :2]

            # f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
            # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            # ax[0].plot(res_before, label='before align')
            # ax[0].plot(res_after, label='after align')
            # ax[0].grid(axis='y')
            # ax[0].set_xlabel('Image index')
            # ax[0].set_ylabel(r'$\Sigma$(residual${}^2$)')
            # ax[0].legend(loc='upper center')
            # ax[0].set_title('Residual reference-image', y=1., pad=10, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            # temp = np.unique(psflib_offsets, axis=0)
            # for i in range(temp.shape[0]):
            #     ww = np.where((psflib_offsets == temp[i]).all(axis=1))[0]
            #     ax[1].scatter(shifts[ww, 0]*pixel_scale*1000., shifts[ww, 1]*pixel_scale*1000., c=colors[i], label='dpos %.0f' % (i+1))
            #     ax[1].plot([-temp[i, 0]*pixel_scale*1000., -temp[i, 0]*pixel_scale*1000.], [-temp[i, 1]*pixel_scale*1000.-5., -temp[i, 1]*pixel_scale*1000.+5.], color=colors[i])
            #     ax[1].plot([-temp[i, 0]*pixel_scale*1000.-5., -temp[i, 0]*pixel_scale*1000.+5.], [-temp[i, 1]*pixel_scale*1000., -temp[i, 1]*pixel_scale*1000.], color=colors[i])
            # ax[1].axis('square')
            # xlim = ax[1].get_xlim()
            # temp = xlim[1]-xlim[0]
            # ax[1].set_xlim([xlim[0]-0.35*temp, xlim[1]+0.35*temp])
            # ax[1].grid(axis='both')
            # ax[1].set_xlabel('Image x-shift [mas]')
            # ax[1].set_ylabel('Image y-shift [mas]')
            # ax[1].legend(loc='center right')
            # ax[1].set_title('Image shifts', y=1., pad=10, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
            # plt.suptitle('Reference image alignment -- '+centering+' method (%.0f s runtime)' % (tend-tstart))
            # plt.tight_layout()
            # plt.savefig(centering+'_ref.pdf')
            # # plt.savefig(centering+'_ref_bad.pdf')
            # plt.show()

        # Need to align the images so that they have the same centers
        image_center = np.array(psflib_data[0].shape)/2.
        for i, image in enumerate(psflib_data):
            recentered_image = pyklip.klip.align_and_scale(image, new_center=image_center, old_center=psflib_centers[i])
            psflib_data[i] = recentered_image

        # Append the target images as well
        psflib_data = np.append(psflib_data, self._input, axis=0)
        psflib_filenames = np.append(psflib_filenames, self._filenames, axis=0)
        psflib_centers = np.append(psflib_centers, self._centers, axis=0)

        # Create the PSF library
        psflib = rdi.PSFLibrary(psflib_data, image_center, psflib_filenames, compute_correlation=True)

        # Prepare the library with the target dataset
        psflib.prepare_library(self)

        self._psflib = psflib
        return

    def fix_bad_pixels(self,
                       data,
                       pxdq):
        """
        Fix bad pixels using a median filter.
        
        Parameters
        ----------
        data : array
            A 3D data cube of images to be fixed.
        pxdq : array
            A 3D data cube of bad pixel maps.

        Returns
        -------
        data_fixed : array
            Fixed images.
        """

        if data.ndim != 3:
            raise UserWarning('Requires 3D data cube')

        # Fix bad pixels using median filter
        data_fixed = data.copy()
        for i in range(data.shape[0]):
            bad = pxdq[i] != 0
            data_fixed[i][bad] = median_filter(data[i], size=5)[bad]

        return data_fixed

    def fourier_imshift(self,
                        image,
                        shift):
        """
        From JWST stage 3 pipeline.

        Parameters
        ----------
        image : array
            A 2D image to be shifted.
        shift : array
            xshift, yshift.

        Returns
        -------
        offset : array
            Shifted image.
        """
        if image.ndim == 2:
            shift = np.asanyarray(shift)[:2]
            offset_image = fourier_shift(np.fft.fftn(image), shift[::-1])
            offset = np.fft.ifftn(offset_image).real

        elif image.ndim == 3:
            nslices = image.shape[0]
            shift = np.asanyarray(shift)[:, :2]
            if shift.shape[0] != nslices:
                raise ValueError('The number of provided shifts must be equal to the number of slices in the input image')

            offset = np.empty_like(image, dtype=float)
            for k in range(nslices):
                offset[k] = self.fourier_imshift(image[k], shift[k])

        else:
            raise ValueError('Input image must be either a 2D or a 3D array')

        return offset

    def shift_subtract(self,
                       pp,
                       reference,
                       target,
                       mask=None):
        """
        From JWST stage 3 pipeline.

        Parameters
        ----------
        pp : tuple
            xshift, yshift, beta.
        reference : array
            A 2D image to be aligned to.
        target : array
            A 2D image to align to reference.
        mask : array
            A 2D image indicating pixels to be considered during the fit.

        Returns
        -------
        res : array
            A 1D vector containing the difference between reference and
            target.
        """
        shift = pp[:2]
        beta = pp[2]

        offset = self.fourier_imshift(target, shift)

        if mask is not None:
            return ((reference-beta*offset)*mask).ravel()
        else:
            return (reference-beta*offset).ravel()

    def align_fourierLSQ(self,
                         reference,
                         target,
                         mask=None):
        """
        From JWST stage 3 pipeline.

        Parameters
        ----------
        reference : array
            A 2D image to be aligned to.
        target : array
            A 2D image to align to reference.
        mask : array
            A 2D image indicating pixels to be considered during the fit.

        Returns
        -------
        pp : tuple
            xshift, yshift, beta.
        """
        p0 = [0., 0., 1.]
        pp = leastsq(self.shift_subtract,
                     p0,
                     args=(reference, target, mask),
                     full_output=True)

        return pp

    def align_jwstpipe(self,
                       reference,
                       data):
        """
        Align a 3D data cube of images to a reference image using the same
        algorithm as the JWST stage 3 pipeline.

        Parameters
        ----------
        reference : array
            A 2D image to be aligned to.
        target : array
            A 3D data cube of images to align to reference.

        Returns
        -------
        shifts : array
            xshift, yshift, beta to align each image.
        res_before : array
            Sum of squares of residuals between the reference and each image
            before alignment.
        res_after : array
            Sum of squares of residuals between the reference and each image
            after alignment.
        """
        if data.ndim != 3:
            raise UserWarning('Requires 3D data cube')

        shifts = []
        res_before = []
        res_after = []
        for i in range(data.shape[0]):
            pp = self.align_fourierLSQ(reference, data[i].copy())
            shifts += [pp[0]]
            res_before += [np.sum((reference-pp[0][2]*data[i])**2)]
            res_after += [np.sum(pp[2]['fvec']**2)]

            # f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
            # p0 = ax[0].imshow(reference-data[i], origin='lower')
            # plt.colorbar(p0, ax=ax[0])
            # p1 = ax[1].imshow(reference-pp[0][2]*self.fourier_imshift(data[i].copy(), pp[0][:2]), origin='lower')
            # plt.colorbar(p1, ax=ax[1])
            # plt.show()
            # import pdb; pdb.set_trace()

        return np.array(shifts), np.array(res_before), np.array(res_after)

    def scale_subtract(self,
                       pp,
                       reference,
                       target,
                       mask=None):
        """
        Scale and subtract the target from the reference while computing the
        best image shift using scikit image registration.

        Parameters
        ----------
        pp : tuple
            beta.
        reference : array
            A 2D image to be aligned to.
        target : array
            A 2D image to align to reference.
        mask : array
            A 2D image indicating pixels to be considered during the fit.

        Returns
        -------
        res : array
            A 1D vector containing the difference between reference and
            target.
        """
        shift, error, diffphase = phase_cross_correlation(reference,
                                                          pp[0]*target,
                                                          upsample_factor=100)
        shift = shift[::-1]

        offset = self.fourier_imshift(target, shift)

        if mask is not None:
            return ((reference-pp[0]*offset)*mask).ravel()
        else:
            return (reference-pp[0]*offset).ravel()

    def align_imageregisLSQ(self,
                            reference,
                            target,
                            mask=None):
        """
        Run a least squares optimization on the scaling parameter while
        computing the best image shift using scikit image registration.

        Parameters
        ----------
        reference : array
            A 2D image to be aligned to.
        target : array
            A 2D image to align to reference.
        mask : array
            A 2D image indicating pixels to be considered during the fit.

        Returns
        -------
        pp : tuple
            beta.
        """
        p0 = [1.]
        pp = leastsq(self.scale_subtract,
                     p0,
                     args=(reference, target, mask),
                     full_output=True)

        return pp

    def align_imageregis(self,
                         reference,
                         data):
        """
        Align a 3D data cube of images to a reference image using scikit image
        registration.

        Parameters
        ----------
        reference : array
            A 2D image to be aligned to.
        target : array
            A 3D data cube of images to align to reference.

        Returns
        -------
        shifts : array
            xshift, yshift, beta to align each image.
        res_before : array
            Sum of squares of residuals between the reference and each image
            before alignment.
        res_after : array
            Sum of squares of residuals between the reference and each image
            after alignment.
        """
        if data.ndim != 3:
            raise UserWarning('Requires 3D data cube')

        shifts = []
        res_before = []
        res_after = []
        for i in range(data.shape[0]):
            pp = self.align_imageregisLSQ(reference, data[i].copy())
            shift, error, diffphase = phase_cross_correlation(reference,
                                                              pp[0][0]*data[i].copy(),
                                                              upsample_factor=1000)
            shift = shift[::-1]
            shifts += [np.array([shift[0], shift[1], pp[0][0]])]
            res_before += [np.sum((reference-pp[0][0]*data[i])**2)]
            res_after += [np.sum((reference-pp[0][0]*self.fourier_imshift(data[i].copy(), shift))**2)]

            # f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
            # p0 = ax[0].imshow(reference-data[i], origin='lower')
            # plt.colorbar(p0, ax=ax[0])
            # p1 = ax[1].imshow(reference-self.fourier_imshift(data[i].copy(), shift), origin='lower')
            # plt.colorbar(p1, ax=ax[1])
            # plt.show()
            # import pdb; pdb.set_trace()

        return np.array(shifts), np.array(res_before), np.array(res_after)

    def savedata(self, filepath, data, klipparams=None, filetype='', zaxis=None, more_keywords=None):
        """
        Saves data for this instrument.

        Parameters
        ----------
        filepath : str
            Filepath to save to.
        data : array
            Data to save.
        klipparams : str
            A string of KLIP parameters. Write it to the 'PSFPARAM' keyword.
        filtype : str
            Type of file (e.g. "KL Mode Cube", "PSF Subtracted Spectral Cube").
            Written to 'FILETYPE' keyword.
        zaxis : list of int
            A list of values for the zaxis of the datacube (for KL mode cubes
            currently).
        more_keywords : dict
            A dictionary {key: value, key:value} of header keywords and values
            which will be written into the primary header.
        """
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(data=data))

        # Save all the files we used in the reduction. We'll assume you used
        # all the input files. Remove duplicates from list.
        filenames = np.unique(self.filenames)
        nfiles = np.size(filenames)
        hdulist[0].header['DRPNFILE'] = (nfiles, 'Num raw files used in pyKLIP')
        for i, filename in enumerate(filenames):
            hdulist[0].header['FILE_{0}'.format(i)] = filename + '.fits'

        # Write out PSF subtraction parameters and get pyKLIP revision number
        pykliproot = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        # The universal_newline argument is just so Python 3 returns a string
        # instead of bytes. This will probably come to bite me later.
        try:
            pyklipver = pyklip.__version__
        except:
            pyklipver = 'unknown'
        hdulist[0].header['PSFSUB'] = ('pyKLIP', 'PSF Subtraction Algo')
        hdulist[0].header.add_history('Reduced with pyKLIP using commit {0}'.format(pyklipver))
        hdulist[0].header['CREATOR'] = 'pyKLIP-{0}'.format(pyklipver)

        # Store commit number for pyklip
        hdulist[0].header['pyklipv'] = (pyklipver, 'pyKLIP version that was used')

        if klipparams is not None:
            hdulist[0].header['PSFPARAM'] = (klipparams, 'KLIP parameters')
            hdulist[0].header.add_history('pyKLIP reduction with parameters {0}'.format(klipparams))

        # Write z axis units if necessary
        if zaxis is not None:
            # Writing a KL mode cube
            if 'KL Mode' in filetype:
                hdulist[0].header['CTYPE3'] = 'KLMODES'
                # Write them individually
                for i, klmode in enumerate(zaxis):
                    hdulist[0].header['KLMODE{0}'.format(i)] = (klmode, 'KL Mode of slice {0}'.format(i))
                hdulist[0].header['CUNIT3'] = 'N/A'
                hdulist[0].header['CRVAL3'] = 1
                hdulist[0].header['CRPIX3'] = 1.
                hdulist[0].header['CD3_3'] = 1.

        # Store WCS information
        wcshdr = self.output_wcs[0].to_header()
        for key in wcshdr.keys():
            hdulist[0].header[key] = wcshdr[key]

        # Store extra keywords in header
        if more_keywords is not None:
            for hdr_key in more_keywords:
                hdulist[0].header[hdr_key] = more_keywords[hdr_key]

        # But update the image center
        center = self.output_centers[0]
        hdulist[0].header.update({'PSFCENTX': center[0], 'PSFCENTY': center[1]})
        hdulist[0].header.update({'CRPIX1': center[0], 'CRPIX2': center[1]})
        hdulist[0].header.add_history('Image recentered to {0}'.format(str(center)))

        try:
            hdulist.writeto(filepath, overwrite=True)
        except TypeError:
            hdulist.writeto(filepath, clobber=True)
        hdulist.close()


def trim_miri_data(data, filt):
    '''
    Trim the MIRI data to remove regions that receive no illumination. 
    
    Parameters
    ----------
    data : list of datacubes
        List of 3D datacubes that will be trimmed
    filt : string
        Filter data was gathered in

    Returns
    -------
    data_trim : list of datacubes
        Trimmed 3D datacubes
    trim : list
        Number of pixels trimmed from the left (trim[0]) and bottom (trim[1]))
    '''

    # Pixel values to trim around based on filter/mask, these were determined using
    # the MIRI psfmask files found on CRDS. 
    if filt.lower() == 'f1065c':
        l,r,b,t = 5, 217, 14, 227
    elif filt.lower() == 'f1140c':
        l,r,b,t = 7, 216, 13, 227
    elif filt.lower() == 'f1550c':
        l,r,b,t = 8, 215, 13, 226
    elif filt.lower() == 'f2300c':
        l,r,b,t = 3, 299, 9, 277

    # Copy data and trim accordingly
    data_trim = copy.deepcopy(data)
    for i, arr in enumerate(data):
        data_trim[i] = arr[:,l:r+1,b:t+1] #Want to include the final row/column

    # Trim is how many left and bottom pixels were cut off
    trim = [l , b]

    return data_trim, trim

def organise_files(filepaths, copy_dir='./ORGANISED/', hierarchy='TARGPROP/FILTER'):
    """
    Function to take a list of JWST files, and then copy and organise them
    into folders based on header keys.

    Parameters
    ----------
    file_list : list of str
        List of strings for each file.
    copy_dir : str
        Directory to copy files to.
    hierarchy : str
        Structure of the new directory organisation, using available header
        keywords.
    """
    # Check if directory we are copying to exists
    if not os.path.isdir(copy_dir):
        os.makedirs(copy_dir)

    # Get the keys we want to sort by
    divisions = hierarchy.split('/')

    # Loop over all of the files
    for file in filepaths:
        with fits.open(file) as f:
            # Loop over each of the keys we are interested in to create a
            # directory string
            working_dir = copy_dir
            for i in range(len(divisions)):
                key_val = f[0].header[divisions[i]]
                working_dir += key_val+'/'

            # Check if this new directory string exists
            if not os.path.isdir(working_dir):
                os.makedirs(working_dir)

            # Save file to the new directory
            file_suffix = file.split('/')[-1]
            shutil.copyfile(file, working_dir+file_suffix)

    return None
