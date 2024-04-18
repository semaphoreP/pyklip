from pyklip.instruments.Instrument import Data
import pyklip.rdi as rdi
import pyklip.klip

from astropy.io import fits
from astropy import wcs
from astroquery.svo_fps import SvoFps

import numpy as np
import os, shutil, re
import copy
import warnings

class JWSTData(Data):
    """
    The pyKLIP instrument class for JWST data.
    
    """
    
    ####################
    ### Constructors ###
    ####################
    
    def __init__(self,
                 filepaths,
                 psflib_filepaths=None,
                 highpass=False,
                 center_include_offset=True):
        """
        Initialize the pyKLIP instrument class for space telescope data.
        
        Parameters
        ----------
        filepaths : 1D-array
            Paths of the input science observations.
        psflib_filepaths : 1D-array, optional
            Paths of the input reference observations. The default is None.
        center_include_offset : bool
            Toggle as to whether the relative header offset values of each
            image is applied during image centering. 
        
        Returns
        -------
        None.
        
        """
        
        # Initialize pyKLIP Data class.
        super(JWSTData, self).__init__()

        # Load NIRCam, NIRISS, and MIRI filters from the SVO Filter Profile Service.
        # http://svo2.cab.inta-csic.es/theory/fps/
        self.wave_nircam = {}
        filter_list = SvoFps.get_filter_list(facility='JWST', instrument='NIRCAM')
        for i in range(len(filter_list)):
            name = filter_list['filterID'][i]
            name = name[name.rfind('.') + 1:]
            self.wave_nircam[name] = filter_list['WavelengthMean'][i] / 1e4  # micron
        self.wave_niriss = {}
        filter_list = SvoFps.get_filter_list(facility='JWST', instrument='NIRISS')
        for i in range(len(filter_list)):
            name = filter_list['filterID'][i]
            name = name[name.rfind('.') + 1:]
            self.wave_niriss[name] = filter_list['WavelengthMean'][i] / 1e4  # micron
        self.wave_miri = {}
        filter_list = SvoFps.get_filter_list(facility='JWST', instrument='MIRI')
        for i in range(len(filter_list)):
            name = filter_list['filterID'][i]
            name = name[name.rfind('.') + 1:]
            self.wave_miri[name] = filter_list['WavelengthMean'][i] / 1e4  # micron
        self.wave_miri['FND'] = 13.  # micron
        del filter_list

        # Optional variables
        self.center_include_offset = center_include_offset
                
        # Read science and reference files.
        self.readdata(filepaths)
        if psflib_filepaths is not None and len(psflib_filepaths) != 0:
            self.readpsflib(psflib_filepaths, highpass)
        else:
            self._psflib = None
        
        pass
    
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
    def OWA(self):
        return self._OWA
    @OWA.setter
    def OWA(self, newval):
        self._OWA = newval
    
    @property
    def psflib(self):
        return self._psflib
    @psflib.setter
    def psflib(self, newval):
        self._psflib = newval
    
    @property
    def output(self):
        return self._output
    @output.setter
    def output(self, newval):
        self._output = newval
    
    ###############
    ### Methods ###
    ###############
    
    def readdata(self,
                 filepaths):
        """
        Read the input science observations.
        
        Parameters
        ----------
        filepaths : 1D-array
            Paths of the input science observations.
        
        Returns
        -------
        None.
        
        """
        
        # Check input.
        if isinstance(filepaths, str):
            filepaths = np.array([filepaths])
        if len(filepaths) == 0:
            raise UserWarning('No science files provided to pyKLIP')
        
        # Loop through science files.
        input_all = []
        centers_all = []  # pix
        filenames_all = []
        PAs_all = []  # deg
        wvs_all = []  # m
        wcs_all = []
        PIXSCALE = []  # arcsec
        for i, filepath in enumerate(filepaths):
            
            # Read science file.
            hdul = fits.open(filepath)
            phead = hdul[0].header
            shead = hdul['SCI'].header
            TELESCOP = phead['TELESCOP']
            INSTRUME = phead['INSTRUME']
            data = hdul['SCI'].data
            try:
                pxdq = hdul['DQ'].data
            except:
                pxdq = np.zeros_like(data).astype('int')
            if data.ndim == 2:
                data = data[np.newaxis, :]
                pxdq = pxdq[np.newaxis, :]
            if data.ndim != 3:
                raise UserWarning('Requires 2D/3D data cube')
            NINTS = data.shape[0]
            pix_scale = np.sqrt(shead['PIXAR_A2']) 
            PIXSCALE += [pix_scale] #Assume square pixels

            # Nan out non-science pixels.
            data[pxdq & 512 == 512] = np.nan
            
            # Get centers.
            if self.center_include_offset == True:
                # Use the offset values from the header to adjust the center
                centers = np.array([shead['CRPIX1'] - 1 + phead['XOFFSET'] / pix_scale, 
                    shead['CRPIX2'] - 1 + phead['YOFFSET'] / pix_scale] * NINTS)
            else:
                # Assume the CRPIX define the correct center for each image
                centers = np.array([shead['CRPIX1'] - 1, shead['CRPIX2'] - 1] * NINTS)

            # Get metadata.
            input_all += [data]
            centers_all += [centers]
            filenames_all += [os.path.split(filepath)[1] + '_INT%.0f' % (j + 1) for j in range(NINTS)]
            PAs_all += [shead['ROLL_REF'] - (shead['V3I_YANG']*shead['VPARITY'])] * NINTS
            if INSTRUME == 'NIRCAM':
                if phead['PUPIL'] in self.wave_nircam.keys():
                    CWAVEL = self.wave_nircam[phead['PUPIL']]
                else:
                    CWAVEL = self.wave_nircam[phead['FILTER']]
            elif INSTRUME == 'NIRISS':
                CWAVEL = self.wave_niriss[phead['FILTER']]
            elif INSTRUME == 'MIRI':
                CWAVEL = self.wave_miri[phead['FILTER']]
            else:
                raise UserWarning('Data originates from unknown JWST instrument')
            wvs_all += [1e-6 * CWAVEL] * NINTS
            wcs_hdr = wcs.WCS(header=hdul['SCI'].header, naxis=hdul['SCI'].header['WCSAXES'])
            for j in range(NINTS):
                wcs_all += [wcs_hdr.deepcopy()]
            hdul.close()

        input_all = np.concatenate(input_all)
        if input_all.ndim != 3:
            raise UserWarning('Some science files do not have matching image shapes')
        centers_all = np.concatenate(centers_all).reshape(-1, 2)
        filenames_all = np.array(filenames_all)
        filenums_all = np.array(range(len(filenames_all)))
        PAs_all = np.array(PAs_all)
        wvs_all = np.array(wvs_all)
        wcs_all = np.array(wcs_all)
        PIXSCALE = np.unique(np.array(PIXSCALE))
        if len(PIXSCALE) != 1:
            raise UserWarning('Some science files do not have matching pixel scales')
        if TELESCOP == 'JWST' and phead['EXP_TYPE'] in ['NRC_CORON', 'MIR_LYOT']:
            iwa_all = np.min(wvs_all) / 6.5 * 180. / np.pi * 3600. / PIXSCALE[0]  # pix
        elif TELESCOP == 'JWST' and phead['EXP_TYPE'] in ['MIR_4QPM']:
            iwa_all = 0.5 * np.min(wvs_all) / 6.5 * 180. / np.pi * 3600. / PIXSCALE[0]  # pix
        else:
            iwa_all = 1.  # pix
        owa_all = np.sum(np.array(input_all.shape[1:]) / 2.)  # pix

        # Recenter science images so that the star is at the center of the array.
        new_center = (np.array(data.shape[1:])-1)/ 2.
        new_center = new_center[::-1]
        for i, image in enumerate(input_all):
            recentered_image = pyklip.klip.align_and_scale(image, new_center=new_center, old_center=centers_all[i])
            input_all[i] = recentered_image
            centers_all[i] = new_center
        
        # Assign pyKLIP variables.
        self._input = input_all
        self._centers = centers_all
        self._filenames = filenames_all
        self._filenums = filenums_all
        self._PAs = PAs_all
        self._wvs = wvs_all
        self._wcs = wcs_all
        self._IWA = iwa_all
        self._OWA = owa_all
        
        pass
    
    def readpsflib(self,
                   psflib_filepaths,
                   highpass=False):
        """
        Read the input reference observations.
        
        Parameters
        ----------
        psflib_filepaths : 1D-array, optional
            Paths of the input reference observations. The default is None.
        
        Returns
        -------
        None.
        
        """
        
        # Check input.
        if isinstance(psflib_filepaths, str):
            psflib_filepaths = np.array([psflib_filepaths])
        if len(psflib_filepaths) == 0:
            raise UserWarning('No reference files provided to pyKLIP')
        
        # Loop through reference files.
        psflib_data_all = []
        psflib_centers_all = []  # pix
        psflib_filenames_all = []
        for i, filepath in enumerate(psflib_filepaths):
            
            # Read reference file.
            hdul = fits.open(filepath)
            data = hdul['SCI'].data
            try:
                pxdq = hdul['DQ'].data
            except:
                pxdq = np.zeros_like(data).astype('int')
            phead = hdul[0].header
            shead = hdul['SCI'].header
            if data.ndim == 2:
                data = data[np.newaxis, :]
                pxdq = pxdq[np.newaxis, :]
            if data.ndim != 3:
                raise UserWarning('Requires 2D/3D data cube')
            NINTS = data.shape[0]
            pix_scale = np.sqrt(shead['PIXAR_A2']) 
            
            # Nan out non-science pixels.
            data[pxdq & 512 == 512] = np.nan

            # Get centers.
            if self.center_include_offset == True:
                # Use the offset values from the header to adjust the center
                centers = np.array([shead['CRPIX1'] - 1 + phead['XOFFSET'] / pix_scale, 
                    shead['CRPIX2'] - 1 + phead['YOFFSET'] / pix_scale] * NINTS)
            else:
                # Assume the CRPIX define the correct center
                centers = np.array([shead['CRPIX1'] - 1, shead['CRPIX2'] - 1] * NINTS)
            
            # Get metadata.
            psflib_data_all += [data]
            psflib_centers_all += [centers]
            psflib_filenames_all += [os.path.split(filepath)[1] + '_INT%.0f' % (j + 1) for j in range(NINTS)]
            hdul.close()
        psflib_data_all = np.concatenate(psflib_data_all)
        if psflib_data_all.ndim != 3:
            raise UserWarning('Some reference files do not have matching image shapes')
        psflib_centers_all = np.concatenate(psflib_centers_all).reshape(-1, 2)
        psflib_filenames_all = np.array(psflib_filenames_all)
        
        # Recenter reference images.
        new_center = (np.array(data.shape[1:])-1)/ 2.
        new_center = new_center[::-1]
        for i, image in enumerate(psflib_data_all):
            recentered_image = pyklip.klip.align_and_scale(image, new_center=new_center, old_center=psflib_centers_all[i])
            psflib_data_all[i] = recentered_image
            psflib_centers_all[i] = new_center
        
        # Append science data.
        psflib_data_all = np.append(psflib_data_all, self._input, axis=0)
        psflib_centers_all = np.append(psflib_centers_all, self._centers, axis=0)
        psflib_filenames_all = np.append(psflib_filenames_all, self._filenames, axis=0)
        
        # Initialize PSF library.
        psflib = rdi.PSFLibrary(psflib_data_all, new_center, psflib_filenames_all, compute_correlation=True, highpass=highpass)
        
        # Prepare PSF library.
        psflib.prepare_library(self)
        
        # Assign pyKLIP variables.
        self._psflib = psflib
        
        pass
    
    def savedata(self,
                 filepath,
                 data,
                 klipparams=None,
                 filetype='',
                 zaxis=None,
                 more_keywords=None):
        """
        Function to save the data products that will be called internally by
        pyKLIP.
        
        Parameters
        ----------
        filepath : path
            Path of the output FITS file.
        data : 3D-array
            KLIP-subtracted data of shape (nkl, ny, nx).
        klipparams : str, optional
            PyKLIP keyword arguments used for the KLIP subtraction. The default
            is None.
        filetype : str, optional
            Data type of the pyKLIP product. The default is ''.
        zaxis : list, optional
            List of KL modes used for the KLIP subtraction. The default is
            None.
        more_keywords : dict, optional
            Dictionary of additional header keywords to be written to the
            output FITS file. The default is None.
        
        Returns
        -------
        None.
        
        """
        
        # Make FITS file.
        hdul = fits.HDUList()
        hdul.append(fits.PrimaryHDU(data))
        
        # Write all used files to header. Ignore duplicates.
        filenames = np.unique(self.filenames)
        Nfiles = np.size(filenames)
        hdul[0].header['DRPNFILE'] = (Nfiles, 'Num raw files used in pyKLIP')
        for i, filename in enumerate(filenames):
            if i < 1000:
                hdul[0].header['FILE_{0}'.format(i)] = filename + '.fits'
            else:
                print('WARNING: Too many files to be written to header, skipping')
                break
        
        # Write PSF subtraction parameters and pyKLIP version to header.
        try:
            pyklipver = pyklip.__version__
        except:
            pyklipver = 'unknown'
        hdul[0].header['PSFSUB'] = ('pyKLIP', 'PSF Subtraction Algo')
        hdul[0].header.add_history('Reduced with pyKLIP using commit {0}'.format(pyklipver))
        hdul[0].header['CREATOR'] = 'pyKLIP-{0}'.format(pyklipver)
        hdul[0].header['pyklipv'] = (pyklipver, 'pyKLIP version that was used')
        if klipparams is not None:
            hdul[0].header['PSFPARAM'] = (klipparams, 'KLIP parameters')
            hdul[0].header.add_history('pyKLIP reduction with parameters {0}'.format(klipparams))
        
        # Write z-axis units to header if necessary.
        if zaxis is not None:
            if 'KL Mode' in filetype:
                hdul[0].header['CTYPE3'] = 'KLMODES'
                for i, klmode in enumerate(zaxis):
                    hdul[0].header['KLMODE{0}'.format(i)] = (klmode, 'KL Mode of slice {0}'.format(i))

        # Write extra keywords to header if necessary.
        if more_keywords is not None:
            for hdr_key in more_keywords:
                hdul[0].header[hdr_key] = more_keywords[hdr_key]
        
        # Update image center.
        center = self.output_centers[0]
        hdul[0].header.update({'PSFCENTX': center[0], 'PSFCENTY': center[1]})
        hdul[0].header.update({'CRPIX1': center[0] + 1, 'CRPIX2': center[1] + 1})
        hdul[0].header.add_history('Image recentered to {0}'.format(str(center)))
        
        # Write FITS file.
        try:
            hdul.writeto(filepath, overwrite=True)
        except TypeError:
            hdul.writeto(filepath, clobber=True)
        hdul.close()
        
        pass