__author__ = 'jruffio'
import multiprocessing as mp
import ctypes

import numpy as np
import pyklip.spectra_management as spec
import os

from pyklip.fmlib.nofm import NoFM
import pyklip.fm as fm
import pyklip.fakes as fakes

from scipy import interpolate, linalg
from copy import copy

#import matplotlib.pyplot as plt
debug = False


class ExtractSpec(NoFM):
    """
    Planet Characterization class. Goal to characterize the astrometry and photometry of a planet
    """
    def __init__(self, inputs_shape,
                 numbasis,
                 sep, pa,
                 input_psfs,
                 input_psfs_wvs,
                 input_psfs_pas=None,
                 datatype="float",
                 stamp_size = None):
        """
        Defining the planet to characterizae

        Args:
            inputs_shape: shape of the inputs numpy array. Typically (N, y, x)
            numbasis: 1d numpy array consisting of the number of basis vectors to use
            sep: separation of the planet
            pa: position angle of the planet
            input_psfs: the psf of the image. A numpy array with shape (wv, y, x) 
                        shape of (N_cubes, wvs, y, x) also acceptable to have PSFs change in time.
            input_psfs_wvs: the wavelegnths that correspond to the input psfs
            input_psfs_pas: the parangs when each input psf was taken (should be unique)
            flux_conversion: an array of length N to convert from contrast to DN for each frame. Units of DN/contrast
            wavelengths: wavelengths of data. Can just be a string like 'H' for H-band
            spectrallib: if not None, a list of spectra
            star_spt: star spectral type, if None default to some random one
            refine_fit: refine the separation and pa supplied
        """
        # allocate super class
        super(ExtractSpec, self).__init__(inputs_shape, np.array(numbasis))

        if stamp_size is None:
            self.stamp_size = 10
        else:
            self.stamp_size = stamp_size

        if datatype=="double":
            self.data_type = ctypes.c_double
        elif datatype=="float":
            self.data_type = ctypes.c_float

        self.N_numbasis =  np.size(numbasis)
        self.ny = self.inputs_shape[1]
        self.nx = self.inputs_shape[2]
        self.N_frames = self.inputs_shape[0]

        self.inputs_shape = inputs_shape
        self.numbasis = numbasis
        self.sep = sep
        self.pa = pa


        # 2018-04-10 AG: Generalizing so everything is in units of input PSF
        # Specifically removing the normlization of the input PSF
        self.input_psfs = input_psfs
        self.input_psfs_wvs = list(np.array(input_psfs_wvs,dtype=self.data_type))
        self.nl = np.size(input_psfs_wvs)

        self.psf_centx_notscaled = {}
        self.psf_centy_notscaled = {}

        if len(self.input_psfs.shape) == 3:
            # default what we exepct
            self.nl, self.ny_psf, self.nx_psf =  self.input_psfs.shape

            x_psf_grid, y_psf_grid = np.meshgrid(np.arange(self.nx_psf * 1.)-self.nx_psf//2,np.arange(self.ny_psf* 1.)-self.ny_psf//2)
            psfs_func_list = []
            for wv_index in range(self.nl):
                model_psf = self.input_psfs[wv_index, :, :] #* self.flux_conversion * self.spectrallib[0][wv_index] * self.dflux
                psfs_func_list.append(interpolate.LSQBivariateSpline(x_psf_grid.ravel(),y_psf_grid.ravel(),model_psf.ravel(),x_psf_grid[0,0:self.nx_psf-1]+0.5,y_psf_grid[0:self.ny_psf-1,0]+0.5))

            self.psfs_in_time = False
        else:
            # account for time variability of PSF
            self.ncubes, self.nl, self.ny_psf, self.nx_psf =  self.input_psfs.shape

            self.input_psfs_pas = input_psfs_pas

            x_psf_grid, y_psf_grid = np.meshgrid(np.arange(self.nx_psf * 1.)-self.nx_psf//2,np.arange(self.ny_psf* 1.)-self.ny_psf//2)
            psfs_func_list = []
            for pa_index in range(self.ncubes):
                psfs_func_list_perpa = []
                for wv_index in range(self.nl):
                    model_psf = self.input_psfs[pa_index, wv_index, :, :] #* self.flux_conversion * self.spectrallib[0][wv_index] * self.dflux
                    psfs_func_list_perpa.append(interpolate.LSQBivariateSpline(x_psf_grid.ravel(),y_psf_grid.ravel(),model_psf.ravel(),x_psf_grid[0,0:self.nx_psf-1]+0.5,y_psf_grid[0:self.ny_psf-1,0]+0.5))
                psfs_func_list.append(psfs_func_list_perpa)

            self.psfs_in_time = True
            
        self.psfs_func_list = psfs_func_list


    def alloc_fmout(self, output_img_shape):
        """
        Allocates shared memory for the output of the shared memory

        Args:
            output_img_shape: shape of output image (usually N,y,x,b)

        Returns:
            fmout: mp.array to store FM data in
            fmout_shape: shape of FM data array

        """

        # The 3rd dimension (self.N_frames corresponds to the spectrum)
        # The +1 in (self.N_frames+1) is for the klipped image
        fmout_size = self.N_numbasis*self.N_frames*(self.N_frames+1)*self.stamp_size*self.stamp_size
        # 2018-04-10 AG: force fmout to be type int
        fmout = mp.Array(self.data_type, int(fmout_size))
        # fmout shape is defined as:
        #   (self.N_numbasis,self.N_frames,(self.N_frames+1),self.stamp_size*self.stamp_size)
        # 1st dim: The size of the numbasis input. numasis gives the list of the number of KL modes we want to try out
        #           e.g. numbasis = [10,20,50].
        # 2nd dim: It is the Forward model dimension. It contains the forard model for each frame in the dataset.
        #           N_frames = N_cubes*(Number of spectral channel=37)
        # 3nd dim: It contains both the "spectral dimension" and the klipped image.
        #           The regular klipped data is fmout[:,:, -1,:]
        #           The regular forward model is fmout[:,:, 0:self.N_frames,:]
        #           Multiply a vector of fluxes to this dimension of fmout[:,:, 0:self.N_frames,:] and you should get
        #           forward model for that given spectrum.
        # 4th dim: pixels value. It has the size of the number of pixels in the stamp self.stamp_size*self.stamp_size.
        fmout_shape = (self.N_numbasis,self.N_frames,(self.N_frames+1),self.stamp_size*self.stamp_size )

        return fmout, fmout_shape


    # def alloc_perturbmag(self, output_img_shape, numbasis):
    #     """
    #     Allocates shared memory to store the fractional magnitude of the linear KLIP perturbation
    #     Stores a number for each frame = max(oversub + selfsub)/std(PCA(image))
    #
    #     Args:
    #         output_img_shape: shape of output image (usually N,y,x,b)
    #         numbasis: array/list of number of KL basis cutoffs requested
    #
    #     Returns:
    #         perturbmag: mp.array to store linaer perturbation magnitude
    #         perturbmag_shape: shape of linear perturbation magnitude
    #
    #     """
    #     perturbmag_shape = (output_img_shape[0], np.size(numbasis))
    #     perturbmag = mp.Array(ctypes.c_double, np.prod(perturbmag_shape))
    #
    #     return perturbmag, perturbmag_shape


    def generate_models(self, input_img_shape, section_ind, pas, wvs, radstart, radend, phistart, phiend, padding, ref_center, parang, ref_wv, flipx, stamp_size = None):
        """
        Generate model PSFs at the correct location of this segment for each image denoated by its wv and parallactic angle

        Args:
            pas: array of N parallactic angles corresponding to N images [degrees]
            wvs: array of N wavelengths of those images
            radstart: radius of start of segment
            radend: radius of end of segment
            phistart: azimuthal start of segment [radians]
            phiend: azimuthal end of segment [radians]
            padding: amount of padding on each side of sector
            ref_center: center of image
            parang: parallactic angle of input image [DEGREES]
            ref_wv: wavelength of science image
            stamp_size: size of the stamp for spectral extraction
            flipx: if True, flip x coordinate in final image

        Return:
            models: array of size (N, p) where p is the number of pixels in the segment
        """
        # create some parameters for a blank canvas to draw psfs on
        nx = input_img_shape[1]
        ny = input_img_shape[0]
        x_grid, y_grid = np.meshgrid(np.arange(nx * 1.)-ref_center[0], np.arange(ny * 1.)-ref_center[1])


        numwv, ny_psf, nx_psf =  self.nl, self.ny_psf, self.nx_psf

        # create bounds for PSF stamp size
        row_m = np.floor(ny_psf/2.0)    # row_minus
        row_p = np.ceil(ny_psf/2.0)     # row_plus
        col_m = np.floor(nx_psf/2.0)    # col_minus
        col_p = np.ceil(nx_psf/2.0)     # col_plus

        if stamp_size is not None:
            stamp_mask = np.zeros((ny,nx))
            # create bounds for spectral extraction stamp size
            row_m_stamp = np.floor(stamp_size/2.0)    # row_minus
            row_p_stamp = np.ceil(stamp_size/2.0)     # row_plus
            col_m_stamp = np.floor(stamp_size/2.0)    # col_minus
            col_p_stamp = np.ceil(stamp_size/2.0)     # col_plus
            stamp_indices=[]

        # a blank img array of write model PSFs into
        whiteboard = np.zeros((ny,nx))
        if debug:
            canvases = []
        models = []
        #print(self.input_psfs.shape)
        for pa, wv in zip(pas, wvs):
            #print(self.pa,self.sep)
            #print(pa,wv)
            # grab PSF given wavelength
            wv_index = spec.find_nearest(self.input_psfs_wvs,wv)[1]
            #model_psf = self.input_psfs[wv_index[0], :, :] #* self.flux_conversion * self.spectrallib[0][wv_index] * self.dflux

            # find center of psf
            # to reduce calculation of sin and cos, see if it has already been calculated before
            if pa not in self.psf_centx_notscaled:
                # flipx requires the opposite rotation
                sign = -1.
                if flipx:
                    sign = 1.
                self.psf_centx_notscaled[pa] = self.sep * np.cos(np.radians(90. - sign*self.pa - pa))
                self.psf_centy_notscaled[pa] = self.sep * np.sin(np.radians(90. - sign*self.pa - pa))
            psf_centx = (ref_wv/wv) * self.psf_centx_notscaled[pa]
            psf_centy = (ref_wv/wv) * self.psf_centy_notscaled[pa]

            # create a coordinate system for the image that is with respect to the model PSF
            # round to nearest pixel and add offset for center
            l = round(psf_centx + ref_center[0])
            k = round(psf_centy + ref_center[1])
            # recenter coordinate system about the location of the planet
            x_vec_stamp_centered = x_grid[0, int(l-col_m):int(l+col_p)]-psf_centx
            y_vec_stamp_centered = y_grid[int(k-row_m):int(k+row_p), 0]-psf_centy
            # rescale to account for the align and scaling of the refernce PSFs
            # e.g. for longer wvs, the PSF has shrunk, so we need to shrink the coordinate system
            x_vec_stamp_centered /= (ref_wv/wv)
            y_vec_stamp_centered /= (ref_wv/wv)

            # use intepolation spline to generate a model PSF and write to temp img
            if not self.psfs_in_time:
                # just grab the right wavelength
                psf_func = self.psfs_func_list[int(wv_index)]
            else:
                pa_index = spec.find_nearest(self.input_psfs_pas, pa)[1]
                psf_func = self.psfs_func_list[int(pa_index)][int(wv_index)]
            whiteboard[int(k-row_m):int(k+row_p), int(l-col_m):int(l+col_p)] = \
                    psf_func(x_vec_stamp_centered,y_vec_stamp_centered).transpose()

            # write model img to output (segment is collapsed in x/y so need to reshape)
            whiteboard = np.reshape(whiteboard, [input_img_shape[0] * input_img_shape[1]], copy=False)
            segment_with_model = copy(whiteboard[section_ind])
            whiteboard = np.reshape(whiteboard, [input_img_shape[0],input_img_shape[1]], copy=False)

            models.append(segment_with_model)
            if stamp_size is not None:
                # These are actually indices of indices. they indicate which indices correspond to the stamp in section_ind
                stamp_mask[int(k-row_m_stamp):int(k+row_p_stamp), int(l-col_m_stamp):int(l+col_p_stamp)] = 1
                stamp_mask = np.reshape(stamp_mask, [nx*ny], copy=False)
                stamp_indices.append(np.where(stamp_mask[section_ind] == 1)[0])
                stamp_mask = np.reshape(stamp_mask, [ny,nx], copy=False)
                stamp_mask[int(k-row_m_stamp):int(k+row_p_stamp), int(l-col_m_stamp):int(l+col_p_stamp)] = 0

        if stamp_size is not None:
            return np.array(models),stamp_indices
        else:
            return np.array(models)




    def fm_from_eigen(self, klmodes=None, evals=None, evecs=None, input_img_shape=None, input_img_num=None, ref_psfs_indicies=None, section_ind=None,section_ind_nopadding=None, aligned_imgs=None, pas=None,
                     wvs=None, radstart=None, radend=None, phistart=None, phiend=None, padding=None,IOWA = None, ref_center=None,
                     parang=None, ref_wv=None, numbasis=None, fmout=None, perturbmag=None, klipped=None, flipx=True, **kwargs):
        """
        Generate forward models using the KL modes, eigenvectors, and eigenvectors from KLIP. Calls fm.py functions to
        perform the forward modelling

        Args:
            klmodes: unpertrubed KL modes
            evals: eigenvalues of the covariance matrix that generated the KL modes in ascending order
                   (lambda_0 is the 0 index) (shape of [nummaxKL])
            evecs: corresponding eigenvectors (shape of [p, nummaxKL])
            input_image_shape: 2-D shape of inpt images ([ysize, xsize])
            input_img_num: index of sciece frame
            ref_psfs_indicies: array of indicies for each reference PSF
            section_ind: array indicies into the 2-D x-y image that correspond to this section.
                         Note needs be called as section_ind[0]
            pas: array of N parallactic angles corresponding to N reference images [degrees]
            wvs: array of N wavelengths of those referebce images
            radstart: radius of start of segment
            radend: radius of end of segment
            phistart: azimuthal start of segment [radians]
            phiend: azimuthal end of segment [radians]
            padding: amount of padding on each side of sector
            IOWA: tuple (IWA,OWA) where IWA = Inner working angle and OWA = Outer working angle both in pixels.
                It defines the separation interva in which klip will be run.
            ref_center: center of image
            numbasis: array of KL basis cutoffs
            parang: parallactic angle of input image [DEGREES]
            ref_wv: wavelength of science image
            fmout: numpy output array for FM output. Shape is (N, y, x, b)
            perturbmag: numpy output for size of linear perturbation. Shape is (N, b)
            klipped: PSF subtracted image. Shape of ( size(section), b)
            kwargs: any other variables that we don't use but are part of the input
        """
        sci = aligned_imgs[input_img_num, section_ind[0]]
        refs = aligned_imgs[ref_psfs_indicies, :]
        refs = refs[:, section_ind[0]]


        # generate models for the PSF of the science image
        model_sci, stamp_indices = self.generate_models(input_img_shape, section_ind, [parang], [ref_wv], radstart, radend, phistart, phiend, padding, ref_center, parang, ref_wv, flipx, stamp_size=self.stamp_size)
        model_sci = model_sci[0]
        stamp_indices = stamp_indices[0]

        # generate models of the PSF for each reference segments. Output is of shape (N, pix_in_segment)
        models_ref = self.generate_models(input_img_shape, section_ind, pas, wvs, radstart, radend, phistart, phiend, padding, ref_center, parang, ref_wv, flipx)

        # using original Kl modes and reference models, compute the perturbed KL modes (spectra is already in models)
        #delta_KL = fm.perturb_specIncluded(evals, evecs, klmodes, refs, models_ref)
        delta_KL_nospec = fm.pertrurb_nospec(evals, evecs, klmodes, refs, models_ref)

        # calculate postklip_psf using delta_KL
        oversubtraction, selfsubtraction = fm.calculate_fm(delta_KL_nospec, klmodes, numbasis, sci, model_sci, inputflux=None)
        # klipped_oversub.shape = (size(numbasis),Npix)
        # klipped_selfsub.shape = (size(numbasis),N_lambda or N_ref,N_pix)
        # klipped_oversub = Sum(<S|KL>KL)
        # klipped_selfsub = Sum(<N|DKL>KL) + Sum(<N|KL>DKL)


        # Note: The following could be used if we want to derotate the image but JB doesn't think we have to.
        # # write forward modelled PSF to fmout (as output)
        # # need to derotate the image in this step
        # for thisnumbasisindex in range(np.size(numbasis)):
        #         fm._save_rotated_section(input_img_shape, postklip_psf[thisnumbasisindex], section_ind,
        #                          fmout[input_img_num, :, :,thisnumbasisindex], None, parang,
        #                          radstart, radend, phistart, phiend, padding,IOWA, ref_center, flipx=True)


        # fmout shape is defined as:
        #   (self.N_numbasis,self.N_frames,(self.N_frames+1),self.stamp_size*self.stamp_size)
        # 1st dim: The size of the numbasis input. numasis gives the list of the number of KL modes we want to try out
        #           e.g. numbasis = [10,20,50].
        # 2nd dim: It is the Forward model dimension. It contains the forard model for each frame in the dataset.
        #           N_frames = N_cubes*(Number of spectral channel=37)
        # 3nd dim: It contains both the "spectral dimension" and the klipped image.
        #           The regular klipped data is fmout[:,:, -1,:]
        #           The regular forward model is fmout[:,:, 0:self.N_frames,:]
        #           Multiply a vector of fluxes to this dimension of fmout[:,:, 0:self.N_frames,:] and you should get
        #           forward model for that given spectrum.
        # 4th dim: pixels value. It has the size of the number of pixels in the stamp self.stamp_size*self.stamp_size.
        for k in range(self.N_numbasis):
            fmout[k,input_img_num, input_img_num,:] = fmout[k,input_img_num, input_img_num,:]+model_sci[stamp_indices]
        fmout[:,input_img_num, input_img_num,:] = fmout[:,input_img_num, input_img_num,:]-oversubtraction[:,stamp_indices]
        fmout[:,input_img_num, ref_psfs_indicies,:] = fmout[:,input_img_num, ref_psfs_indicies,:]-selfsubtraction[:,:,stamp_indices]
        fmout[:,input_img_num, -1,:] = klipped.T[:,stamp_indices]




    def cleanup_fmout(self, fmout):
        """
        After running KLIP-FM, we need to reshape fmout so that the numKL dimension is the first one and not the last

        Args:
            fmout: numpy array of ouput of FM

        Return:
            fmout: same but cleaned up if necessary
        """
        # Here we actually extract the spectrum


        return fmout

def invert_spect_fmodel(fmout, dataset, method = "JB", units = "natural",
                        scaling_factor=1.0):
    """
    A. Greenbaum Nov 2016
    
    Args:
        fmout: the forward model matrix which has structure:
               [numbasis, n_frames, n_frames+1, npix]
        dataset: from GPI.GPIData(filelist) -- typically set highpass=True also
        method: "JB" or "LP" to try the 2 different inversion methods (JB's or Laurent's)
        units: "natural" means the answer is scaled to the input PSF (default)
               fmout will be in these units.
               "scaled" means the output is scaled to "scaling_factor" argument
        scaling_factor: multiplies output spectrum and forward model, user set for 
                        desired calibration factor. units="scaled" must be set in 
                        args for this to work!
    Returns:
        A tuple containing the spectrum and the forward model
        (spectrum, forwardmodel)
        spectrum shape:(len(numbasis), nwav)
        
    """
    N_frames = fmout.shape[2] - 1 # The last element in this axis contains klipped image
    N_cubes = np.size(np.unique(dataset.filenums)) # 
    nl = N_frames // N_cubes
    stamp_N_pix = fmout.shape[-1]

    # Selection matrix (N_cubes, 1) shape
    spec_identity = np.identity(nl)
    selec = np.tile(spec_identity,(N_frames//nl, 1))

    # set up array for klipped image for each numbasis, n_frames x npix
    klipped = np.zeros((fmout.shape[0], fmout.shape[1], fmout.shape[3]))
    estim_spec = np.zeros((fmout.shape[0], nl))

    # The first dimension in fmout is numbasis, and there can be multiple of these,
    # Especially if you want to see how the spectrum behaves when you change parameters.
    # We'll also set aside an array to store the forward model matrix
    fm_coadd_mat = np.zeros((len(fmout), nl*stamp_N_pix, nl))
    for ii in range(len(fmout)):
        klipped[ii, ...] = fmout[ii,:, -1,:]
        # klipped_coadd will be coadded over N_cubes
        klipped_coadd = np.zeros((int(nl),int(stamp_N_pix)))
        for k in range(N_cubes):
            klipped_coadd = klipped_coadd + klipped[ii, k*nl:(k+1)*nl,:]
        print(klipped_coadd.shape)
        #klipped_coadd.shape = [int(nl),int(stamp_size),int(stamp_size)]
        # This is the 'raw' forward model, need to rearrange to solve FM*spec = klipped
        FM_noSpec = fmout[ii, :,:N_frames, :]

        # Move spectral dimension to the end (Effectively move pixel dimension to the middle)
        # [nframes, nframes, npix] -> [nframes, npix, nframes]
        FM_noSpec = np.rollaxis(FM_noSpec, 2, 1)

        # S^T . FM[npix, nframes, nframes] . S
        # essentially coadds over N_cubes via selection matrix
        # reduces to [nwav, npix, nwav]
        fm_noSpec_coadd = np.dot(selec.T,np.dot(np.rollaxis(FM_noSpec,1,0),selec))
        if method == "JB":
            #
            #JBR's matrix inversion adds up over all exposures, then inverts
            #
            #Back to a 2D array pixel array in the middle
            fm_noSpec_coadd = np.reshape(fm_noSpec_coadd, [int(nl),stamp_N_pix, int(nl)], copy=False)
            # Flatten over first 3 dims for the FM matrix to solve FM*spect = klipped
            fm_noSpec_coadd_mat = np.reshape(fm_noSpec_coadd,(int(nl*stamp_N_pix),int(nl)))
            # Invert the FM matrix
            pinv_fm_coadd_mat = np.linalg.pinv(fm_noSpec_coadd_mat)
            # solve via FM^-1 . klipped_PSF (flattened) << both are coadded over N_cubes
            estim_spec[ii,:]=np.dot(pinv_fm_coadd_mat, klipped_coadd.ravel())
            fm_coadd_mat[ii,:, :] = fm_noSpec_coadd_mat
        elif method == "LP":
            #
            #LP's matrix inversion adds over frames and one wavelength axis, then inverts
            #
            A = np.zeros((nl, nl))
            b = np.zeros(nl)
            fm = fm_noSpec_coadd.reshape(int(nl), stamp_N_pix, int(nl))
            fm_coadd_mat[ii,:, :] = \
                fm_noSpec_coadd.reshape(int(nl*stamp_N_pix), int(nl))
            fm = np.rollaxis(fm, 2,0)
            fm = np.rollaxis(fm, 2,1)
            data = klipped_coadd
            for q in range(nl):
                A[q,:] = np.dot(fm[q,:].T,fm[q,:])[q,:]
                b[q] = np.dot(fm[q,:].T,data[q])[q]
            estim_spec[ii,:] = np.dot(np.linalg.inv(A), b)
        elif method == "leastsq":
            # MF's suggestion of solving using a least sq function 
            # instead of matrix inversions
            
            #Back to a 2D array pixel array in the middle
            fm_noSpec_coadd = np.reshape(fm_noSpec_coadd, [int(nl),stamp_N_pix,int(nl)], copy=False)
            # Flatten over first 3 dims for the FM matrix to solve FM*spect = klipped
            fm_noSpec_coadd_mat = np.reshape(fm_noSpec_coadd,(int(nl*stamp_N_pix),int(nl)))
            # Saving the coadded FM
            fm_coadd_mat[ii,:, :] = fm_noSpec_coadd_mat

            # used leastsq solver
            results = linalg.lstsq(fm_noSpec_coadd_mat, klipped_coadd.ravel())
            # grab the spectrum, not using the other parts for now.
            estim_spec[ii,:], res, rank, s = results

        else:
            print("method not understood. Choose either JB, LP or leastsq.")

    if units=="scaled":
        return scaling_factor*estim_spec, fm_coadd_mat / scaling_factor
    else:
        return estim_spec, fm_coadd_mat

