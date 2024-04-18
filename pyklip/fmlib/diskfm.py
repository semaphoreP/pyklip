# pylint: disable=C0103
from sys import version_info
from os import path
import multiprocessing as mp
from copy import deepcopy
import ctypes

import pickle
import h5py

import numpy as np
import scipy.ndimage as ndimage

from pyklip.fmlib.nofm import NoFM
import pyklip.fm as fm
from pyklip.klip import rotate

# define the global variables for that code
class DiskFM(NoFM):
    """Defining a model disk to which we apply the Forward Modelling. There are 3 ways:

            * "Save Basis mode" (save_basis=true), we are preparing to save the FM basis
            * "Load Basis mode" (load_from_basis = true), most of the parameters are
              derived from the previous fm.klip_dataset which measured FM basis.
            * "Simple FM mode" (save_basis = load_from_basis = False). Just
              for a unique disk FM.


        Args:
            inputs_shape: shape of the inputs numpy array. Typically (N, x, y)
            numbasis: 1d numpy array consisting of the number of basis vectors to use
            dataset: an instance of Instrument.Data. We need it to know the
                     parameters to "prepare" first inital model.
            model_disk: a model of the disk of size (wvs, x, y) or (x, y)
            basis_filename: filename to save and load the KL basis. Filenames can haves
                            2 recognizable extensions: .h5 or .pkl. We strongly
                            recommand .h5 as pickle have problem of compatibility
                            between python 2 and 3 and sometimes between computer
                            (e.g. KL modes not readable on another computer)
            load_from_basis: if True, load the KL basis at basis_filename. It only need
                             to be done once, after which you can measure FM with
                             only update_model()
            save_basis: if True, save the KL basis at basis_filename. If load_from_basis
                            is True, save_basis is automatically set to False, it is
                            useless to load and save the matrix at the same time.
            aligned_center: array of 2 elements [x,y] that all the model will be
                            centered on for image registration.
                            FIXME: This is the most problematic thing currently, the
                            aligned_center of the model and of the images can be set
                            independently, which will create false results.
                            - In "Load Basis mode", this parameter is not read, we just
                            use the aligned_center set for the images in the previous
                            fm.klip_dataset and save in basis_filename
                            - In "Save Basis mode", or "Simple FM mode" we define it
                            and then check that it is the same one used for the images
                            in fm.klip_dataset
            mode: deprecated parameter, ignored here and defined in fm.klip_dataset
            annuli: deprecated parameter, ignored here and defined in fm.klip_dataset
            subsections: deprecated parameter, ignored here and defined
                         in fm.klip_dataset
            numthreads: deprecated parameter. All centering are done in fm.klip_dataset 

        Returns:
            A DiskFM Object

    """
    def __init__(self,
                 inputs_shape,
                 numbasis,
                 dataset,
                 model_disk,
                 basis_filename="",
                 kl_basis_file=None,
                 load_from_basis=False,
                 save_basis=False,
                 aligned_center=None,
                 psf_library=None,
                 mode=None,
                 annuli=None,
                 subsections=None,
                 numthreads=None):
        """

            Initilaizes the DiskFM class

        """

        if load_from_basis:
            numbasis = 1
            inputs_shape = 1
            dataset = None

        # make sure the dimensions have the good shape
        # and that they are numpy arrays to access their shape
        if hasattr(numbasis, "__len__"):
            numbasis = np.array(numbasis)
        else:
            numbasis = np.array([numbasis])

        if hasattr(inputs_shape, "__len__"):
            inputs_shape = np.array(inputs_shape)
        else:
            inputs_shape = np.array([inputs_shape])

        if mode is not None:
            print(mode)
            print("Warning: Argument 'mode' in pyklip.fmlib.diskfm.DiskFM class definition "\
                "is deprecated (not used and will be removed in a future version). " \
                "KLIP reduction parameters (mode, annuli and subsections) "\
                "are only defined once in klip_dataset [pyklip.parallelized]. ")

        if numthreads is not None:
            print("Warning: Argument 'numthreads' in pyklip.fmlib.diskfm.DiskFM class definition "\
                "is deprecated (not used and will be removed in a future version). " \
                "All centering are done in fm.klip_dataset ")

        if annuli is not None:
            print("Warning: Argument 'annuli' in pyklip.fmlib.diskfm.DiskFM class definition "\
                "is deprecated (not used and will be removed in a future version). " \
                "KLIP reduction parameters (mode, annuli and subsections) "\
                "are only defined once in klip_dataset [pyklip.parallelized].")

        if subsections is not None:
            print("Warning: Argument 'subsections' in pyklip.fmlib.diskfm.DiskFM class definition "\
                "is deprecated (not used and will be removed in a future version). " \
                "KLIP reduction parameters (mode, annuli and subsections) "\
                "are only defined once in klip_dataset [pyklip.parallelized].")

        super(DiskFM, self).__init__(inputs_shape, numbasis)

        self.supports_rdi = True # temporary flag until all FM classes supports RDI. 

        self.data_type = ctypes.c_double

        self.basis_filename = basis_filename
        self.save_basis = save_basis
        self.load_from_basis = load_from_basis

        if self.load_from_basis:
            # Its useless to save and load at the same time.
            self.save_basis = False
            save_basis = False

        if self.save_basis is True:
            manager = mp.Manager()
            self.klmodes_dict = manager.dict()
            self.evecs_dict = manager.dict()
            self.evals_dict = manager.dict()
            self.aligned_images_dict = manager.dict()
            self.ref_psfs_indicies_dict = manager.dict()
            self.section_ind_dict = manager.dict()

            self.radstart_dict = manager.dict()
            self.radend_dict = manager.dict()
            self.phistart_dict = manager.dict()
            self.phiend_dict = manager.dict()
            self.input_img_num_dict = manager.dict()

            self.klparam_dict = manager.dict()
        # Coords where align_and_scale places model center

        if self.load_from_basis is True:  # We want to load the FM basis

            # We load the FM basis files, before preparing the model to
            # be sure that the aligned_center is identical to the one used
            # when measuring the KL
            self.load_basis_files(psf_library=psf_library,
                                  kl_basis_file=kl_basis_file)

        else:  # We want to save the basis or just a single disk FM

            # Attributes of input
            self.inputs_shape = inputs_shape

            self.numbasis = numbasis

            # Outputs attributes
            output_imgs_shape = inputs_shape + self.numbasis.shape
            self.output_imgs_shape = output_imgs_shape

            self.PAs = dataset.PAs
            self.wvs = dataset.wvs

            self.nwvs = int(np.size(np.unique(
                self.wvs)))  # Get the number of wvls
            self.nfiles = int(self.inputs_shape[0] /
                              self.nwvs)  # Get the number of files

            # default aligned_center if none (same default as fm.parallelized):
            if aligned_center is None:
                centers = dataset.centers
                aligned_center = [
                    np.mean(centers[:, 0]),
                    np.mean(centers[:, 1])
                ]

            # define the center
            self.aligned_center = aligned_center

        # Prepare the first disk for FM
        self.update_disk(model_disk)

    def update_disk(self, model_disk):
        """
        Takes model disk and rotates it to the PAs of the input images for use as
        reference PSFS

        The disk can be either an 3D array of shape (wvs,y,x) for data of the same shape
        or a 2D Array of shape (y,x), in which case, if the dataset is multiwavelength
        the same model is used for all wavelenths.

        Args:
            model_disk: Disk to be forward modeled.

        Returns:
            None
        """

        self.model_disks = np.zeros(self.inputs_shape)

        # Extract the # of WL per files
        n_wv_per_file = self.nwvs  # Number of wavelenths per file.

        model_disk_shape = np.shape(model_disk)

        if (np.size(model_disk_shape) == 2) & (n_wv_per_file > 1):
            # This is a single WL 2D model in a multi-wl 3D data,
            # in that case we repeat this model at each WL
            self.model_disk = np.broadcast_to(model_disk, (n_wv_per_file, ) +
                                              model_disk.shape)
            model_disk_shape = np.shape(model_disk)
        else:
            # This is either a multi WL 3D model in a multi-wl 3D data
            # or a single WL 3D model in a single-wl 2D data, we do nothing
            self.model_disk = model_disk

        # Check if we have a disk at multiple wavelengths
        if np.size(model_disk_shape) > 2:  # Then it's a multiWL model
            n_model_wvs = model_disk_shape[0]

            if n_model_wvs != n_wv_per_file:
                # Both models and data are multiWL, but not the same number of WLs !
                raise ValueError(
                    """Number of wls in disk model ({0}) don't match number of wls in
                    the data ({1})""".format(n_model_wvs, n_wv_per_file))

            for k in np.arange(self.nfiles):
                for j, _ in enumerate(range(n_model_wvs)):
                    model_copy = deepcopy(model_disk[j, :, :])
                    model_copy = rotate(
                        model_copy,
                        self.PAs[k * n_wv_per_file + j],
                        self.aligned_center,
                        flipx=True,
                    )
                    model_copy[np.where(np.isnan(model_copy))] = 0.0
                    self.model_disks[k * n_wv_per_file + j, :, :] = model_copy

        else:  # This is a 2D disk model and a wl = 1 case

            for i, pa_here in enumerate(self.PAs):
                model_copy = deepcopy(model_disk)
                model_copy = rotate(model_copy,
                                    pa_here,
                                    self.aligned_center,
                                    flipx=True)
                model_copy[np.where(np.isnan(model_copy))] = 0.0
                self.model_disks[i] = model_copy

        self.model_disks = np.reshape(
            self.model_disks,
            (self.inputs_shape[0],
             self.inputs_shape[1] * self.inputs_shape[2]),
        )

    def alloc_fmout(self, output_img_shape):
        """Allocates shared memory for the output of the shared memory


        Args:
            output_img_shape: shape of output image (usually N,y,x,b)

        Returns:
            [mp.array to store FM data in, shape of FM data array]

        """

        fmout_size = int(np.prod(output_img_shape))
        fmout_shape = output_img_shape
        fmout = mp.Array(self.data_type, fmout_size)
        return fmout, fmout_shape

    def fm_from_eigen(self,
                      klmodes=None,
                      evals=None,
                      evecs=None,
                      input_img_shape=None,
                      output_img_shape=None,
                      input_img_num=None,
                      ref_psfs_indicies=None,
                      section_ind=None,
                      aligned_imgs=None,
                      radstart=None,
                      radend=None,
                      phistart=None,
                      phiend=None,
                      padding=None,
                      IOWA=None,
                      ref_center=None,
                      parang=None,
                      numbasis=None,
                      fmout=None,
                      flipx=True,
                      mode=None,
                      **kwargs):
        """
        Generate forward models using the KL modes, eigenvectors, and eigenvectors from
        KLIP. Calls fm.py functions to perform the forward modelling. If we wish to save
        the KL modes, it save in dictionnaries.

        Args:
            klmodes: unpertrubed KL modes
            evals: eigenvalues of the covariance matrix that generated the KL modes in
                    ascending order(lambda_0 is the 0 index) (shape of [nummaxKL])
            evecs: corresponding eigenvectors (shape of [p, nummaxKL])
            input_image_shape: 2-D shape of inpt images ([ysize, xsize])
            input_img_num: index of sciece frame
            ref_psfs_indicies: array of indicies for each reference PSF
            section_ind: array indicies into the 2-D x-y image that correspond to
                            this section. Note: needs be called as section_ind[0]
            radstart: radius of start of segment
            radend: radius of end of segment
            phistart: azimuthal start of segment [radians]
            phiend: azimuthal end of segment [radians]
            padding: amount of padding on each side of sector
            IOWA: tuple (IWA,OWA) IWA = Inner working angle & OWA = Outer working angle,
                    both in pixels. It defines the separation interva in which klip will
                    be run.
            ref_center: center of image
            parang: parallactic angle of input image [DEGREES]
            numbasis: array of KL basis cutoffs
            fmout: numpy output array for FM output. Shape is (N, y, x, b)
            mode: mode of the reduction ('RDI', 'ADI', 'SDI'). If RDI only, we only 
                    measure the oversubctraction
            kwargs: any other variables that we don't use but are part of the input

        Returns:
            None

        """

        # we check that the aligned_center used to center the disk (self.aligned_center)
        # If the same used to center the image in klip_dataset.
        # If not, we should not continue.
        if self.aligned_center != ref_center:
            err_string = """The aligned_center for the model {0} and for
                            the data {1} is different.
                            Change and rerun""".format(self.aligned_center,
                                                       ref_center)

            print(err_string)
            raise Exception(err_string)

        if self.load_from_basis == False:
            sci = aligned_imgs[input_img_num, section_ind[0]]
            refs = aligned_imgs[ref_psfs_indicies, :]
            refs = refs[:, section_ind[0]]
        else:
            wlstrkey = 'wl' + str(int(self.wvs[input_img_num] * 1000)).zfill(4)
            sci = self.aligned_images_dict[wlstrkey][input_img_num,
                                                     section_ind[0]]
            # in the case of load_from_basis, the images are already
            # saved in the DiskFM object, we can save a few tens of
            # Mbytes (per cpu) by not saving them
            # and just passing them to the nex function

        # use the disk model stored
        model_sci = self.model_disks[input_img_num, section_ind[0]]
        model_sci[np.where(np.isnan(model_sci))] = 0
        model_ref = self.model_disks[ref_psfs_indicies, :]
        model_ref = model_ref[:, section_ind[0]]
        model_ref[np.where(np.isnan(model_ref))] = 0
        if mode == 'RDI':
            #if only RDI we skip the deltaKL calculation since we do only over-subctraction
            delta_KL = klmodes * 0.
        else:
            # using original Kl modes and reference models, compute the perturbed KL modes
            # (spectra is already in models)
            if self.load_from_basis == False:
                delta_KL = fm.perturb_specIncluded(
                    evals,
                    evecs,
                    klmodes,
                    refs,
                    model_ref,
                    return_perturb_covar=False,
                )
            else:
                # in the case of load_from_basis, the images are already saved in the
                # DiskFM object, we can save a few tens of Mbytes (per cpu) by not 
                # saving them and just passing them to the nex function
                delta_KL = fm.perturb_specIncluded(
                    evals,
                    evecs,
                    klmodes,
                    self.aligned_images_dict[wlstrkey][ref_psfs_indicies, :]
                    [:, section_ind[0]],
                    model_ref,
                    return_perturb_covar=False,
                )

        # calculate postklip_psf using delta_KL
        postklip_psf, _, _ = fm.calculate_fm(delta_KL,
                                             klmodes,
                                             numbasis,
                                             sci,
                                             model_sci,
                                             inputflux=None)

        # write forward modelled disk to fmout (as output)
        # need to derotate the image in this step

        for thisnumbasisindex in range(np.size(numbasis)):
            fm._save_rotated_section(input_img_shape,
                                     postklip_psf[thisnumbasisindex],
                                     section_ind,
                                     fmout[input_img_num, :, :,
                                           thisnumbasisindex],
                                     None,
                                     parang,
                                     radstart,
                                     radend,
                                     phistart,
                                     phiend,
                                     padding,
                                     IOWA,
                                     ref_center,
                                     flipx=flipx)

        # We save the KL basis and params for this image and section in a dictionnaries
        if self.save_basis is True:
            # save the parameter used in KLIP-FM. We save a float to avoid pbs
            # in the saving and loading

            if mode == 'RDI':
                self.klparam_dict['isRDI'] = float(1.)
            else:
                self.klparam_dict['isRDI'] = float(0.)

            [IWA, OWA] = IOWA
            self.klparam_dict['IWA'] = float(IWA)
            self.klparam_dict['OWA'] = float(OWA)

            self.klparam_dict['input_img_shape'] = np.array(input_img_shape, dtype=float)
            self.klparam_dict['numbasis'] = float(numbasis)
            self.klparam_dict['output_imgs_shape'] = np.array(output_img_shape, dtype=float)

            # To have a single identifier for each set of aligned images,
            # we save the wavelenght in nm
            wlstrkey = 'wl' + str(int(self.wvs[input_img_num] * 1000)).zfill(4)
            self.aligned_images_dict[wlstrkey] = aligned_imgs

            # save the center for aligning the image in KLIP-FM. In practice, this
            # center will be used for all the models after we load.
            self.klparam_dict['aligned_center_x'] = float(ref_center[0])
            self.klparam_dict['aligned_center_y'] = float(ref_center[1])

            # We save information about the dataset that will be used when we load the KL basis
            self.klparam_dict['PAs'] = np.array(self.PAs, dtype=float)
            self.klparam_dict['wvs'] = np.array(self.wvs, dtype=float)

            self.klparam_dict['nwvs'] = np.array(self.nwvs, dtype=float)
            self.klparam_dict['nfiles'] = np.array(self.nfiles, dtype=float)

            # To have a single identifier for each set of section/image for the
            # dictionnaries key, we use section first pixel and image number
            curr_im = str(input_img_num).zfill(3)
            namkey = 'idsec' + str(section_ind[0][0]) + 'i' + curr_im
            # saving the KL modes dictionnaries
            self.klmodes_dict[namkey] = klmodes
            self.evals_dict[namkey] = evals
            self.evecs_dict[namkey] = evecs
            self.ref_psfs_indicies_dict[namkey] = ref_psfs_indicies
            self.section_ind_dict[namkey] = section_ind

            # saving the section delimiters dictionnaries
            self.radstart_dict[namkey] = radstart
            self.radend_dict[namkey] = radend
            self.phistart_dict[namkey] = phistart
            self.phiend_dict[namkey] = phiend
            self.input_img_num_dict[namkey] = input_img_num

    def cleanup_fmout(self, fmout):
        """
        After running KLIP-FM, we need to reshape fmout so that the numKL dimension is
        the first one and not the last. We also use this function to save the KL basis
        because it is called by fm.py at the end fm.klip_parallelized

        Args:
            fmout: numpy array of ouput of FM

        Returns:
            Same but cleaned up if necessary
        """

        # save the KL basis.
        if self.save_basis:
            self.save_kl_basis()

        # FIXME We save the matrix here it here because it is called by fm.py at the end
        # fm.klip_parallelized but this is not ideal.

        dims = fmout.shape
        fmout = np.rollaxis(
            fmout.reshape((dims[0], dims[1], dims[2], dims[3])), 3)
        return fmout

    def save_fmout(self,
                   dataset,
                   fmout,
                   outputdir,
                   fileprefix,
                   numbasis,
                   klipparams=None,
                   calibrate_flux=False,
                   pixel_weights=1,
                   **kwargs):
        """
        Uses dataset parameters to save the forward model, the output of
        fm_paralellized or klip_dataset. No returm, data are saved
        in "fileprefix" .fits files

        Args:
            dataset: an instance of Instrument.Data . Will use its
                     dataset.savedata() function to save data
            fmout: output of forward modelling.
            outputdir: directory to save output files
            fileprefix: filename prefix for saved files
            numbasis: number of KL basis vectors to use
                      (can be a scalar or list like)
            klipparams: string with KLIP-FM parameters
            calibrate_flux: if True, flux calibrate the data in the same way as
                            the klipped data
            pixel_weights: weights for each pixel for weighted mean. Leave this as a
                           single number for simple mean

        Returns:
            None

        """

        weighted = len(np.shape(pixel_weights)) > 1
        numwvs = dataset.numwvs
        fmout_spec = fmout.reshape([
            fmout.shape[0],
            fmout.shape[1] // numwvs,
            numwvs,
            fmout.shape[2],
            fmout.shape[3],
        ])  # (b, N_cube, wvs, y, x) 5-D cube

        # collapse in time and wavelength to examine KL modes
        KLmode_cube = np.nanmean(pixel_weights * fmout_spec, axis=(1, 2))
        if weighted:
            # if the pixel weights aren't just 1 (i.e., weighted case),
            # we need to normalize for that
            KLmode_cube /= np.nanmean(pixel_weights, axis=(1, 2))

        # broadband flux calibration for KL mode cube
        if calibrate_flux:
            KLmode_cube = dataset.calibrate_output(KLmode_cube, spectral=False)

        dataset.savedata(
            path.join(outputdir, fileprefix + "-fmpsf-KLmodes-all.fits"),
            KLmode_cube,
            klipparams=klipparams.format(numbasis=str(numbasis)),
            filetype="KL Mode Cube",
            zaxis=numbasis,
        )

        # if there is more than one wavelength, save also spectral cubes
        if dataset.numwvs > 1:

            KLmode_spectral_cubes = np.nanmean(pixel_weights * fmout_spec,
                                               axis=1)
            if weighted:
                # if the pixel weights aren't just 1 (i.e., weighted case), we need to
                # normalize for that.
                KLmode_spectral_cubes /= np.nanmean(pixel_weights, axis=1)

            for KLcutoff, spectral_cube in zip(numbasis,
                                               KLmode_spectral_cubes):
                # calibrate spectral cube if needed
                if calibrate_flux:
                    spectral_cube = dataset.calibrate_output(spectral_cube,
                                                             spectral=True)
                dataset.savedata(
                    path.join(
                        outputdir, fileprefix +
                        "-fmpsf-KL{0}-speccube.fits".format(KLcutoff)),
                    spectral_cube,
                    klipparams=klipparams.format(numbasis=KLcutoff),
                    filetype="PSF Subtracted Spectral Cube",
                )

    def save_kl_basis(self):
        """
        Save the KL basis and other needed parameters

        Args:
            None

        Returns:
            None

        """

        # Convert everything to np arrays and types to be safe for the saving.
        for key in self.section_ind_dict.keys():
            self.section_ind_dict[key] = np.asarray(self.section_ind_dict[key])
            self.radstart_dict[key] = float(self.radstart_dict[key])
            self.radend_dict[key] = float(self.radend_dict[key])
            self.phistart_dict[key] = float(self.phistart_dict[key])
            self.phiend_dict[key] = float(self.phiend_dict[key])

        _, file_extension = path.splitext(self.basis_filename)

        if file_extension == ".pkl":
            # transform mp dicts to normal dicts
            pkl_file = open(self.basis_filename, "wb")

            pickle.dump(dict(aligned_images_dict), pkl_file, protocol=2)

            pickle.dump(dict(klmodes_dict), pkl_file, protocol=2)
            pickle.dump(dict(evecs_dict), pkl_file, protocol=2)
            pickle.dump(dict(evals_dict), pkl_file, protocol=2)
            pickle.dump(dict(ref_psfs_indicies_dict), pkl_file, protocol=2)
            pickle.dump(dict(section_ind_dict), pkl_file, protocol=2)

            pickle.dump(dict(radstart_dict), pkl_file, protocol=2)
            pickle.dump(dict(radend_dict), pkl_file, protocol=2)
            pickle.dump(dict(phistart_dict), pkl_file, protocol=2)
            pickle.dump(dict(phiend_dict), pkl_file, protocol=2)
            pickle.dump(dict(input_img_num_dict), pkl_file, protocol=2)

            pickle.dump(dict(klparam_dict), pkl_file, protocol=2)

        elif file_extension == ".h5":
            # transform mp dicts to normal dicts
            # make a single dictionnary and save in h5

            saving_in_h5_dict = {
                'aligned_images_dict': dict(self.aligned_images_dict),
                'klmodes_dict': dict(self.klmodes_dict),
                'evecs_dict': dict(self.evecs_dict),
                'evals_dict': dict(self.evals_dict),
                'ref_psfs_indicies_dict': dict(self.ref_psfs_indicies_dict),
                'section_ind_dict': dict(self.section_ind_dict),
                'radstart_dict': dict(self.radstart_dict),
                'radend_dict': dict(self.radend_dict),
                'phistart_dict': dict(self.phistart_dict),
                'phiend_dict': dict(self.phiend_dict),
                'input_img_num_dict': dict(self.input_img_num_dict),
                'klparam_dict': dict(self.klparam_dict),
            }

            _save_dict_to_hdf5(saving_in_h5_dict, self.basis_filename)

            del saving_in_h5_dict

        else:
            raise ValueError(file_extension +
                             """ is not a possible extension. Filenames can
                haves 2 recognizable extension2: .h5 and .pkl""")

    def load_basis_files(self, psf_library=None, kl_basis_file=None):
        """
        Loads in previously saved basis files and sets variables for fm_from_eigen

        Args:
            dataset: an instance of Instrument.Data, after fm.klip_dataset.
                     Allow me to pass in the structure some correction parameters
                     set by fm.klip_dataset, such as IWA, OWA, aligned_center.
                     KL basis and sections information are passed via global variables

        Returns:
            None
        """
        if kl_basis_file is not None:
            file_extension = ""
        else:
            _, file_extension = path.splitext(self.basis_filename)
        manager = mp.Manager()

        # Load in file
        if file_extension == ".pkl":
            pkl_file = open(self.basis_filename, "rb")
            if version_info.major == 3:
                # Using encoding='latin1' is required for unpickling NumPy arrays
                # and instances of datetime, date and time pickled by Python 2.
                self.aligned_images_dict = manager.dict(
                    pickle.load(pkl_file, encoding="latin1"))

                self.klmodes_dict = manager.dict(
                    pickle.load(pkl_file, encoding="latin1"))
                self.evecs_dict = manager.dict(
                    pickle.load(pkl_file, encoding="latin1"))
                self.evals_dict = manager.dict(
                    pickle.load(pkl_file, encoding="latin1"))
                self.ref_psfs_indicies_dict = manager.dict(
                    pickle.load(pkl_file, encoding="latin1"))
                self.section_ind_dict = manager.dict(
                    pickle.load(pkl_file, encoding="latin1"))

                self.radstart_dict = manager.dict(
                    pickle.load(pkl_file, encoding="latin1"))
                self.radend_dict = manager.dict(
                    pickle.load(pkl_file, encoding="latin1"))
                self.phistart_dict = manager.dict(
                    pickle.load(pkl_file, encoding="latin1"))
                self.phiend_dict = manager.dict(
                    pickle.load(pkl_file, encoding="latin1"))
                self.input_img_num_dict = manager.dict(
                    pickle.load(pkl_file, encoding="latin1"))

                self.klparam_dict = pickle.load(pkl_file, encoding="latin1")

            else:
                self.aligned_images_dict = manager.dict(pickle.load(pkl_file))

                self.klmodes_dict = manager.dict(pickle.load(pkl_file))
                self.evecs_dict = manager.dict(pickle.load(pkl_file))
                self.evals_dict = manager.dict(pickle.load(pkl_file))
                self.ref_psfs_indicies_dict = manager.dict(
                    pickle.load(pkl_file))
                self.section_ind_dict = manager.dict(pickle.load(pkl_file))

                self.radstart_dict = manager.dict(pickle.load(pkl_file))
                self.radend_dict = manager.dict(pickle.load(pkl_file))
                self.phistart_dict = manager.dict(pickle.load(pkl_file))
                self.phiend_dict = manager.dict(pickle.load(pkl_file))
                self.input_img_num_dict = manager.dict(pickle.load(pkl_file))

                self.klparam_dict = pickle.load(pkl_file)

        else:

            if file_extension == ".h5":
                kl_basis_file = _load_dict_from_hdf5(self.basis_filename)

        self.aligned_images_dict = manager.dict(
            kl_basis_file['aligned_images_dict'])

        self.klmodes_dict = manager.dict(kl_basis_file['klmodes_dict'])
        self.evecs_dict = manager.dict(kl_basis_file['evecs_dict'])
        self.evals_dict = manager.dict(kl_basis_file['evals_dict'])
        self.ref_psfs_indicies_dict = manager.dict(
            kl_basis_file['ref_psfs_indicies_dict'])
        self.section_ind_dict = manager.dict(kl_basis_file['section_ind_dict'])

        self.radstart_dict = manager.dict(kl_basis_file['radstart_dict'])
        self.radend_dict = manager.dict(kl_basis_file['radend_dict'])
        self.phistart_dict = manager.dict(kl_basis_file['phistart_dict'])
        self.phiend_dict = manager.dict(kl_basis_file['phiend_dict'])
        self.input_img_num_dict = manager.dict(
            kl_basis_file['input_img_num_dict'])

        self.klparam_dict = kl_basis_file['klparam_dict']

        del kl_basis_file

        # read key name for each section and image
        self.dict_keys = sorted(self.klmodes_dict.keys())

        # load parameters of the correction that fm.klip_dataset produced
        # when we saved the FM basis.

        self.isRDI = (self.klparam_dict['isRDI'] == 1)
        self.IWA = self.klparam_dict['IWA']
        self.OWA = self.klparam_dict['OWA']

        numbasis = self.klparam_dict['numbasis'].astype(int)
        if hasattr(numbasis, "__len__"):
            numbasis = np.array(numbasis)
        else:
            numbasis = np.array([numbasis])

        self.numbasis = numbasis

        self.aligned_center = [
            self.klparam_dict['aligned_center_x'],
            self.klparam_dict['aligned_center_y'],
        ]

        output_imgs_shape = tuple(
            self.klparam_dict['output_imgs_shape'].astype(int))

        self.output_imgs_shape = output_imgs_shape

        # Those are loaded to avoid depending at all on the dataset when we load the KL basis
        self.PAs = self.klparam_dict['PAs']
        self.wvs = self.klparam_dict['wvs']

        self.nwvs = int(self.klparam_dict['nwvs'])  # Get the number of wvls
        self.nfiles = int(
            self.klparam_dict['nfiles'])  # Get the number of wvls

        dim_frame = self.klparam_dict['input_img_shape']
        self.inputs_shape = np.array(
            (self.nfiles * self.nwvs, int(dim_frame[0]), int(dim_frame[1])))

        # After loading it, we stop saving the KL basis to avoid saving it every time
        # we run self.fm_parallelize.
        self.save_basis = False

    def fm_parallelized(self):
        """
        Functions like fm.klip_dataset, but it uses previously measured KL modes,
        section positions, and klip parameter to return the forward modelling.
        Do not save fits.

        Args:
            None

        Returns:
            fmout_np, a numpy array, output of forward modelling
                    * if N_wl = 1, size is [n_KL,x,y]
                    * if N_wl > 1, size is  [n_KL,N_wl,x,y]

        """

        fmout_data, fmout_shape = self.alloc_fmout(self.output_imgs_shape)
        fmout_np = fm._arraytonumpy(fmout_data,
                                    fmout_shape,
                                    dtype=self.data_type)
        # this line is added to be able to use fm._save_rotated_section
        # which uses global var outputs_shape
        fm.outputs_shape = self.output_imgs_shape

        wvs = self.wvs

        if self.isRDI:
            mode = 'RDI'
        else:
            mode = None
            # We are only interested in the RDI mode
            # if not we don't care since it does not have an
            # impact at this point

        for key in self.dict_keys:  # loop pver the sections/images

            img_num = self.input_img_num_dict[key]

            # To have a single identifier for each set of aligned images,
            # we save the wavelenght in nm
            wl_here = wvs[img_num]
            wlstr = 'wl' + str(int(wl_here * 1000)).zfill(4)

            # in load mode, we do not pass aligned_images_dict
            # because it is already in the class to
            # save memory
            self.fm_from_eigen(
                klmodes=self.klmodes_dict[key],
                evals=self.evals_dict[key],
                evecs=self.evecs_dict[key],
                input_img_shape=[self.inputs_shape[1], self.inputs_shape[2]],
                output_img_shape=self.output_imgs_shape,
                input_img_num=img_num,
                ref_psfs_indicies=self.ref_psfs_indicies_dict[key],
                section_ind=self.section_ind_dict[key],
                radstart=self.radstart_dict[key],
                radend=self.radend_dict[key],
                phistart=self.phistart_dict[key],
                phiend=self.phiend_dict[key],
                padding=0.0,
                IOWA=(self.IWA, self.OWA),
                ref_center=self.aligned_center,
                parang=self.PAs[img_num],
                numbasis=self.numbasis,
                fmout=fmout_np,
                mode=mode)

        # put any finishing touches on the FM Output
        fmout_np = fm._arraytonumpy(fmout_data,
                                    fmout_shape,
                                    dtype=self.data_type)
        fmout_np = self.cleanup_fmout(fmout_np)

        # Check if we have a disk model at multiple wavelengths.
        # If true then it's a non- collapsed spec mode disk and we need to reorganise
        # fmout_return. We use the same mean so that it corresponds to
        # klip image-speccube.fits produced by.fm.klip_dataset
        if np.size(np.shape(self.model_disk)) > 2:

            n_wv_per_file = self.nwvs  # Number of WL per file.

            # Collapse across all files, keeping the wavelengths intact.
            fmout_return = np.zeros([
                np.size(self.numbasis),
                n_wv_per_file,
                self.inputs_shape[1],
                self.inputs_shape[2],
            ])
            for i in np.arange(n_wv_per_file):
                fmout_return[:, i, :, :] = np.nansum(
                    fmout_np[:, i::n_wv_per_file, :, :], axis=1) / float(
                        self.nfiles)

        else:
            # If false then this is a collapsed-spec mode or pol mode: collapsed
            # across all files
            fmout_return = np.nanmean(fmout_np, axis=1)

        return fmout_return


##############################################################################
###### 4 routines to save and load h5 in dictionnaries
##############################################################################


def _save_dict_to_hdf5(dic, filename):
    """
    Saving a nested dictionnary into a h5 file

    Args:
        dic: the dictionnary to file
        filename: the filename of the h5 where it will be saved

    Returns:
        None

    """
    with h5py.File(filename, "w") as h5file:
        _recursively_save_dict_contents_to_group(h5file, '/', dic)


def _load_dict_from_hdf5(filename):
    """
    Load a dictionnary from a h5 file

    Args:
        filename: the filename of the h5

    Returns:
        the dictionnary exctracted

    """

    with h5py.File(filename, "r") as h5file:
        return _recursively_load_dict_contents_from_group(h5file, '/')


def _recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Recursively explore the dictionnary for saving it

    Args:
        h5file: the file in which we save, opened with h5py.File
        path: the separator to aggregate the keys. Should not be set to a value that
            is likely to be in the dictionnary keys already
        dic: the dictionnary to deconstruct

    Returns
        None

    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, int, float, str, bytes, np.number)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            _recursively_save_dict_contents_to_group(h5file, path + key + '/',
                                                     item)
        else:
            raise ValueError("Cannot save {0} type in h5 (key = {1})".format(
                type(item), path + key))


def _recursively_load_dict_contents_from_group(h5file, path):
    """
    Recursively explore the dictionnary for loading it

    Args:
        h5file: the file from which we load, opened with h5py.File
        path: the separator to aggregate the keys. Should be the same one used in
        _recursively_save_dict_contents_to_group
    Returns
        the rebuilt dictionnary
    """

    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = _recursively_load_dict_contents_from_group(
                h5file, path + key + '/')
    return ans
