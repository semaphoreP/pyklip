.. _diskfm_gpi-label:

Disk Foward Modelling (DiskFM)
=====================================================
This tutorial presents how to use the forward modelling routines specific to disk modelling
and disk parameter retrieval.

Attribution
-----------
If you use this feature, please cite:

 * `Mazoyer, J., Arriaga, P., Hom, J., et al. 2020, Proc. SPIE, 11447, 1144759. <https://ui.adsabs.harvard.edu/abs/2020SPIE11447E..59M/abstract>`_

Why DiskFM?
--------------------------
As noted in `Pueyo (2016) <http://arxiv.org/abs/1604.06097>`_, "*in practice Forward
Modeling with disks is complicated by the
fact that [it] cannot be simplified using a simple PSF as the astrophysical model:
every hypothetical disk morphology must be explored*". Indeed, because of their complex
geometries, the forward modelling have to be repeated a lot of time on disks
with slightly different parameters. All these geometries are then compared
to the klipped reduced image of the data, within an MCMC or a Chi-square wrapper.

However, once measured for a set of reduction parameters, the Karhunen-Loeve (KL) basis
do not change. One can save the KL vectors in a file once so they do not have to be
recomputed every time. For a new disk model, the forward modelling is therfore only a
array reformating and a matrix multiplication, which can be optimized to be only a few
seconds. These routines are implemented in PyKLIP and showed on this page. DiskFM currently
only supports KLIP ADI, SDI ADI+SDI and RDI reductions (but currently not RDI+ADI/SDI or NMF).

DiskFM Requirements
--------------------------
`diskFM` on a single model can be done on a personnal computer. However, the full parameter space
exploration with the Chi-square or MCMC wrapper (out of the scope of this this tutorial) can be
computationally very intensive, taking easily a few days, even parallized on a large
server.

You also need the following pieces of data to forward model the data:

* A model of disk (this tutorial do not include disk modelling)
* The instrument PSF or a model of the PSF
* A set of to run PSF subtraction on



Set up
--------------------------
First import an instrument data set and convolve your 2D disk model by the instrument PSF:

.. code-block:: python

    import glob
    import numpy as np
    import pyklip.instruments.GPI as GPI
    from astropy.convolution import convolve
    from pyklip.fmlib.diskfm import DiskFM
    import pyklip.fm as fm

    # read in the data into a dataset
    filelist = sorted(glob.glob("path/to/dataset/*.fits"))
    dataset = GPI.GPIData(filelist)

    # convolved the 2D disk model
    disk_model_convolved = convolve(disk_model,instrument_psf, boundary = 'wrap')


Simple disk Forward Modelling
--------------------------
This code then shows how to initialize the `DiskFM` object and to do a forward modelling:

.. code-block:: python

    numbasis = [3, 10, 20] # different KL numbers we applied to the disk.
    aligned_center=[140, 140] # indicate the position of the star

    # initialize the DiskFM object
    diskobj = DiskFM(dataset.input.shape, numbasis, dataset, disk_model_convolved,
                    aligned_center=aligned_center)


To run the forward modelling, just run:

.. code-block:: python

    fm.klip_dataset(dataset, diskobj, outputdir="output_path/", fileprefix="my_favorite_disk",
                    numbasis=numbasis, maxnumbasis=100, aligned_center=aligned_center,
                    mode='ADI', annuli=1, subsections=1, movement=1)


the code will save two fits files in `outputdir`, containing the klipped data and the
associated disk forward model.

Most of the parameters implemented for psf forward model KLIP correction with pyklip can be used (see
`Picking KLIP Parameters for Disks <https://pyklip.readthedocs.io/en/latest/klip_gpi.html#picking-klip-parameters-for-disks>`_)
with the following exceptions:

* spectrum specific keywords (spectrum, flux_overlap, calibrate_flux)
* specific correction modes filtering the data (corr_smooth, highpass)
* other specific correction parameters (N_pix_sector, padding, annuli_spacing)

Mode parameter can be set only to `'ADI'`, `'SDI'` and `'ADI+SDI'`.`aligned_center` is
the position were the klip reduction will center the reduced image.
The code will raise an error if it is not set to the position to which you set the star
in your model.


DiskFM for MCMC or Chi-Square
--------------------------
For an MCMC or Chi-Square you can create the KL basis and then save them to forward
model multiple models on a dataset without recomputing them every time.
If you would like save the KL basis then you will need to signal it during
the initialization of the `DiskFM` object, then apply `fm.klip_dataset` to measure and
ave the forward model KL basis and parameters:

.. code-block:: python

    diskobj = DiskFM(dataset.input.shape, numbasis, dataset,
                    disk_model_convolved, aligned_center=aligned_center,
                    basis_filename = 'path/to/dir/klip-basis.h5', save_basis = True)


    fm.klip_dataset(dataset, diskobj, outputdir="output_path/", fileprefix="my_favorite_disk",
                    numbasis=numbasis, maxnumbasis=100, aligned_center=aligned_center,
                    mode='ADI', annuli=1, subsections=1, movement=1)


Then, in any python session you can create a disk object and you can forward model disks
with the loaded KL basis vectors without needing to measure this basis.
The disk forward model will be output to `fmout`:

.. code-block:: python

    diskobj = DiskFM(dataset.input.shape, numbasis, dataset,
                    disk_model_convolved, aligned_center=aligned_center,
                    basis_filename='path/to/dir/klip-basis.h5', load_from_basis=True)

    # do the forward modelling on a new model
    new_disk_model_convolved=convolve(new_disk_model,instrument_psf, boundary='wrap')
    diskobj.update_disk(new_disk_model_convolved)
    fmout=diskobj.fm_parallelized()

    # do the forward modelling on a third model
    third_disk_model_convolved=convolve(third_disk_model,instrument_psf, boundary='wrap')
    diskobj.update_disk(third_disk_model_convolved)
    fmout=diskobj.fm_parallelized()

These last 3 lines are specifically what should be repeated withinin the MCMC
or Chi-Square wapper.


Note that even if you have already created a `DiskFM` object to save the FM
(*ie* even if you have runned `diskFM` with `save_basis = True`) in this python session,
you still need to re-create the `DiskFM` object and load it (*ie*, you still
need `diskFM` with `load_from_basis = True`).

In previous version, the dataset itself (input images) were not saved in the .h5 files, only the KL 
coeficients. This caused problems because you could run the same KL coefficients with slightly different
datasets (for example the order of the frames were not identical) the code would run but provide wrong forward models.
This has now beed solved and all the information necessary is saved inside the .h5 file, including
intial frames and reduction paramters.


Speeding up DiskFM
--------------------------
The time is a key element here if you want to produce hundreds of thousands of forward
modelling models. A smart choice of pyklip parameters can reduce the time for a single
disk forward model:

* use OWA to limit only in the zone where the disk is.
* limit the number of sections (small annuli and subsections number).
* reduce the number of wavelengths. We recall this very usefull pyklip function to rebin
  quickly the number of wavelength, which should be applied immediatly after loading
  the dataset:

.. code-block:: python

    dataset.spectral_collapse(collapse_channels=1, align_frames=True)

* determine the best KL number parameters in advance and use only one, e.g.:

.. code-block:: python

    numbasis = [3]


Finally, due to the fact that numpy also parallelizes linear algebra routines
across multiple cores, performance can actually sharply decrease when multiprocessing
in a mcmc. Please read `Note on parallelized performance
<https://pyklip.readthedocs.io/en/latest/install.html#note-on-parallelized-performance>`_
on this subject.


Multiwavelength DiskFM
--------------------------
If you put a multi-wavelenght dataset (e.g. IFS), the code will produce a multi-wavelenght forward
model. In that case, you can use a simple 2D model for the disk and the code will duplicate this model
and apply the forward modelling separately on each of those at every wavelengths. Or you can use a 3D model
(n_wl, x, y) and the code will apply the forward modelling separately on each of those at every wavelengths.

Alhtough everything we said in the previous sections on saving and loading the KL basis still
apply multiwavelength disk forward modelling is long (it can take up to a few minutes or hours
for a single forward modelling depending on the number of wavelengths) and we do not
recommand to use this in an MCMC wrapper.

Full DiskFM tutorial
--------------------------
We recall all the steps in a single block

.. code-block:: python

    import glob
    import numpy as np
    import pyklip.instruments.GPI as GPI
    from astropy.convolution import convolve
    from pyklip.fmlib.diskfm import DiskFM
    import pyklip.fm as fm

    # read in the data into a dataset
    filelist = sorted(glob.glob("path/to/dataset/*.fits"))
    dataset = GPI.GPIData(filelist)

    # in case of multiWL data, you might want to stack them first to speed things up
    dataset.spectral_collapse(collapse_channels=1, align_frames=True)

    numbasis = [3] # different KL numbers we applied to the disk.
    aligned_center=[140, 140] # indicate the position of the star

    # convolved the disk model
    disk_model_convolved = convolve(disk_model,instrument_psf, boundary = 'wrap')

    # initialize the DiskFM class
    diskobj = DiskFM(dataset.input.shape, numbasis, dataset,
                    disk_model_convolved, aligned_center=aligned_center,
                    basis_filename = 'path/to/dir/klip-basis.pkl', save_basis = True)

    # run klip to find and save FM basis
    fm.klip_dataset(dataset, diskobj, outputdir="path/", fileprefix="my_favorite_disk",
                    numbasis=numbasis, maxnumbasis=100, aligned_center=aligned_center,
                    mode='ADI', annuli=2, subsections=1, minrot=3)


    # ----------------------------------------------------------------------------
    # starting from here you can close the session and reopen later if you want
    # ----------------------------------------------------------------------------

    # load Klip parameters and FM basis
    diskobj = DiskFM(dataset.input.shape, numbasis, dataset,
                    disk_model_convolved, aligned_center=aligned_center,
                    basis_filename='path/to/dir/klip-basis.h5', load_from_basis=True)

    # do the forward modelling on a new model
    new_disk_model_convolved=convolve(new_disk_model,instrument_psf, boundary='wrap')
    diskobj.update_disk(new_disk_model_convolved)
    fmout=diskobj.fm_parallelized()



