.. _fmmf-label:

Forward Model Matched Filter (FMMF)
==================================

This tutorial will provide the necessary steps to run the Forward Model Matched Filter (FMMF)
that is described in `Ruffio et al. (2016) <https://arxiv.org/pdf/1705.05477.pdf>`_.

Attribution
-----------
If you use this feature, please cite:

 * `Ruffio, J.-B., Macintosh, B., Wang, J. J., et al. 2017, ApJ, 842, 14. <https://ui.adsabs.harvard.edu/abs/2017ApJ...842...14R/abstract>`_

Why FMMF?
-----------------

Speckle subtraction algorithms like PCA or LOCI are not planet detection algorithms. An additional step needs to be performed to compute the SNR.
The SNR can simply be computed from aperture photometry divided by the standard deviation of the noise calculated in an annulus at the same separation, but this is not the optimal approach.

In signal processing, a matched filter is the linear filter maximizing the Signal to Noise Ratio (SNR) of a known signal in the presence of additive noise.

Matched filters are used in Direct imaging to detect point sources using the expected shape of the planet Point Spread
Function (PSF) as a template.

Detection of directly imaged exoplanets is challenging since PSF subtraction algorithms (like pyKLIP)
distort the PSF of the planet. `Pueyo (2016) <http://arxiv.org/abs/1604.06097>`_ provide a technique to
forward model the PSF of a planet through KLIP.

FMMF uses this forward model as the template of the matched filter therefore improving the planet sensitivity of the
algorithm compared to a conventional approach.

FMMF Requirements
-----------------

FMMF is computationally very intensive. It will take up to a few days to run a typical GPI dataset for example on a
basic cluster node (but it's worth it! ;)).

You also need the following pieces of data to forward model the data.

* Data to run PSF subtraction on
* A model or data of the instrumental PSF
* For IFS data, an estimate of the spectrum of the planet that one expects to find.

Running FMMF (Example using GPI data)
-----------------
There are 3 steps in running FMMF:

* Read the data using an instrument class
* Define the MatchedFilter object
* Call ``klip_dataset`` to run the reduction.

We use the GPI test dataset, which is included in pyklip, to illustrate the method. See :ref:`genericdata-label` for a way to generate a dataset object for any instrument.
Here, we can high-pass-filter the data using the ``highpass`` keyword when defining the ``GPI.GPIData`` object.
Experience has shown that the high-pass filter is an important step in the reduction, which will need to be performed by the user when using a different instrument.

.. code-block:: python

    if __name__ ==  '__main__':
        inputDir = "path/to/dataset/"
        outputDir = "path/to/save/dir/"

        from pyklip.instruments import GPI
        import glob
        import os

        dataset = GPI.GPIData(glob.glob(os.path.join(inputDir,"*.fits")), highpass=True)

In order to perform a matched filter, we need a model of the planet PSF. For GPI, there is a built-in routine in
the dataset object to calculate it from the satellite spots.
The PSF model should also be high-pass filtered to better represent the signal in data, although we skip this step here for simplicity.

.. code-block:: python

    if __name__ ==  '__main__':
        #[...]
        import numpy as np
        dataset.generate_psf_cube(20,same_wv_only=True)
        PSF_cube_arr = dataset.psfs # Shape is [nwvs,ny,nx]
        PSF_cube_wvs = np.unique(dataset.wvs)


In addition of the planet PSF, we need to assume a spectrum for the planet.
The spectrum should be defined such that:

* it has the total flux of the star, ie correspond to a contrast of 1.
* it represents the total integrated flux of the PSF and not the simple peak value.
* it should be multiplied by the atmospheric and instrumental transmission.
* It has the same size as the number of images in the dataset.
* Note that ``MatchedFilter`` expects a list, so make it a list of one spectrum like this: ``[spectrum_vec]``.

We now need to define the ``fmlib`` object, which is the object that will tell ``klip_dataset`` the kind of reduction that we want to do (ie, FMMF).
``numbasis`` is the number of KL modes to be used. ``maxnumbasis`` is the number of frames to be selected from the dataset and used to compute the covariance matrix.

.. code-block:: python

    if __name__ ==  '__main__':
        #[...]
        # Flat spectrum
        spectrum_vec = np.ones((dataset.input.shape[0],))
        # Number KL modes used for KLIP
        numbasis = [5]
        # Number of images in the reference library
        maxnumbasis = [10]

        # Build the FM class to do matched filter
        import pyklip.fmlib.matchedFilter as mf
        fm_class = mf.MatchedFilter(dataset.input.shape,numbasis,
                                         PSF_cube_arr, PSF_cube_wvs,
                                         [spectrum_vec])

FMMF is computationally extremely expensive. We recommend running it on computer or nodes with 16+ cores and 64+GB or RAM depending on the size of the dataset.
The example below can be tested on a laptop, but it will still likely take around 30min.
Before starting the reduction, we still need to define the subdivision of the field of view; the sectors. we restrict the reduction to the separation of the planet using ``annulus_bounds = [[28,33]]``.
The sectors are then defined to contain a number of pixels that is as close as possible to ``N_pix_sector = 200``. The annulis will be sub-divided according to that constraint.
A significant difference compared to normal klip is that the sectors needs to be padded for the matched filter.
Set the value of ``padding`` to half the width of the PSF array.

.. code-block:: python

    if __name__ ==  '__main__':
        #[...]
        import pyklip.fm as fm
        prefix = "betpic-131210-J_GPI" #used in the filename of the outputs
        annulus_bounds = [[28,33]]# This annulus is centered at the location of bet Pic b in the test dataset
        N_pix_sector = 200
        padding = PSF_cube_arr.shape[1]//2
        movement = 2.0
        fm.klip_dataset(dataset, fm_class, outputdir=outputDir, fileprefix=prefix, numbasis=numbasis,
                        annuli=annulus_bounds, N_pix_sector=N_pix_sector, padding=padding, movement=movement)

This function will produce 6 output files with the following extensions:

* ``[...]FMMF-KL#.fits``: This is the matched-filter map, which should be proportional to the S/N of the planet.
* ``[...]FMCont-KL#.fits``: "Cont" stands for contrast. This is the estimated planet to star flux ratio from the maximum likelihood. Note that the estimated flux ratio, might still be subject to a certain amount of self- and over-subtraction and should therefore be calibrated with simulated planet injection and recovery.
* ``[...]FMCC-KL#.fits``: This is similar to FMMF, but the difference is that the local estimate of the standard deviation is not used as a weight when combining all the data together in a single map.
* ``[...]FMN_pix-KL#.fits``: This map includes the number of valid pixels used in the dataset at any planet location.
* ``[...]klipped-KL#-speccube.fits``: This is a spectral cube of the klipped reduction.
* ``[...]klipped-KL#-KLmodes-all.fits``: This is the 2D image resulting from flattening the klipped spectral cube.

Note that FMMF will not perform optimally on high SNR objects since the forward model will no longer be accurate.
The forward model is indeed only a linear approximation of the speckle subtracted planet PSF, which will break when too little ADI/SDI diversity is present or if the planet is too bright.

Because the noise is assumed to be uncorrelated, the estimated SNR in the FMMF map is overestimated.
It needs to be renormalized. This can be done using the function ``get_image_stat_map_perPixMasking``.
This function is designed to compute the standard deviation of the image in concentric annuli.
It will repeat this operation locally for each pixel in the image by masking the neighboring pixels.
The goal is to prevent a putative planet to contaminate its own SNR estimation by artificially increasing the empirical standard deviation.
If ``type="stddev"``, the function returns the standard deviation map, and if ``type="SNR"``, the SNR map is returned.
``mask_radius`` is the radius of the mask around each pixel, and ``Dr`` is the radial width of the annulus.


.. code-block:: python

    if __name__ ==  '__main__':
        #[...]
        filename = os.path.join(outputDir,"betpic-131210-J_GPI-FMMF-KL5.fits") # Change filename if needed
        import astropy.io.fits as pyfits
        hdulist = pyfits.open(filename)
        FMMF = hdulist[1].data
        hdulist.close()

        from pyklip.kpp.stat.statPerPix_utils import get_image_stat_map_perPixMasking
        FMMF_SNR = get_image_stat_map_perPixMasking(FMMF,
                                         mask_radius = 7,
                                         Dr = 2,
                                         type = "SNR")

The output ``FMMF_SNR`` is the calibrated SNR map that can be used for planet detection.

