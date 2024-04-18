.. _fmspect-label:

Spectrum Extraction using extractSpec FM
========================================

This document describes with an example how to use KLIP-FM to extract 
a spectrum, described in 
`Pueyo et al. (2016) <http://adsabs.harvard.edu/abs/2016ApJ...824..117P>`_ 
to account the effect of the companion signal in the reference library
when measuring its spectrum.

Attribution
-----------
If you use this feature, please cite:

 * `Greenbaum, A. Z., Pueyo, L., Ruffio, J.-B., et al. 2018, AJ, 155, 226. <https://ui.adsabs.harvard.edu/abs/2018AJ....155..226G/abstract>`_

Set up:
---------

Here we will just read in the dataset and grab the instrumental PSF. The example code here shows how it is done with GPI, 
but you will want to refer to the :ref:`instruments-label` for the instrument you are working. 
As the code notes, it is important what the units of your instrumental PSF is in, as the
code will return the spectrum relative to the input PSF model. 

::

    import glob
    import numpy as np
    import pyklip.instruments.GPI as GPI
    import pyklip.fmlib.extractSpec as es
    import pyklip.fm as fm
    import pyklip.fakes as fakes
    import matplotlib.pyplot as plt

    files = glob.glob("\path\to\dataset\*.fits")
    dataset = GPI.GPIData(files, highpass=True)
    # Need to specify a model PSF (either via this method, or any other way)
    model_psfs = dataset.generate_psf_cube(20) 
    # in this case model_psfs has shape (N_lambda, 20, 20)
    # The units of your model PSF are important, the return spectrum will be
    # relative to the input PSF model, see next example

    ###### Useful values based on dataset ######
    N_frames = len(dataset.input)
    N_cubes = np.size(np.unique(dataset.filenums))
    nl = N_frames // N_cubes



Calibrating stellar flux for GPI example:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Converting to contrast units for GPI data is done using the flux of the satellite spots.
The GPI dataset object has attribute spot_flux that represent the average peak flux of
the four spots. The normalization factor is computed by dividing the spot flux 
spectrum by the ratio between the stellar flux and the spot flux (stored in 
spot_ratio) and adjusting for the ratio between the peak and the sum of the spot
PSF. 

For any instrument you can scale your model PSF by its respective calibration
factors if the model PSF is not already scaled to be the flux of the star. Alternatively,
you can choose to skip this step and calibrate your spectrum into astrophysical units as the
very end. 

GPI Example::

    # First set up a PSF model and sums -- this is necessary for GPI because 
    # dataset.spot_flux contains peak values of the satellite spots and we 
    # have to correct for the full aperture. 
    PSF_cube = dataset.psfs
    model_psf_sum = np.nansum(PSF_cube, axis=(1,2))
    model_psf_peak = np.nanmax(PSF_cube, axis=(1,2))
    # Now divide the sum by the peak for each wavelength slice
    aper_over_peak_ratio = model_psf_sum/model_psf_peak

    # star-to-spot calibration factor
    band = dataset.prihdrs[0]['APODIZER'].split('_')[1]
    spot_to_star_ratio = dataset.spot_ratio[band]

    spot_peak_spectrum = \
        np.median(dataset.spot_flux.reshape(len(dataset.spot_flux)//nl, nl), axis=0)
    calibfactor = aper_over_peak_ratio*spot_peak_spectrum / spot_to_star_ratio

    # calibrated_PSF_model is the stellar flux in counts for each wavelength
    calibrated_PSF_model = PSF_cube*calibfactor

This is your model_psf for generating the forward model and will return the 
spectrum in contrast units relative to the star. 


Computing the forward model and recovering the spectrum with invert_spect_fmodel
--------------------------------------------------------------------------------
We will use the `ExtractSpec` class to forward model the PSF of the planet and 
the `invert_spect_fm` function in `pyklip.fmlib.extractSpec` to recover the spectrum.
`invert_spect_fm` returns a spectrum in units relative to the input PSF. 

These are the numbers you change::

    ###### parameters you specify ######
    pars = (45, 222) # replace with known separation and pa of companion
    planet_sep, planet_pa = pars
    numbasis = [50,] # "k_klip", this can be a list of any size.
                     # a forward model will be computed for each element.
    num_k_klip = len(numbasis) # how many k_klips running
    maxnumbasis = 100 # Max components to be calculated
    movement = 2.0 # aggressiveness for choosing reference library
    stamp_size = 10.0 # how big of a stamp around the companion in pixels
                      # stamp will be stamp_size**2 pixels
    numthreads=4 # number of threads, machine specific
    spectra_template = None # a template spectrum, if you want
    
Generating the forward model with pyKLIP::

    ###### The forward model class ######
    fm_class = ExtractSpec(dataset.input.shape,
                           numbasis,
                           planet_sep,
                           planet_pa,
                           calibrated_PSF_model,
                           np.unique(dataset.wvs),
                           stamp_size = stamp_size)

    ###### Now run KLIP! ######
    fm.klip_dataset(dataset, fm_class,
                    fileprefix="fmspect",
                    annuli=[[planet_sep-stamp_size,planet_sep+stamp_size]],
                    subsections=[[(planet_pa-stamp_size)/180.*np.pi,\
                                  (planet_pa+stamp_size)/180.*np.pi]],
                    movement=movement,
                    numbasis = numbasis, 
                    maxnumbasis=maxnumbasis,
                    numthreads=numthreads,
                    spectrum=spectra_template,
                    save_klipped=True, highpass=True,
                    outputdir="\path\to\output")

    # Forward model is stored in dataset.fmout, this is how it is organized:
    # the klipped psf
    klipped = dataset.fmout[:,:,-1,:]
    # The rest is the forward model, dimensions:
    # [num_k_klip, N_frames, N_frames,  stamp_size*stamp_size]
    # If numbasis is a list, the first dimension will be the size of that list,
    # a forward model calculated at each value of numbasis.

Now you can recover the spectrum::

    # If you want to scale your spectrum by a calibration factor:
    units = "scaled"
    scaling_factor = my_calibration_factor
    #e.g., for GPI this could be the star-to-spot ratio
    # otherwise, the defaults are:
    units = "natural" # (default) returned relative to input PSF model
    scale_factor=1.0 # (default) not used if units not set to "scaled"


    exspect, fm_matrix = es.invert_spect_fmodel(dataset.fmout, dataset, units=units,
                                                scaling_factor=scaling_factor, 
                                                method="leastsq")
    # method indicates which matrix inversion method to use, they all tend
    # to yield similar results when things are well-behaved. Here are the options:
    # "JB" matrix inversion adds up over all exposures, then inverts
    # "leastsq" uses a leastsq solver.
    # "LP" inversion adds over frames and one wavelength axis, then inverts
    # (LP is not generally recommended)

The units of the spectrum, FM matrix, and klipped data are all in raw data units
in this example. Calibration of instrument and atmospheric transmmission and 
stellar spectrum can be done via the input PSF model and optionally applying 
the scaling factor to invert_spect_fmodel. It can also be done after extracting
the spectrum. 

Simulating + recovering a simulated source
------------------------------------------

Example::

    # PSF model template for each cube observation, copies of the PSF model:
    inputpsfs = np.tile(calibrated_PSF_model, (N_cubes, 1, 1))
    bulk_contrast = 1e-2
    fake_psf = inputpsfs*bulk_contrast
    fake_flux = bulk_contrast*np.ones(dataset.wvs.shape)
    #for ll in range(N_cubes):
    #    fake_flux[ll*nl:(ll+1)*nl] = exspect[0, :]
    pa = planet_pa+180

    tmp_dataset = GPI.GPIData(files, highpass=False)
    fakes.inject_planet(tmp_dataset.input, tmp_dataset.centers, fake_psf,\
                                    tmp_dataset.wcs, planet_sep, pa)

    fm_class = es.ExtractSpec(tmp_dataset.input.shape,
                               numbasis,
                               planet_sep,
                               pa,
                               calibrated_PSF_model,
                               np.unique(dataset.wvs),
                               stamp_size = stamp_size)

    fm.klip_dataset(tmp_dataset, fm_class,
                        fileprefix="fakespect",
                        annuli=[[planet_sep-stamp_size,planet_sep+stamp_size]],
                        subsections=[[(pa-stamp_size)/180.*np.pi,\
                                      (pa+stamp_size)/180.*np.pi]],
                        movement=movement,
                        numbasis = numbasis, 
                        maxnumbasis=maxnumbasis,
                        numthreads=numthreads,
                        spectrum=spectra_template,
                        save_klipped=True, highpass=True,
                        outputdir="demo_output/")

    fake_spect, fakefm = es.invert_spect_fmodel(tmp_dataset.fmout, tmp_dataset, 
                          method="leastsq", units="scaled", scaling_factor=2.0)


Comparing the klipped data to the FM
--------------------------------------------
You may want to look at how well your forward model represents the klipped 
data, measure residual error, etc. All the information you need is in the
output of invert_spect_fmodel: the spectrum and FM matrix. 

Recall the klipped data is in fmout::

    klipped_data = tmp_dataset.fmout[:,:,-1, :]
    klipped_coadd = np.zeros((num_k_klip, nl, stamp_size*stamp_size))
    for ll in range(N_cubes):
        klipped_coadd = klipped_coadd + klipped_data[0, ll*nl:(ll+1)*nl, :]
    # turn it back into a 2D arrat at each wavelength, k_klip
    klipped_coadd.shape = [nl, int(stamp_size), int(stamp_size)]
    # summed over each wavelength channel, but you can view them individually
    plt.imshow(klipped_coadd.sum(axis=0), interpolation="nearest")
    plt.colorbar()

Plot the forward model by taking the dot product with the extracted spectrum::

    k=0 # choose which numbasis
    fm_image_k = np.dot(fakefm[k,:,:], fake_spect[k].transpose())
    # reshape the image back to 2D
    fm_image_k = fm_image_k.reshape(nl, stamp_size, stamp_size)
    # summed over each wavelength channel
    plt.imshow(fm_image_k.sum(axis=0), interpolation="nearest")
    plt.colorbar()


Calculating Errobars
--------------------
One may want to calculate errorbars by injecting signals at an annulus of 
same separation as the real signal and measuring the spread of the recovered
spectra (loop through the procedure above)::

    def recover_fake(files, position, fake_flux):
        # We will need to create a new dataset each time.
        
        # PSF model template for each cube observation, copies of the PSF model:
        inputpsfs = np.tile(calibrated_PSF_model, (N_cubes, 1, 1))
        bulk_contrast = 1e-2
        fake_psf = inputpsfs*fake_flux[0,None,None]
        pa = planet_pa+180

        tmp_dataset = GPI.GPIData(files, highpass=False)
        fakes.inject_planet(tmp_dataset.input, tmp_dataset.centers, fake_psf,\
                                        tmp_dataset.wcs, planet_sep, pa)

        fm_class = es.ExtractSpec(tmp_dataset.input.shape,
                                   numbasis,
                                   planet_sep,
                                   pa,
                                   calibrated_PSF_model,
                                   np.unique(dataset.wvs),
                                   stamp_size = stamp_size)

        fm.klip_dataset(tmp_dataset, fm_class,
                            fileprefix="fakespect",
                            annuli=[[planet_sep-stamp_size,planet_sep+stamp_size]],
                            subsections=[[(pa-stamp_size)/180.*np.pi,\
                                          (pa+stamp_size)/180.*np.pi]],
                            movement=movement,
                            numbasis = numbasis, 
                            maxnumbasis=maxnumbasis,
                            numthreads=numthreads,
                            spectrum=spectra_template,
                            save_klipped=True, highpass=True,
                            outputdir="demo_output/")
        fake_spect, fakefm = es.invert_spect_fmodel(tmp_dataset.fmout, 
                                               tmp_dataset, method="leastsq",
                                               units="scaled", scaling_factor=2.0)
        del tmp_dataset
        return fake_spect

    # This could take a long time to run
    # Define a set of PAs to put in fake sources
    npas = 11
    pas = (np.linspace(planet_pa, planet_pa+360, num=npas+2)%360)[1:-1]

    # For numbasis "k"
    # repeat the spectrum over each cube in the dataset
    input_spect = np.tile(exspect[k,:], N_cubes)[0,:]
    fake_spectra = np.zeros((npas, nl))
    for p, pa in enumerate(pas):
        fake_spectra[p,:] = recover_fake(files, (planet_sep, pa), input_spect)


Other details, like the forward model or klipped data for the injected signal could be useful.


If the real companion signal is too bright, the forward model may fail to capture all the flux
It could be helpful to look at whether the recovered spectra for the simulated signal are 
evenly distributed around the simulated spectrum or if they are systematically lower flux::

    offset[ii] = estim_spec[ii] - np.median(fake_spectra, axis=0)

