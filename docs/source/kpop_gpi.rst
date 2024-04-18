.. _kpop-label:


Planet detection
=====================================================
Speckle subtraction algorithms like KLIP or LOCI are not planet detection algorithms. An additional step needs to be performed to compute a SNR for example.
The SNR can be computed from an aperture photometry, which is then divided by noise standard deviation. The standard deviation is often calculated in an annulus at the same separation as the pixel.
This approach is equivalent to a cross correlation of the image with an aperture.
In this tutorial, we present simple functions to compute a cross-correlation for broadband images, a simple matched filter for spectral cubes, a SNR map, and a function to quickly identify the brightest blobs in the image.

.. note::
    The different terminology between cross correlation and matched filter is little arbitrary here since the cross correlation is a kind of matched filter.
    Here, we say matched filter when a division by the local variance is used.


Please find an example ipython notebook (``pyklip/examples/kpop_tutorial.ipynb``) using beta Pictoris test data available in the test directory of pyklip.

Attribution
-----------
If you use this feature, please cite:

 * `Ruffio, J.-B., Macintosh, B., Wang, J. J., et al. 2017, ApJ, 842, 14. <https://ui.adsabs.harvard.edu/abs/2017ApJ...842...14R/abstract>`_

Cross-correlation
-----------------

The cross correlation is the simplest step to perform before computing a SNR map.
``calculate_cc`` calculate the correlation of an image with a kernel, which represents the shape of the planet PSF.
It ensures that the image isn't shifted when using even dimensions with ``scipy.signal.correlate2d``.
A spectrum can also be given to perform a weighted mean if the input image is a cube.

.. code-block:: python

        import astropy.io.fits as pyfits
        filename = "path/to/image/image.fits"
        hdulist = pyfits.open(filename)
        image = hdulist[1].data
        hdulist.close()

One can use different kernels. We provide two simple kernels: aperture (ie, hat) or 2d gaussian.

.. code-block:: python

        from pyklip.kpp.utils.mathfunc import *
        x_grid,y_grid= np.meshgrid(np.arange(-10,10),np.arange(-10,10))
        kernel_hat = hat(x_grid,y_grid, radius=3)
        kernel_gauss = gauss2d(x_grid,y_grid, amplitude = 1.0, xo = 0.0, yo = 0.0, sigma_x = 1.0, sigma_y = 1.0)

The cross correlated image is then given by:

.. code-block:: python

        from pyklip.kpp.metrics.crossCorr import calculate_cc
        image_cc = calculate_cc(image, kernel_gauss,spectrum = None, nans2zero=True)

The next step would to calculate the SNR map of ``image_cc``; see section below.

SNR map
-----------------

There are two routines to compute a SNR map.
The fast version ``get_image_stat_map`` computes the standard deviation in concentric annuli.
The center of the image is defined by ``center=[cen_x,cen_y]``.
Two consecutive annuli radii are separated by ``r_step`` and their width is ``Dr``.
A caveat of this routine is that the standard deviation calculation will be biased by the presence of real point sources.

.. code-block:: python

        center = []#[cen_x,cen_y]

        from pyklip.kpp.stat.stat_utils import get_image_stat_map
        SNR_map = get_image_stat_map(image_cc,
                                   centroid = center,
                                   r_step=2,
                                   Dr = 2,
                                   type = "SNR")

A slower version of the routine will perform on similar operation for each pixel in the image.
It will mask a region of radius ``mask_radius``, and compute the standard deviation in an annulus of width ``Dr`` with the same separation as the current pixel.

.. code-block:: python

        center = []#[cen_x,cen_y]

        from pyklip.kpp.stat.statPerPix_utils import get_image_stat_map_perPixMasking
        SNR_map = get_image_stat_map_perPixMasking(image_cc,
                                                   centroid = center,
                                                   mask_radius=5,
                                                   Dr = 2,
                                                   type = "SNR")

Simple matched filter
-----------------

A more optimal way to detect a planet is to divide pixel values by their variance.
If the data is a spectral cube, we can also a template spectrum  of the planet to improve our sensitivity.
``run_matchedfilter`` performs a matched filter using a 3D model of the planet including the planet PSF and a model of the spectrum of the planet ``planet_sp``.
We illustrate the example with a simple 2D gaussian PSF and a flat spectrum.
The function also estimates the local variance, which is used to normalize the matched filter.

.. code-block:: python

    import astropy.io.fits as pyfits
    filename = "path/to/spectral/cube/cube.fits"
    hdulist = pyfits.open(filename)
    cube = hdulist[1].data
    nl,ny,nx = cube.shape
    hdulist.close()

    # Definition of the planet spectrum
    planet_sp = np.ones(nz)

    # Definition of the PSF
    from pyklip.kpp.utils.mathfunc import *
    x_grid,y_grid= np.meshgrid(np.arange(-10,10),np.arange(-10,10))
    PSF = gauss2d(x_grid,y_grid, amplitude = 1.0, xo = 0.0, yo = 0.0, sigma_x = 1.0, sigma_y = 1.0)
    PSF = np.tile(PSF,(nl,1,1))*planet_sp[:,None,None]

    from pyklip.kpp.metrics.matchedfilter import run_matchedfilter
    mf_map,cc_map,flux_map = run_matchedfilter(cube, PSF,N_threads=None,maskedge=True)


Point-source detection
-----------------

The function ``point_source_detection`` identifies the brightest point sources in an SNR map and returns a table including their SNR and location.
The algorithm is iterative. A disk of radius ``mask_radius`` is masked around the brightest candidate at each iteration.

The table includes the following columns described below:
``["index","value","PA","Sep (pix)","Sep (as)","x","y","row","col"]``

* 1/ index of the candidate
* 2/ Value of the maximum
* 3/ Position angle in degree from North in [0,360]
* 4/ Separation in pixel
* 5/ Separation in arcsec
* 6/ x position in pixel
* 7/ y position in pixel
* 8/ row index
* 9/ column index

.. code-block:: python

    import csv
    from pyklip.kpp.detection.detection import point_source_detection

    detec_threshold = 3 # lower SNR to consider
    pix2as = 1 # platescale (pixel to arcsecond)
    mask_radius = 15 # Size of the mask to be applied at each iteration
    maskout_edge = 10 # Size of the mask to be applied at the edge of the field of view. Works even if the outskirt is full of nans.

    candidates_table = point_source_detection(SNR_map, center,detec_threshold,pix2as=pix2as,
                                             mask_radius = mask_radius,maskout_edge=maskout_edge,IWA=None, OWA=None)

The table can optionally be saved on disk:

.. code-block:: python

    savedetections = os.path.join(outputDir,"detections.csv")
    with open(savedetections, 'w+') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';')
        csvwriter.writerows([["index","value","PA","Sep (pix)","Sep (as)","x","y","row","col"]])
        csvwriter.writerows(candidates_table)

