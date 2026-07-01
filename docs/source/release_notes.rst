.. _release-notes:

Release Notes
==============

Version 2.10.1
 * Fix deprecation of manual setting of `numpy.array.shape`. Now `numpy.reshape()` is always used.
 * FitPSF now raises an error if trying to fit too close to the edge of the image
 * Data class has `pad_input` and `pad_output` functions to pad the sides with NaNs

Version 2.10
 * Add file-level parallelization to klip_dataset via new numchunks and min_chunk_size parameters. This provides a third axis of parallelism (alongside wavelength and sector) by processing KLIP by image chunks. (Jason Wang)
 * Fix numpy v2 TypeError in DiskFM when saving the KL basis, based on issue from Eckhart Spalding (Jason Wang)
 * Fix fm.py issue where ADI frames at same parang were being used when mode="RDI" only (Jason Wang)

Version 2.9.1
 * Fix bug with ADI not using frames at the same wavelength for numpy ~2.2.6 (Kayli Glidic, Jason Wang)
 * Updated DiskFM docs to emphasize it only works with one KL mode cutoff. 
 * Fix link in docs to example GPI datacubes

Version 2.9
 * Fix several deprecation warnings (depedencies that will deprecate API in the future). 
 * BREAKING (minor): `FitPSF.sampler.chain` is no longer accessible due to changes in emcee v3. Use `FitPSF.mcmc_chain` instead. (Jason Wang).
   
    * The :ref:`bka-label` tutorial has been updated to reflect this when inspecting the MCMC chains
 
 * Several modules have internal changes to fix deprecation warnings with no change to functionality (Jason Wang)
 
    * `pyklip.instruments.utils.nair`
    * `pyklip.instruments.GPI`: just the code for recalc_centers
    * `pyklip.fitpsf`
    * `pyklip.instrument.utils.radonCenter`

 * Switched packaging from setup.py to pyproject.toml since setup.py is being deprecated. 

Version 2.8.4
 * Fix some star-center book-keeping edge-cases in JWST interface (Giovanni Strampelli)

Version 2.8.3
 * Update some code to remove deprecatipn warnings (William Balmer)
 * Replace depreicated scipy.interpolate.interp2d with RectBivariateSpline in searchRadon (Jason Wang)

Version 2.8.2
 * JWST interface: can optionally specify the keywords used to find the iamge center (Aarynn Carter)
 * Check for non-printable characters when writing klip params to headers (Aarynn Carter)
 * Error check that RDI centers is a 2-element object

Version 2.8.1
 * Add missing tqdm to dependencies

Version 2.8
 * Updated JWST Interface to work with new spaceKLIP
 * SNR map has additional azimuthal masking option

Version 2.7.1
 * Hotfix to address scipy deprecation of keyword in `eigh()` functio (Jason Wang, Jens Kammerer)
 * PSF library is consistently high pass filtered with dataset (Max Millar-Blanchaer)

Version 2.7
 * Added NMF data imputation functionality and bugfixes in NMF implementation (Bin Ren)
 * Added detailed attribution guidance in docs (Jason Wang)
 * Add ability to skip derotation step after KLIP. See `skip_derot` flag in ``parallelized.klip_dataset`` (Jason Wang)
 * Workflow updates to the doc and testing framework (Jason Wang)
 * Replace deprecated numpy datatypes and other deprecated syntax (Jason Wang)
 * pip install now uses requirements.txt for dependencies (Jason Wang)

Version 2.6
 * JWST module to support both NIRCAM and MIRI coronagraphy. See SpaceKLIP for pipeline capabilities. (Aarynn Carter, Jens Kammerer)
 * New CHARIS instrument tutorial (Minghan Chen)
 * Fix P1640 photutils import issue (Jason Wang)

Version 2.5
 * Add support for RDI for FM classes FMPlanetPSF and MatchedFilter (FMMF) (Jason Wang)
 * Improved error checking for RDI PSF Library before running KLIP (Jason Wang)
 * Update MagAO/VisAO astrometric calibration (William Balmer)
 * Merged ``get_pickles_model_spectrum()`` into ``get_star_spectrum()`` in spectral_management (Minghan Chen)
 * Improvements and bug fixes in CHARIS instrument module (Minghan Chen)
 * Fixed bug where ``parallelized.klip_dataset`` would crash due to insufficient PSFs introduced in Version 2.2 (Jason Wang & Kate Follette)

Version 2.4.1
 * Use pyKLIP version number rather than git commit to track versioning in headers (Jason Wang)

Version 2.4
 * Forward modeling can handle time dependent PSFs now (Jason Wang)
 * Added STIS.py interface and demo notebook (Robert Thompson)
 * Removed an extra 2x scaling in ``klip.nan_gaussian_filter()`` (Jason Wang)
 * Fixed RDI bug where the reference library only has 1 image (Aarynn Carter)
 * Fixed bug in background subtraction in ``GPIData.generate_psf_cube()`` (JB Ruffio)

Version 2.3
 * GPI interface improvements: coronagrpahic throughput, updated astrometric calibration, edge cases (Jason Wang, Rob De Rosa)
 * GPI interface: Removed wind butterfly PCA subtraction has it was not effective (JB Ruffio)
 * For PFS library, fixed diagonal elements of correlation matrix (JB Ruffio)
 * Improvements to DiskFM implementation and python > 3.7 compatability (Johan Mazoyer)
 * Fixed bug where pyKLIP crashes if you only have one science frame (Aarynn Carter)
 * Added warning for debug mode, and supressing print statements if not in verbose mode (Jea Adams)
 * Reorganized navigation bar for docs (Jason Wang)

Version 2.2
 * Field dependent throughput to account for changes in the off-axis PSF due to e.g., coronagraphic throughput (Jea Adams)
 * Added `verbose` flag that can be used to turn off print statements within pyklip (Jea Adams)
 * Various bug fixes (Jason Wang, Johan Mazoyer)
 * Added for explanatory material to docs so that they are more accessible (Jea Adams)

Version 2.1
 * RDI support in forward modeling framework (currently works for DiskFM, support for other FM modules coming) (Johan Mazoyer)
 * GenericData is more feature rich (better saving, automatic wcs generation) (Jason Wang)
 * Minor bug fixes and documentation updates

Version 2.0.1
 * Update Python 3 version to Python 3.6

Version 2.0
 * Forgot to update for a long while. Lots of new changes. A few key summaries below.
 * Forward modeling for planet detection, astrometry, photometry, spectral extraction, and disk forward modeling
 * Support for Keck/NIRC2, Keck/OSIRIS, Subaru/CHARIS, VLT/SPHERE, MagAO/VisAO, and a generic instrument interface for all else
 * Alternative algorithms to KLIP: emperically weighted PCA, non-negative matrix factorization
 * RDI library support
 * Automated tests to ensure correctness of main features
 * Now released on PyPI/pip

Version 1.1
 * Updated installation to be much easier
 * Reorganized repo structure to match standard python repos
 * Improvements to automatic planet detection code

Version 1.0
 * Initial Release
 * Fully-functional KLIP implementation for ADI and SDI
 * Interface for GPI data in both spectral and polarimetry mode
 * Utility functions like fake injection and contrast calculation
