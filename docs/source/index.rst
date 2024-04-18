.. pyKLIP documentation master file, created by
   sphinx-quickstart on Thu Mar 12 16:05:49 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. image:: pyklip_logo_150.png
   :align: center
   :alt: pyKLIP: om nom nom star
   :height: 150px
   :width: 150px


pyKLIP
==================================


pyKLIP is a python library for direct imaging of exoplanets and disks. It uses an implmentation of
`KLIP <http://arxiv.org/abs/1207.4197>`_ and `KLIP-FM <http://arxiv.org/abs/1604.06097>`_ to perform point spread
function (PSF) subtraction. KLIP is based off of principal component analysis to model and subtract off the stellar PSF
to look for faint exoplanets and disks and are around it.

pyKLIP is open source, BSD-licened, and available at
`this Bitbucket repo <https://bitbucket.org/pyKLIP/pyklip>`_. You can use the issue tracker there to submit issues and
all contributions are welcome!

Features
--------
* Capable of running ADI, SDI, ADI+SDI with spectral templates to optimize the PSF subtraction
* Library of KLIP-FM capabilties including forward-modelling a PSF, detection algorithms, and spectral extraction.
* A Forward Model Matched Filter of KLIP is available for GPI as well as post-processing planet detection algorithms.
* Parallelized with both a quick memory-intensive mode and a slower memory-lite mode
* Modularized to support data from multiple instruments. Currently there are interfaces to
  `P1640 <http://www.amnh.org/our-research/physical-sciences/astrophysics/research/project-1640>`_,
  `GPI <http://planetimager.org/>`_, SPHERE, CHARIS, MagAO/VisAO, and Keck/NIRC2.
* If confused about what a function is doing, read the docstring for it. We have tried our best to document everything
* See :ref:`release-notes` for update notes

Bugs/Feature Requests
------------------------

Please use the `Issue Tracker <https://bitbucket.org/pyKLIP/pyklip/issues?status=new&status=open>`_ on Bitbucket to
submit bugs and new feature requests. Anyone is able to open issues.

Attribution
-----------------------
The development of pyKLIP is led by Jason Wang with contributions made by Jonathan Aguilar, JB Ruffio, Rob de Rosa, 
Schuyler Wolff, Abhijith Rajan, Zack Briesemeister, Kate Follette, Maxwell Millar-Blanchaer, Alexandra Greenbaum, 
Simon Ko, Tom Esposito, Elijah Spiro, Pauline Arriaga, Bin Ren, Alan Rainot, Arthur Vigan, Johan Mazoyer, Graça Rocha, Jacob Golomb, 
Jea Adams, Aarynn Carter, Minghan Chen, Robert Thompson, Jens Kammerer, and Laurent Pueyo. 

General pyKLIP Citation
^^^^^^^^^^^^^^^^^^^^^^^^

If you use pyKLIP, please cite the `Astrophysical Source Code Library record <http://ascl.net/1506.001>`_ of it:

 * `Wang, J. J., Ruffio, J.-B., De Rosa, R. J., et al. 2015, ASCL, ascl:1506.001 <http://adsabs.harvard.edu/abs/2015ascl.soft06001W>`_

Analysis Feature Citations
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you use any of the KLIP forward modeling features, please cite:

 * `Pueyo, L. 2016, ApJ, 824, 117. <https://ui.adsabs.harvard.edu/abs/2016ApJ...824..117P/abstract>`_

If you use the FMMF, mathced filter, SNR map, or other related planet detection modules, please cite:

 * `Ruffio, J.-B., Macintosh, B., Wang, J. J., et al. 2017, ApJ, 842, 14. <https://ui.adsabs.harvard.edu/abs/2017ApJ...842...14R/abstract>`_

If you use the point source forward modelling framework to obtain astrometry or photometry of point sources, please cite:
 
 * `Wang, J. J., Graham, J. R., Pueyo, L., et al. 2016, AJ, 152, 97. <https://ui.adsabs.harvard.edu/abs/2016AJ....152...97W/abstract>`_
 * `Foreman-Mackey, D., Hogg, D. W., Lang, D., et al. 2013, PASP, 125, 306. <https://ui.adsabs.harvard.edu/abs/2013PASP..125..306F/abstract>`_

If you use the point source spectral extraction forward modelling framework, please cite:

 * `Greenbaum, A. Z., Pueyo, L., Ruffio, J.-B., et al. 2018, AJ, 155, 226. <https://ui.adsabs.harvard.edu/abs/2018AJ....155..226G/abstract>`_

If you used the circumstellar disk forward modelling frameowrk, please cite:

 * `Mazoyer, J., Arriaga, P., Hom, J., et al. 2020, Proc. SPIE, 11447, 1144759. <https://ui.adsabs.harvard.edu/abs/2020SPIE11447E..59M/abstract>`_

If you used the planet evidence module for planet detection, please cite:

 * `Golomb, J., Rocha, G., Meshkat, T., et al. 2021, AJ, 162, 304. <https://ui.adsabs.harvard.edu/abs/2021AJ....162..304G/abstract>`_

If you use the NMF algorithm, please cite:

 * `Ren, B., Pueyo, L., Zhu, G. B., et al. 2018, ApJ, 852, 104. <https://ui.adsabs.harvard.edu/abs/2018ApJ...852..104R/abstract>`_

Instrument Interface Citations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you use the JWST interface, please cite:

 * `Kammerer, J., Girard, J., Carter, A. L., et al. 2022, Proc. SPIE, 12180, 121803N. <https://ui.adsabs.harvard.edu/abs/2022SPIE12180E..3NK/abstract>`_
 * `Carter, A. L., Hinkley, S., Kammerer, J., et al. 2023, ApJL, 951, L20. <https://ui.adsabs.harvard.edu/abs/2023ApJ...951L..20C/abstract>`_

If you use the CHARIS interface, please cite:

 * Chen, M., Wang, J. J., Brandt, T. D., et al. submitted. 

If you use the Keck interface, please acknowledge Tom Esposito's development of the interface in your Acknowledgements section.

If you use the MagAO/VisAO interface, please cite:

 * `Follette, K. B., Rameau, J., Dong, R., et al. 2017, AJ, 153, 264. <https://ui.adsabs.harvard.edu/abs/2017AJ....153..264F/abstract>`_


Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Setup

   install
   release_notes
   
.. toctree::
   :maxdepth: 2
   :caption: The Basics

   klip_gpi
   instruments/index
   contrast_curves
   rdi

.. toctree::
   :maxdepth: 2
   :caption: Planet Detection

   kpop_gpi
   fmmf
   planetevidence


.. toctree::
   :maxdepth: 2
   :caption: Characterization

   bka
   fm_spect
   diskfm_gpi

.. toctree::
   :maxdepth: 2
   :caption: API

   developing/index
   pyklip




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

