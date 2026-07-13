# pyKLIP #
[![Documentation Status](https://readthedocs.org/projects/pyklip/badge/?version=latest)](http://pyklip.readthedocs.io/en/latest/?badge=latest) [![CI tests](https://github.com/semaphoreP/pyklip/actions/workflows/pytests.yml/badge.svg)](https://github.com/semaphoreP/pyklip/actions/workflows/pytests.yml) [![Coverage Status](https://coveralls.io/repos/github/semaphoreP/pyklip/badge.svg?branch=main)](https://coveralls.io/github/semaphoreP/pyklip?branch=main) [![ASCL Reference](https://img.shields.io/badge/ascl-1506.001-blue.svg?colorB=262255)](http://ascl.net/1506.001) [![PyPI version](https://badge.fury.io/py/pyklip.svg)](https://badge.fury.io/py/pyklip)

**Announcement: pyKLIP has moved to a [Github repo](https://github.com/semaphoreP/pyklip) due to Bitbucket deprecating its public issue tracker.**
    
A python library for PSF subtraction for both exoplanet and disk imaging. It uses a parallelized and optimzied implmentation of [KLIP](http://arxiv.org/abs/1207.4197) that supports ADI, SDI, and RDI with a variety of tunable parameters. For characterization, forward modelling tools include a suite of tools built off [KLIP-FM](http://arxiv.org/abs/1604.06097) for astrometry, spectroscopy, planet detection, and disk modelling. pyKLIP is modular and supports data from the Gemini Planet Imager, P1640, Keck/NIRC2, MagAO/VisAO, and SPHERE, with more on the way.

Want to get started? Check out the [quick GPI KLIP tutorial](http://pyklip.readthedocs.io/en/latest/klip_gpi.html) for the basics of pyKLIP.

Development led by Jason Wang. Contributions made by Jonathan Aguilar, JB Ruffio, Rob de Rosa, Schuyler Wolff, Abhijith Rajan, Zack Briesemeister, Kate Follette, Maxwell Millar-Blanchaer, Alexandra Greenbaum, Simon Ko, Tom Esposito, Elijah Spiro, Pauline Arriaga, Bin Ren, Alan Rainot, Arthur Vigan, Graça Rocha, Jacob Golomb, Johan Mazoyer, Jea Adams, Aarynn Carter, Minghan Chen, Robert Thompson, Jens Kammerer, William Balmer, Giovanni Strampelli, and Laurent Pueyo (see contributors.txt for a details).

If you use pyKLIP in your research, please cite the Astrophysical Source Code Library record of it: [ASCL](http://ascl.net/1506.001) or [ADS](http://adsabs.harvard.edu/abs/2015ascl.soft06001W).

> Wang, J. J., Ruffio, J.-B., De Rosa, R. J., et al. 2015, Astrophysics Source Code Library, ascl:1506.001

For setup instructions, example code, and API details, 
[**read the documentation**](http://pyklip.readthedocs.io/en/latest/) online!