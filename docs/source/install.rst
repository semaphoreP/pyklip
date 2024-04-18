.. _install-label:

Installation
==============

Quick Install
-------------
`pyKLIP` is pip-installable::

    $ pip install pyklip

However, if you need to modify the source code, we recommend you do the installation from source code below instead.
You can overwrite a pip-install using the install from source code instructions below, so you can always do the pip install
first, and reinstall from source code later when you need to. 

Note that the pip-installable pyklip does not install all dependencies for all the various modules, just the core functionality.
This should emcompass the large majority of use cases. 
However, there is a chance that you need to install additional packages for specific modules. We are working on fixing this, so that
everything is installed by default, but avoiding installing too many dependencies. 

Install from Source Code
------------------------

Dependencies
^^^^^^^^^^^^
Before you install pyKLIP, you will need to install the following packages, which are useful for most astronomical
data analysis situations anyways. We recommend you use python 3 for pyKLIP, but it can be adapted to python 2.7 as well.

* numpy
* scipy
* astropy
* Optional: matplotlib, mkl-service

For the optional packages, matplotlib is useful to actually plot the images. For mkl-service, pyKLIP autmoatically
toggles off MKL parallelism during parallelized KLIP if the mkl-service package is installed. Otherwise, you
will need to toggle them off yourself for optimal performance. See notes on parallelized performance below.

For :ref:`bka-label` specifically, you'll also want to install the following packages:

* emcee
* corner

As pyKLIP is computationally expensive, we recommend a powerful computer to optimize the computation. As direct imaging
data comes in many different forms, we cannot say
right here what the hardware requirements are for your data reduction needs. For data from the Gemini Planet Imager
(GPI), a computer with 20+ GB of memory is optimal for an 1 hour sequence taken with the integral field spectrograph and
reduced using ADI+SDI. For broadband polarimetry data from GPI, any laptop can reduce the data.

Install
^^^^^^^

Due to the continually developing nature of pyKLIP, we recommend you use the current version of the code on
`Bitbucket <https://bitbucket.org/pyKLIP/pyklip>`_ and keep it updated.
To install the most up to date developer version, clone this repository if you haven't already::

    $ git clone https://bitbucket.org/pyKLIP/pyklip.git

This clones the repoistory using HTTPS authentication. Once the repository is cloned onto your computer, ``cd`` into it and run the setup file::

    $ python setup.py develop

If you use multiple versions of python, you will need to run ``setup.py`` with each version of python
(this should not apply to most people).

Note on parallelized performance
--------------------------------


Due to the fact that numpy compiled with BLAS and MKL also parallelizes linear algebra routines across multiple cores,
performance can actually sharply decrease when multiprocessing and BLAS/MKL both try to parallelize the KLIP math.
If you are noticing your load averages greatly exceeding the number of threads/CPUs,
try disabling the BLAS/MKL optimization when running pyKLIP.

To disable OpenBLAS, just set the following environment variable before running pyKLIP::

    $ export OPENBLAS_NUM_THREADS=1

`A recent update to anaconda <https://www.continuum.io/blog/developer-blog/anaconda-25-release-now-mkl-optimizations>`_
included some MKL optimizations which may cause load averages to greatly exceed the number of threads specified in pyKLIP.
As with the OpenBLAS optimizations, this can be avoided by setting the maximum number of threads the MKL-enabled processes can use::

    $ export MKL_NUM_THREADS=1

As these optimizations may be useful for other python tasks, you may only want MKL_NUM_THREADS=1 only when pyKLIP is called,
rather than on a system-wide level. By defaulf in ``parallelized.py``, if ``mkl-service`` is installed, the original
maximum number of threads for MKL is saved, and restored to its original value after pyKLIP has finished. You can also
modify the number of threads MKL uses on a per-code basis by running the following piece of code (assuming ``mkl-service`` is installed)::

    import mkl
    mkl.set_num_threads(1)


Parallelization on windows
--------------------------------

On windows, you might run into the following error::

    RuntimeError:
            Attempt to start a new process before the current process
            has finished its bootstrapping phase.
            This probably means that you are on Windows and you have
            forgotten to use the proper idiom in the main module:
                if __name__ == '__main__':
                    freeze_support()
                    ...
            The "freeze_support()" line can be omitted if the program
            is not going to be frozen to produce a Windows executable.

If so, simply encapsulate the main code of your script within the following if statement ``if __name__ == '__main__':`` as shown below.
Note that definitions of new functions or include statements can be placed before this statement. Example:

.. code-block:: python

    import glob
    import pyklip.instruments.GPI as GPI
    import pyklip.parallelized as parallelized

    if __name__ == '__main__':
        filelist = glob.glob("path/to/dataset/*.fits")
        dataset = GPI.GPIData(filelist, highpass=True)

        parallelized.klip_dataset(dataset, outputdir="path/to/save/dir/", fileprefix="myobject",
                                  annuli=9, subsections=4, movement=1, numbasis=[1,20,50,100],
                                  calibrate_flux=True, mode="ADI+SDI")
