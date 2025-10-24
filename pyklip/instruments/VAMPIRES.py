"""
Author: Miles Lucas <mdlucas@hawaii.edu>

Implementation of interface adapter for VAMPIRES data processed with the VAMPIRES DPP (https://github.com/scexao-org/vampires_dpp)
"""
import warnings

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from pyklip.instruments.Instrument import Data
import pyklip

class VAMPIRESData(Data):
    """
    VAMPIRESData

    Note
    ----
    Data is expected to be processed from the VAMPIRES DPP
    """

    MASK_IWA = {  # mask name: IWA in mas
        "FIELDSTOP": 0,
        "CLC-2": 37,
        "CLC-3": 59,
        "CLC-5": 105,
        "CLC-7": 150,
        "DGVVC": 61,
    }

    def __init__(self, filepaths=None, IWA=None, OWA=None):
        super().__init__()
        # initialize class variables
        self._output = None
        self._input = None
        self._centers = None
        self._filenums = None
        self._filenames = None
        self._PAs = None
        self._wvs = None
        self._wcs = None
        self._IWA = IWA
        self._OWA = OWA
        self._flipx = False
        # self.star_flux = None
        # self.contrast_scaling = None
        # extras
        self.prim_hdrs = None
        if filepaths is not None:
            self.readdata(filepaths)

    ###################################
    ### Required Instance Variances ###
    ###################################
    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, value):
        self._input = value

    @property
    def centers(self):
        return self._centers

    @centers.setter
    def centers(self, value):
        self._centers = value

    @property
    def PAs(self):
        return self._PAs

    @PAs.setter
    def PAs(self, value):
        self._PAs = value

    @property
    def wvs(self):
        return self._wvs

    @wvs.setter
    def wvs(self, value):
        self._wvs = value

    @property
    def wcs(self):
        return self._wcs

    @wcs.setter
    def wcs(self, value):
        self._wcs = value

    @property
    def filenames(self):
        return self._filenames

    @filenames.setter
    def filenames(self, value):
        self._filenames = value

    @property
    def filenums(self):
        return self._filenums

    @filenums.setter
    def filenums(self, value):
        self._filenums = value

    @property
    def IWA(self):
        return self._IWA

    @IWA.setter
    def IWA(self, value):
        self._IWA = value

    @property
    def OWA(self):
        return self._OWA

    @OWA.setter
    def OWA(self, value):
        self._OWA = value

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self._output = value

    def readdata(self, filepaths):
        """
        Method to open and read a list of VAMPIRES data
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        print("Loading {0:d} VAMPIRES data files".format(len(filepaths)))

        data = []
        filenums = []
        filenames = []
        PAs = []
        wvs = []
        prim_hdrs = []
        wcss = []
        centers = []

        for index, filepath in enumerate(filepaths):
            vamp_data = _vampires_process_file(filepath)
            data.append(vamp_data["cube"])
            centers.append(vamp_data["centers"])
            PAs.append(vamp_data["PAs"])
            wvs.append(vamp_data["wvs"])
            wcss.append(vamp_data["wcs"])
            prim_hdrs.append(vamp_data["prim_hdr"])
            filenums.append([index for _ in range(vamp_data["centers"].shape[0])])
            filenames.append([index for _ in range(vamp_data["centers"].shape[0])])

        self.input = np.concatenate(data)
        self.centers = np.concatenate(centers)
        self.PAs = np.concatenate(PAs)
        self.wvs = np.concatenate(wvs)
        self.prim_hdrs = prim_hdrs
        self.wcs = np.concatenate(wcss)
        self.filenames = np.concatenate(filenames)
        self.filenums = np.concatenate(filenums)

        ## If IWA was not set by user, determine it
        if self.IWA is None:
            self.IWA = 0  # default value
            if "U_FLDSTP" in self.prim_hdrs[0]:
                coro_mask = self.prim_hdrs[0]["U_FLDSTP"]
                if coro_mask in self.MASK_IWA:
                    coro_iwa_mas = self.MASK_IWA[coro_mask]
                else:
                    coro_iwa_mas = 0
                    msg = (
                        "Did not recognize focal plane optic '{}', IWA set to 0".format(
                            coro_mask
                        )
                    )
                    warnings.warn(msg, stacklevel=2)
                pxscale = self.prim_hdrs[0]["PXSCALE"]  # mas / px
                self.IWA = coro_iwa_mas / pxscale

        ## If OWA was not set by user, determine it
        if self.OWA is None:
            self.OWA = min(self.input.shape[-2:]) / 2

    def savedata(
        self,
        filepath,
        data,
        klipparams=None,
        filetype="",
        zaxis=None,
        fakePlparams=None,
        more_keywords=None,
    ):
        """
        Save data in a VAMPIRES-like fashion. Data is stored in the first HDU.

        Inputs:
        filepath: path to file to output
        data: 2D or 3D data to save
        klipparams: a string of klip parameters
        filetype: filetype of the object (e.g. "KL Mode Cube", "PSF Subtracted Spectral Cube")
        zaxis: a list of values for the zaxis of the datacub (for KL mode cubes currently)
        fakePlparams: fake planet params
        """
        # need to fix WCS
        header = self.prim_hdrs[0]
        if self.output_wcs is not None:
            header.update(self.output_wcs[0].to_header())
            # Note: if PC matrix is identity the to_header() function doesn't
            # create any header entries for PCi_j. Therefore, manually override those
            header["PC1_1"] = header["PC2_2"] = 1
            header["PC1_2"] = header["PC2_1"] = 0
            
        prim_hdu = fits.PrimaryHDU(data=data, header=header)
        hdulist = fits.HDUList([prim_hdu])

        # save all the files we used in the reduction
        # we'll assume you used all the input files
        # remove duplicates from list
        # print("filenames = " + self._filenames)
        filenames = np.unique(self.filenames)
        nfiles = np.size(filenames)
        hdulist[0].header["DRPNFILE"] = nfiles

        # write out psf subtraction parameters
        # this will probably come to bite me later
        try:
            pyklipver = pyklip.__version__
        except Exception:
            pyklipver = "unknown"
        hdulist[0].header["PSFSUB"] = "pyKLIP"
        hdulist[0].header.add_history(
            "Reduced with pyKLIP using commit {0}".format(pyklipver)
        )

        # store commit number for pyklip
        hdulist[0].header["PYKLIPV"] = pyklipver

        if klipparams is not None:
            hdulist[0].header["PSFPARAM"] = klipparams
            hdulist[0].header.add_history(
                "pyKLIP reduction with parameters {0}".format(klipparams)
            )

        if fakePlparams is not None:
            hdulist[0].header["FAKPLPAR"] = fakePlparams
            hdulist[0].header.add_history(
                "pyKLIP reduction with fake planet injection parameters {0}".format(
                    fakePlparams
                )
            )

        if filetype is not None:
            hdulist[0].header["FILETYPE"] = filetype

        if zaxis is not None:
            # Writing a KL mode Cube
            if "KL Mode" in filetype:
                hdulist[0].header["CTYPE3"] = "KLMODES"
                # write them individually
                for i, klmode in enumerate(zaxis):
                    hdulist[0].header["KLMODE{0}".format(i)] = klmode

        hdulist.writeto(filepath, overwrite=True)
        hdulist.close()

    # def calibrate_output(self, img, spectral=False):
    #     # first off, if we're not in


def _vampires_process_file(filepath):
    with fits.open(filepath) as hdulist:
        cube = np.nan_to_num(hdulist[0].data)
        header = hdulist[0].header

        # get wavelengths from the ancilliary headers
        wavelengths = np.array([hdu.header["WAVEAVE"] / 1e3 for hdu in hdulist[2:]])
        derotang = header["DEROTANG"]
        wcs = _vampires_extract_wcs(hdulist)
        # TODO flux scaling

    center = np.array(cube.shape[-2:]) / 2 - 0.5
    # tile centers for each wavelength
    centers = np.tile(center, (len(wavelengths), 1))
    # as well as the derotation angles
    derotangs = np.repeat(derotang, len(wavelengths))
    return {
        "cube": cube,
        "centers": centers,
        "PAs": derotangs,
        "wvs": wavelengths,
        "wcs": wcs,
        "prim_hdr": header,
    }


def _vampires_extract_wcs(hdulist):
    """
    Read out relevant WCS parameters and destroy extraneous entries
    """
    prim_hdr = hdulist[0].header
    wcs_out = WCS(header=prim_hdr, naxis=2)

    pc = np.array(wcs_out.wcs.pc)
    cdelt = np.array(wcs_out.wcs.cdelt)
    cd = pc * cdelt[None, :]  # multiply each column by CDELT_j

    wcs_out.wcs.cd = cd
    wcs_out.wcs.pc = [[1, 0], [0, 1]]

    return [wcs_out.deepcopy() for _ in hdulist[2:]]
