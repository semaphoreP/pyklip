"""
Author: Miles Lucas <mdlucas@hawaii.edu>

Implementation of interface adapter for VAMPIRES data processed with the VAMPIRES DPP (https://github.com/scexao-org/vampires_dpp)
"""
import pyklip.instruments.Instrument.Data as Data

class VAMPIRES(Data):
    """
    VAMPIRES

    Note
    ----
    Data is expected to be processed from the VAMPIRES DPP
    """
    def __init__(self, args):
        super().__init__()

    ## TO IMPLEMENT
    @property
    def input(self):
        ...

    @property
    def output(self):
        ...

    @property
    def centers(self):
        ...

    @property
    def filenames(self):
        ...

    @property
    def filenums(self):
        ...

    @property
    def wvs(self):
        ...

    @property
    def PAs(self):
        ...

    @property
    def flipx(self):
        # VAMPIRES data has East CCW from North
        return False

    @property
    def wcs(self):
        ...

    @property
    def IWA(self):
        ...

    @property
    def OWA(self):
        ...

    def savedata(self, filepath, data, klipparams=None, filetype="", zaxis=None, more_keywords=None):
        ...

    def readdata(self, filepaths):
        ...

    def calibrate_output(self, img, spectral=False):
        ...





