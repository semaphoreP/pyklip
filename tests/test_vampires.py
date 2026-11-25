import os
import sys
import glob

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

if sys.version_info < (3, 3):
    from mock import patch
else:
    from unittest.mock import patch

from pyklip import klip
from pyklip.instruments import VAMPIRES

testdir = os.path.dirname(os.path.abspath(__file__)) + os.path.sep
datadir = os.path.join(testdir, "data", "vampires")
outputdir = os.path.join(datadir, "output")

PXSCALE_DEG = 5.91 / 3.6e6  # PXSCALE header value (mas/px) in deg/px
NORTH_UP_CD = [[-1, 0], [0, 1]]  # CD / pxscale for North-up, East-left


def load_vampires_test_data():
    filelist = sorted(glob.glob(os.path.join(datadir, "*HD1160*.fits")))
    assert len(filelist) == 3
    return filelist


def _patch_header(**cards):
    """
    Patch _vampires_process_file so the primary header gets `cards` applied
    (a value of None deletes the card), without touching the files on disk.
    """
    orig = VAMPIRES._vampires_process_file

    def patched(filepath):
        out = orig(filepath)
        for key, value in cards.items():
            if value is None:
                out["prim_hdr"].remove(key, ignore_missing=True)
            else:
                out["prim_hdr"][key] = value
        return out

    return patch("pyklip.instruments.VAMPIRES._vampires_process_file", patched)


#### Loading ####


def test_process_vampires_data():
    for filepath in load_vampires_test_data():
        data = VAMPIRES._vampires_process_file(filepath)

        for key in ("cube", "centers", "PAs", "wvs", "wcs", "prim_hdr"):
            assert key in data

        assert data["cube"].shape == (4, 536, 536)
        np.testing.assert_allclose(data["centers"], np.tile([267.5, 267.5], (4, 1)))
        assert data["PAs"].shape == (4,)
        np.testing.assert_allclose(data["PAs"], data["prim_hdr"]["DEROTANG"])
        np.testing.assert_allclose(data["wvs"], (0.614, 0.670, 0.721, 0.761), atol=5e-4)
        assert len(data["wcs"]) == 4


def test_vampiresdata_construction():
    filelist = load_vampires_test_data()
    dataset = VAMPIRES.VAMPIRESData(filelist)

    assert dataset.input.shape == (12, 536, 536)
    assert dataset.wvs.shape == (12,)
    assert dataset.centers.shape == (12, 2)
    assert dataset.PAs.shape == (12,)
    assert dataset.filenames.shape == (12,)
    assert dataset.filenums.shape == (12,)
    assert dataset.wcs.shape == (12,)
    np.testing.assert_array_equal(dataset.filenums, np.repeat(range(3), 4))
    assert dataset.flipx is False

    # loading after construction is equivalent
    dataset2 = VAMPIRES.VAMPIRESData()
    dataset2.readdata(filelist)

    np.testing.assert_array_equal(dataset.input, dataset2.input)
    np.testing.assert_array_equal(dataset.wvs, dataset2.wvs)
    np.testing.assert_array_equal(dataset.centers, dataset2.centers)
    np.testing.assert_array_equal(dataset.PAs, dataset2.PAs)
    np.testing.assert_array_equal(dataset.filenames, dataset2.filenames)
    np.testing.assert_array_equal(dataset.filenums, dataset2.filenums)


def test_readdata_accepts_single_string_path():
    filepath = load_vampires_test_data()[0]
    dataset = VAMPIRES.VAMPIRESData(filepath)
    assert dataset.input.shape == (4, 536, 536)
    assert len(dataset.prim_hdrs) == 1


def test_filenames_are_paths():
    filelist = load_vampires_test_data()
    dataset = VAMPIRES.VAMPIRESData(filelist)
    np.testing.assert_array_equal(dataset.filenames, np.repeat(filelist, 4))


#### IWA / OWA ####


def test_iwa_from_focal_plane_mask():
    dataset = VAMPIRES.VAMPIRESData(load_vampires_test_data())
    # test data uses CLC-3 (59 mas) at 5.91 mas/px
    assert dataset.prim_hdrs[0]["U_FLDSTP"] == "CLC-3"
    assert dataset.IWA == pytest.approx(59 / 5.91)
    assert dataset.OWA == pytest.approx(536 / 2)


def test_iwa_unknown_mask_warns():
    with _patch_header(U_FLDSTP="NOT-A-MASK"):
        with pytest.warns(UserWarning, match="NOT-A-MASK"):
            dataset = VAMPIRES.VAMPIRESData(load_vampires_test_data())
    assert dataset.IWA == 0


def test_iwa_missing_mask_keyword():
    with _patch_header(U_FLDSTP=None):
        dataset = VAMPIRES.VAMPIRESData(load_vampires_test_data())
    assert dataset.IWA == 0


def test_user_iwa_owa_not_overridden():
    dataset = VAMPIRES.VAMPIRESData(load_vampires_test_data(), IWA=3.0, OWA=100.0)
    assert dataset.IWA == 3.0
    assert dataset.OWA == 100.0


#### WCS ####


def test_pa_derotation_gives_north_up_wcs():
    for filepath in load_vampires_test_data():
        data = VAMPIRES._vampires_process_file(filepath)
        wcs = data["wcs"][0].deepcopy()
        klip._rotate_wcs_hdr(wcs, data["PAs"][0], flipx=False)
        np.testing.assert_allclose(wcs.wcs.cd / PXSCALE_DEG, NORTH_UP_CD, atol=1e-4)


def test_wcs_reference_is_star():
    for filepath in load_vampires_test_data():
        data = VAMPIRES._vampires_process_file(filepath)
        wcs = data["wcs"][0]
        cy, cx = data["centers"][0]
        ra, dec = wcs.wcs_pix2world([[cx, cy]], 0)[0]
        np.testing.assert_allclose([ra, dec], wcs.wcs.crval, atol=1e-7)  # ~0.4 mas


#### savedata ####


def test_savedata_header():
    os.makedirs(outputdir, exist_ok=True)
    dataset = VAMPIRES.VAMPIRESData(load_vampires_test_data())
    orig_hdr = dataset.prim_hdrs[0].copy()
    dataset.output_wcs = np.array([w.deepcopy() for w in dataset.wcs])
    for w, pa in zip(dataset.output_wcs, dataset.PAs):
        klip._rotate_wcs_hdr(w, pa, flipx=dataset.flipx)

    outpath = os.path.join(outputdir, "test_savedata.fits")
    dataset.savedata(
        outpath,
        np.zeros((3, 536, 536)),
        klipparams="test params",
        filetype="KL Mode Cube",
        zaxis=[1, 2, 3],
        fakePlparams="fake params",
    )

    hdr = fits.getheader(outpath)
    m = WCS(hdr, naxis=2).pixel_scale_matrix
    np.testing.assert_allclose(m / PXSCALE_DEG, NORTH_UP_CD, atol=1e-4)
    assert hdr["DRPNFILE"] == 3
    assert hdr["FILETYPE"] == "KL Mode Cube"
    assert hdr["CTYPE3"] == "KLMODES"
    assert [hdr["KLMODE{}".format(i)] for i in range(3)] == [1, 2, 3]
    assert hdr["PSFSUB"] == "pyKLIP"
    assert hdr["PSFPARAM"] == "test params"
    assert hdr["FAKPLPAR"] == "fake params"
    # saving must not modify the input header
    assert dataset.prim_hdrs[0] == orig_hdr


#### flux calibration ####


def test_calibrate_output_warns_and_passes_through():
    dataset = VAMPIRES.VAMPIRESData()
    img = np.random.default_rng(0).normal(size=(4, 10, 10))
    with pytest.warns(UserWarning, match="vampires-dpp"):
        out = dataset.calibrate_output(img, spectral=True)
    np.testing.assert_array_equal(out, img)


#### KLIP end-to-end ####


@pytest.mark.parametrize("mode", ["ADI", "SDI", "ADI+SDI"])
def test_vampires_klip(mode):
    from pyklip import parallelized

    os.makedirs(outputdir, exist_ok=True)
    dataset = VAMPIRES.VAMPIRESData(load_vampires_test_data())
    numbasis = [1, 2, 3]
    prefix = "test_" + mode.replace("+", "")
    parallelized.klip_dataset(
        dataset=dataset,
        outputdir=outputdir,
        fileprefix=prefix,
        mode=mode,
        annuli=5,
        subsections=4,
        movement=0,
        numbasis=numbasis,
        calibrate_flux=False,
    )

    # check exact filenames (not a glob) so leftovers from other runs don't matter
    klmodes_file = os.path.join(outputdir, "{}-KLmodes-all.fits".format(prefix))
    klcube = fits.getdata(klmodes_file)
    assert klcube.shape == (len(numbasis), 536, 536)
    for k in numbasis:
        speccube_file = os.path.join(outputdir, "{}-KL{}-speccube.fits".format(prefix, k))
        assert fits.getdata(speccube_file).shape == (4, 536, 536)

    # something was actually reduced: finite, non-zero pixels outside the masks
    finite = np.isfinite(klcube)
    assert 0.5 < finite.mean() < 1
    assert np.any(klcube[finite] != 0)
    # output is North-up
    m = WCS(fits.getheader(klmodes_file), naxis=2).pixel_scale_matrix
    np.testing.assert_allclose(m / PXSCALE_DEG, NORTH_UP_CD, atol=1e-4)
