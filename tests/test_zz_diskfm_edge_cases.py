"""test_zz_diskfm_edge_cases.py

Regression tests for DiskFM with degenerate disk models and data: all-zero and all-NaN
models, NaN regions in the model, and a data frame that is entirely NaN. Both the
save path (fm.klip_dataset) and the load path (update_disk + fm_parallelized) are checked.
"""
import glob
import os

import numpy as np
import pytest

import pyklip.instruments.GPI as GPI
import pyklip.fm as fm
from pyklip.fmlib.diskfm import DiskFM

from test_zz_diskfm import make_phony_disk

os.environ["OMP_NUM_THREADS"] = "1"

TESTDIR = os.path.dirname(os.path.abspath(__file__)) + os.path.sep
CENTER = [140, 140]
NUMBASIS = [3]


def load_dataset():
    filelist = sorted(glob.glob(TESTDIR + os.path.join("data", "S20131210*distorcorr.fits")))
    dataset = GPI.GPIData(filelist, quiet=True)
    dataset.spectral_collapse(collapse_channels=1, align_frames=True)
    return dataset


def run_diskfm(dataset, model, outdir):
    """Save the KL basis with fm.klip_dataset, then reload it and forward model again"""
    os.makedirs(str(outdir), exist_ok=True)
    basis_file = os.path.join(str(outdir), "edge_KLbasis.h5")
    diskobj = DiskFM(dataset.input.shape, NUMBASIS, dataset, model, basis_filename=basis_file,
                     save_basis=True, aligned_center=CENTER)
    fm.klip_dataset(dataset, diskobj, numbasis=NUMBASIS, maxnumbasis=100, annuli=1, subsections=1,
                    mode="ADI", outputdir=str(outdir), fileprefix="edge", aligned_center=CENTER,
                    mute_progression=True, highpass=False, minrot=8, calibrate_flux=False)
    fm_save = np.array(dataset.fmout)

    diskobj = DiskFM(None, None, None, model, basis_filename=basis_file, load_from_basis=True)
    diskobj.update_disk(model)
    fm_load = diskobj.fm_parallelized()

    # the reloaded basis reproduces the forward model of klip_dataset (mean over frames)
    assert np.all(np.isfinite(fm_load))
    assert close(fm_load, np.nanmean(fm_save, axis=1))
    return fm_save, fm_load


def close(a, b, rtol=1e-10):
    """a and b agree to rtol relative to the peak of b (pixel-wise rtol fails on pixels near 0)"""
    return np.allclose(a, b, rtol=0, atol=rtol * max(np.nanmax(np.abs(b)), 1e-300), equal_nan=True)


@pytest.mark.parametrize("fill", [0., np.nan])
def test_diskfm_empty_model(fill, tmp_path):
    """A model with no flux (all zeros, or all NaNs which are set to 0) gives a zero forward model"""
    dataset = load_dataset()
    model = np.full(dataset.input.shape[1:], fill)

    fm_save, fm_load = run_diskfm(dataset, model, tmp_path)

    assert np.all(fm_save == 0)
    assert np.all(fm_load == 0)


def test_diskfm_model_nan_region(tmp_path):
    """NaNs in the model are treated as zeros"""
    model = make_phony_disk(281)
    model_nan = model.copy()
    model_nan[200:230, 100:180] = np.nan
    model_zero = np.nan_to_num(model_nan)

    fm_save_nan, fm_load_nan = run_diskfm(load_dataset(), model_nan, tmp_path / "nan")
    fm_save_zero, fm_load_zero = run_diskfm(load_dataset(), model_zero, tmp_path / "zero")

    assert np.all(np.isfinite(fm_save_nan))
    assert np.nanmax(np.abs(fm_load_nan)) > 0
    assert close(fm_save_nan, fm_save_zero)
    assert close(fm_load_nan, fm_load_zero)


def test_diskfm_data_frame_all_nan(tmp_path):
    """An input frame that is entirely NaN is skipped without breaking the forward model"""
    dataset = load_dataset()
    dataset.input[1] = np.nan
    model = make_phony_disk(281)

    fm_save, fm_load = run_diskfm(dataset, model, tmp_path)

    for fmout in (fm_save, fm_load):
        assert np.all(np.isfinite(fmout))
        assert np.max(np.abs(fmout)) > 0
