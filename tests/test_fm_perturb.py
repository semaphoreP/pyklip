"""test_fm_perturb.py

Check that the KL-mode based perturbation (fm.perturb_specIncluded_from_KL)
matches the reference-image based one (fm.perturb_specIncluded), including
sections with NaNs as found in IFS cubes.
"""
import numpy as np
import pytest

import pyklip.fm as fm


def make_section(n_ref=60, n_pix=3000, nan_frac=0.0, badpix_frac=0.0, n_wv=1, seed=0):
    """
    Make a fake KLIP section: correlated reference images, a science image and disk models.
    n_wv > 1 mimics an IFS stack where the references come from several wavelengths.
    """
    rng = np.random.default_rng(seed)
    speckles = rng.normal(size=(8, n_pix))
    wv_scale = np.repeat(np.linspace(1, 1.5, n_wv), int(np.ceil(n_ref / n_wv)))[:n_ref]
    refs = (rng.normal(size=(n_ref, 8)) * wv_scale[:, None]).dot(speckles)
    refs += 0.1 * rng.normal(size=refs.shape)
    sci = rng.normal(size=8).dot(speckles) + 0.1 * rng.normal(size=n_pix)
    models = np.abs(rng.normal(size=(n_ref, n_pix))) * wv_scale[:, None]

    # NaN regions shared by all frames (e.g. outside the IWA/OWA or the IFS field) and random bad pixels
    shared = rng.random(n_pix) < nan_frac
    refs[:, shared] = np.nan
    sci[shared] = np.nan
    refs[rng.random(refs.shape) < badpix_frac] = np.nan
    models[rng.random(models.shape) < badpix_frac] = np.nan
    return sci, refs, models


@pytest.mark.parametrize("nan_frac, badpix_frac, n_wv, numbasis", [
    (0.0, 0.0, 1, 10),
    (0.3, 0.01, 1, 10),
    (0.6, 0.02, 3, 20),
    (0.6, 0.02, 3, 1),
])
def test_perturb_from_KL_matches_refs(nan_frac, badpix_frac, n_wv, numbasis):
    sci, refs, models = make_section(nan_frac=nan_frac, badpix_frac=badpix_frac, n_wv=n_wv)

    _, klmodes, evals, evecs = fm.klip_math(sci, refs, np.array([numbasis]))

    # both functions zero the NaNs of the models in place, so give each its own copy
    delta_KL_refs = fm.perturb_specIncluded(evals, evecs, klmodes, refs, models.copy())
    delta_KL_kl = fm.perturb_specIncluded_from_KL(evals, evecs, klmodes, models.copy())

    assert delta_KL_kl.shape == delta_KL_refs.shape
    assert np.all(np.isfinite(delta_KL_kl))
    assert np.allclose(delta_KL_kl, delta_KL_refs, rtol=1e-10, atol=1e-10 * np.abs(delta_KL_refs).max())


def test_perturb_from_KL_precomputed_evecs_models():
    """DiskFM passes evecs.T . models_ref computed over all frames, with zero weight for non-reference frames"""
    sci, refs, models = make_section(nan_frac=0.3, badpix_frac=0.01, n_wv=2)
    _, klmodes, evals, evecs = fm.klip_math(sci, refs, np.array([10]))

    # all frames of the sequence: the references are a shuffled subset of them
    rng = np.random.default_rng(1)
    n_frames = refs.shape[0] + 15
    ref_indices = rng.permutation(n_frames)[:refs.shape[0]]
    model_frames = np.abs(rng.normal(size=(n_frames, models.shape[1])))
    model_frames[ref_indices] = np.nan_to_num(models)

    evecs_all = np.zeros((n_frames, evecs.shape[1]))
    np.add.at(evecs_all, ref_indices, evecs)
    evecs_models = evecs_all.T.dot(model_frames)

    delta_KL_refs = fm.perturb_specIncluded(evals, evecs, klmodes, refs, models.copy())
    delta_KL_kl = fm.perturb_specIncluded_from_KL(evals, evecs, klmodes, evecs_models=evecs_models)

    assert np.allclose(delta_KL_kl, delta_KL_refs, rtol=1e-10, atol=1e-10 * np.abs(delta_KL_refs).max())
