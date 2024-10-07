import os
import sys
import glob
import time
import numpy as np

if sys.version_info < (3, 3):
    from mock import patch
else:
    from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

from pyklip.instruments import VAMPIRES


def load_vampires_test_data():
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    datadir = os.path.join(directory, "vampires")

    filelist = glob.glob(os.path.join(datadir, "*HD1160*.fits"))
    # should have 3 files in the directory after downloading and unzipping the tarball.
    assert len(filelist) == 3
    return filelist


def test_process_vampmires_data():
    filelist = load_vampires_test_data()
    for filepath in filelist:
        data = VAMPIRES._vampires_process_file(filepath)
        
        for key in ("cube", "centers", "PAs", "wvs", "wcs", "prim_hdr"):
            assert key in data

        assert data["cube"].shape == (4, 756, 756)
        assert data["centers"].shape == (4, 2)
        assert np.all(
            (
                data["centers"][0] == data["centers"][1],
                data["centers"][1] == data["centers"][2],
                data["centers"][2] == data["centers"][3],
            )
        )
        assert data["PAs"].shape == (4,)
        assert data["wvs"].shape == (4,)
        np.testing.assert_allclose(data["wvs"], (0.614, 0.670, 0.721, 0.761), atol=5e-4)


def test_vampiresdata_construction():
    filelist = load_vampires_test_data()

    # test construction
    dataset = VAMPIRES.VAMPIRESData(filelist)

    assert dataset.input.shape == (12, 756, 756)
    assert dataset.wvs.shape == (12,)
    assert dataset.centers.shape == (12, 2)
    assert dataset.PAs.shape == (12,)
    assert dataset.filenames.shape == (12,)
    assert dataset.filenums.shape == (12,)
    assert dataset.wcs.shape == (12,)
    np.testing.assert_allclose(dataset.filenums, np.repeat(range(3), 4))

    # test loading after the fact is equivalent
    dataset2 = VAMPIRES.VAMPIRESData()
    dataset2.readdata(filelist)

    np.testing.assert_allclose(dataset.input, dataset2.input)
    np.testing.assert_allclose(dataset.wvs, dataset2.wvs)
    np.testing.assert_allclose(dataset.centers, dataset2.centers)
    np.testing.assert_allclose(dataset.PAs, dataset2.PAs)
    np.testing.assert_allclose(dataset.filenames, dataset2.filenames)
    np.testing.assert_allclose(dataset.filenums, dataset2.filenums)


# sets up a patch object to mock.
@patch("pyklip.parallelized.klip_parallelized")
def test_vampires_klip(mock_klip_parallelized):
    """
    # Tests P1640 support by running through the P1640 tutorial without the interactive parts.

    # Follows the P1640 tutorial in docs and runs a test using the tutorial as a guideline. Goes through downloading the
    # sample tarball, extracting the datacubes, fitting the grid spots, running KLIP on the datacubes, and outputting
    # the files. The test checks that there are the correct number of files in each step outputted in the correct
    # directories.
    # The test also ignores all interactive modes such as vetting the cubes and grid spots.

    """

    # create a mocked klip parallelized
    mock_klip_parallelized.return_value = (
        np.zeros((3, 4, 756, 756)),
        np.array([140, 140]),
        np.array([1.0]),
    )

    from pyklip import parallelized

    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    datadir = os.path.join(directory, "vampires")
    outputdir = os.path.join(datadir, "output") + os.path.sep

    # time it
    t1 = time.perf_counter()

    # create dataset
    filelist = load_vampires_test_data()
    dataset = VAMPIRES.VAMPIRESData(filelist)

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    parallelized.klip_dataset(
        dataset,
        outputdir=outputdir,
        fileprefix="test",
        annuli=5,
        subsections=4,
        movement=0,
        numbasis=[1, 3, 10],
        calibrate_flux=False,
        mode="SDI",
    )
    # should have 4 outputted files
    output_files = glob.glob(outputdir + "*")
    assert len(output_files) == 4

    print("{0} seconds to run".format(time.perf_counter() - t1))


if __name__ == "__main__":
    test_process_vampmires_data()
    test_vampiresdata_construction()
    test_vampires_klip()
