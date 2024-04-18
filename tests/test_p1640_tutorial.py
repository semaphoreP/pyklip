import os
import sys
import glob
from time import time
import numpy as np

if sys.version_info < (3, 3):
    from mock import patch
else:
    from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')

#sets up a patch object to mock.
@patch('pyklip.parallelized.klip_parallelized')
def test_p1640_tutorial(mock_klip_parallelized):
    """
    Tests P1640 support by running through the P1640 tutorial without the interactive parts.

    Follows the P1640 tutorial in docs and runs a test using the tutorial as a guideline. Goes through downloading the
    sample tarball, extracting the datacubes, fitting the grid spots, running KLIP on the datacubes, and outputting
    the files. The test checks that there are the correct number of files in each step outputted in the correct
    directories.
    The test also ignores all interactive modes such as vetting the cubes and grid spots.

    """

    #create a mocked klip parallelized
    mock_klip_parallelized.return_value = (np.zeros((3, 96, 281, 281)), np.array([140, 140]), np.array([1.]))


    # time it
    t1 = time()

    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    p1640datadir = os.path.join(directory,'p1640')

    filelist = glob.glob(os.path.join(p1640datadir, "*Occulted*.fits"))
    # should have 3 files in the directory after downloading and unzipping the tarball.
    assert (len(filelist) == 3)

    # Ignoring interactive vet the datacubes.
    good_cubes = []
    for file in filelist:
        good_cubes.append(os.path.abspath(file))

    # Fit grid spots
    import pyklip.instruments.P1640_support.P1640spots as P1640spots
    spot_filepath = os.path.join(p1640datadir,'shared_spot_folder')+ os.path.sep
    if not os.path.exists(spot_filepath):
        os.makedirs(spot_filepath)
    # spot_filepath = directory + os.path.sep + 'shared_spot_folder/'
    spot_filesuffix = '-spot'
    spot_fileext = 'csv'
    for test_file in good_cubes:
        spot_positions = P1640spots.get_single_file_spot_positions(test_file, rotated_spots=False)
        P1640spots.write_spots_to_file(test_file, spot_positions, spot_filepath,
                                       spotid=spot_filesuffix, ext=spot_fileext, overwrite=False)
    # should have 12 csv files outputted
    test1 = glob.glob("%stutorial*" % spot_filepath)
    assert (len(test1) == 12)

    # Again, ignoring interactive vet grid spots
    # run KLIP in SDI mode
    import pyklip.instruments.P1640 as P1640
    import pyklip.parallelized as parallelized
    dataset = P1640.P1640Data(filelist, spot_directory=spot_filepath)
    output = os.path.join(p1640datadir,'output')+ os.path.sep
    if not os.path.exists(output):
        os.makedirs(output)

    parallelized.klip_dataset(dataset, outputdir=output, fileprefix="woohoo", annuli=5, subsections=4, movement=3,
                              numbasis=[1, 20, 100], calibrate_flux=False, mode="SDI")
    # should have 4 outputted files
    p1640_globbed = glob.glob(output + "*")
    assert (len(p1640_globbed) == 4)

    print("{0} seconds to run".format(time() - t1))

    # ### for the purpose of this test, we are cleaning after ourselves the DL data
    # ### and the images
    # for item in os.listdir(p1640datadir):
    #     if (item.endswith(".fits")) or (item.endswith(".gz")):
    #         os.remove(os.path.join(p1640datadir, item))

    # for item in os.listdir(output):
    #     if item.endswith(".fits"):
    #         os.remove(os.path.join(output, item))

if __name__ == "__main__":
    test_p1640_tutorial()
