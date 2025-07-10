import numpy as np
import pyklip.instruments.utils.radonCenter as radonCenter

def test_radon_centering():
    """
    Tests radon centering using mock data
    """

    dat = np.zeros([101, 51], dtype=float)

    cent = [dat.shape[1]//2, dat.shape[0]//2]

    y, x = np.indices(dat.shape)
    dx = x - cent[0]
    dy = y - cent[1]
    diag = np.where(np.abs(dx) == np.abs(dy))

    dat[diag] = 1

    xcen, ycen = radonCenter.searchCenter(dat, cent[0]-1, cent[1]+1, size_window=15, theta=[45, 136], size_cost=7)

    assert np.isclose(xcen, cent[0], atol=0.1)
    assert np.isclose(ycen, cent[1], atol=0.1)

if __name__ == "__main__":
    test_radon_centering()
    print("Radon centering test passed.")