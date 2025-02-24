__author__ = 'JB'

import os
from copy import copy
from glob import glob
from sys import stdout
import multiprocessing as mp
import itertools

import astropy.io.fits as pyfits
import numpy as np

from pyklip.instruments import GPI
import pyklip.kpp.utils.mathfunc as kppmath
import pyklip.spectra_management as spec

def calculate_matchedfilter_star(params):
    """
    Convert `f([1,2])` to `f(1,2)` call.
    It allows one to call calculate_shape3D_metric() with a tuple of parameters.
    """
    return calculate_matchedfilter(*params)

def calculate_matchedfilter(row_indices,col_indices,image,PSF,stamp_PSF_sky_mask,stamp_PSF_aper_mask, mute = True):
    '''
    Calculate the matched filter, cross correlation and flux map on a given image or datacube for the pixels targeted by
    row_indices and col_indices.
    These lists of indices can basically be given from the numpy.where function following the example:
        import numpy as np
        row_indices,col_indices = np.where(np.finite(np.mean(cube,axis=0)))
    By truncating the given lists in small pieces it is then easy to parallelized.

    Args:
        row_indices: Row indices list of the pixels where to calculate the metric in cube.
                            Indices should be given from a 2d image.
        col_indices: Column indices list of the pixels where to calculate the metric in cube.
                            Indices should be given from a 2d image.
        image: 2D or 3D image from which one wants the metric map. PSF_cube should be norm-2 normalized.
                    PSF_cube /= np.sqrt(np.sum(PSF_cube**2))
        PSF: 2D or 3D PSF template used for calculated the metric. If nl,ny_PSF,nx_PSF = PSF_cube.shape, nl is the
                         number of wavelength samples, ny_PSF and nx_PSF are the spatial dimensions of the PSF_cube.
        stamp_PSF_sky_mask: 2d mask of size (ny_PSF,nx_PSF) used to mask the central part of a stamp slice. It is used as
                            a type of a high pass filter. Before calculating the metric value of a stamp cube around a given
                            pixel the average value of the surroundings of each slice of that stamp cube will be removed.
                            The pixel used for calculating the average are the one equal to one in the mask.
        stamp_PSF_aper_mask: 3d mask for the aperture.
        mute: If True prevent printed log outputs.

    Return: Vector of length row_indices.size with the value of the metric for the corresponding pixels.
    '''

    image = np.array(image)
    if len(image.shape) == 2:
        cube = np.array([image])
        PSF_cube = np.array([PSF])
    else:
        cube = image
        PSF_cube = np.array(PSF)

    # Shape of the PSF cube
    nl,ny_PSF,nx_PSF = PSF_cube.shape

    # Number of rows and columns to add around a given pixel in order to extract a stamp.
    row_m = int(np.floor(ny_PSF/2.0))    # row_minus
    row_p = int(np.ceil(ny_PSF/2.0))     # row_plus
    col_m = int(np.floor(nx_PSF/2.0))    # col_minus
    col_p = int(np.ceil(nx_PSF/2.0))     # col_plus

    # Number of pixels on which the metric has to be computed
    N_it = row_indices.size
    # Define an shape vector full of nans
    mf_map = np.zeros((N_it,)) + np.nan
    cc_map = np.zeros((N_it,)) + np.nan
    flux_map = np.zeros((N_it,)) + np.nan
    # Loop over all pixels (row_indices[id],col_indices[id])
    for id,k,l in zip(range(N_it),row_indices,col_indices):
        if not mute:
            # Print the progress of the function
            stdout.write("\r{0}/{1}".format(id,N_it))
            stdout.flush()

        # Extract stamp cube around the current pixel from the whoel cube
        stamp_cube = copy(cube[:,(k-row_m):(k+row_p), (l-col_m):(l+col_p)])
        # wavelength dependent variance in the image
        var_per_wv = np.zeros(nl)
        # Remove average value of the surrounding pixels in each slice of the stamp cube
        for slice_id in range(nl):
            stamp_cube[slice_id,:,:] -= np.nanmean(stamp_cube[slice_id,:,:]*stamp_PSF_sky_mask)
            var_per_wv[slice_id] = np.nanvar(stamp_cube[slice_id,:,:]*stamp_PSF_sky_mask)
        try:
            mf_map[id] = np.nansum((stamp_PSF_aper_mask*PSF_cube*stamp_cube)/var_per_wv[:,None,None]) \
                         /np.sqrt(np.nansum((stamp_PSF_aper_mask*PSF_cube)**2/var_per_wv[:,None,None]))
            cc_map[id] = np.nansum(stamp_PSF_aper_mask*PSF_cube*stamp_cube)/np.sqrt(np.nansum((stamp_PSF_aper_mask*PSF_cube)**2))
            flux_map[id] = np.nansum((stamp_PSF_aper_mask*PSF_cube*stamp_cube)/var_per_wv[:,None,None]) \
                         /np.nansum((stamp_PSF_aper_mask*PSF_cube)**2/var_per_wv[:,None,None])
        except:
            # In case ones divide by zero...
            mf_map[id] =  np.nan
            cc_map[id] =  np.nan
            flux_map[id] =  np.nan

    return (mf_map,cc_map,flux_map)

def run_matchedfilter(image, PSF,N_threads=None,maskedge=True, aprad_frac=7./20.):
        """
        Perform a matched filter on the current loaded file.

        Args:
            image: image for which to get the matched filter.
            PSF: Template for the matched filter. It should include any kind of spectrum you which to use of the data is 3d.
            maskedge: If True (default), mask the edges of the image to prevent partial projection of the PSF.
                  If False, does not mask the edges.
            aprad_frac: fraction of the input PSF FOV to use as the aperture radius

        Return: Processed images (matched filter,cross correlation,estimated flux).
        """
        # Number of threads to be used in case of parallelization.
        if N_threads is None:
            N_threads = mp.cpu_count()
        else:
            N_threads = N_threads

        if PSF is not None:
            PSF_cube_arr = PSF
            if np.size(PSF_cube_arr.shape) == 2:
                ny_PSF,nx_PSF = PSF_cube_arr.shape
            if np.size(PSF_cube_arr.shape) == 3:
                nl_PSF,ny_PSF,nx_PSF = PSF_cube_arr.shape

        if not maskedge:
            if (len(image.shape) == 3):
                image_pad = np.pad(image,((0,0),(ny_PSF//2,ny_PSF//2),(nx_PSF//2,nx_PSF//2)),mode="constant",constant_values=np.nan)
            else:
                image_pad = np.pad(image,((ny_PSF//2,ny_PSF//2),(nx_PSF//2,nx_PSF//2)),mode="constant",constant_values=np.nan)
        else:
            image_pad = image

        if image_pad is not None:
            if np.size(image_pad.shape) == 2:
                ny,nx = image_pad.shape
            if np.size(image_pad.shape) == 3:
                nl,ny,nx = image_pad.shape

        if (len(image.shape) == 3):
            flat_cube = np.nanmean(image_pad,axis=0)
        else:
            flat_cube = image_pad

        # Get the nans pixels of the flat_cube. We won't bother trying to calculate metrics for those.
        flat_cube_nans = np.where(np.isnan(flat_cube))

        # Remove the very edges of the image. We can't calculate a proper projection of an image stamp onto the PSF if we
        # are too close from the edges of the array.
        flat_cube_mask = np.ones((ny,nx))
        flat_cube_mask[flat_cube_nans] = np.nan
        flat_cube_noEdges_mask = copy(flat_cube_mask)
        # remove the edges if not already nans
        flat_cube_noEdges_mask[0:ny_PSF//2,:] = np.nan
        flat_cube_noEdges_mask[:,0:nx_PSF//2] = np.nan
        flat_cube_noEdges_mask[(ny-ny_PSF//2):ny,:] = np.nan
        flat_cube_noEdges_mask[:,(nx-nx_PSF//2):nx] = np.nan
        # Get the pixel coordinates corresponding to non nan pixels and not too close from the edges of the array.
        flat_cube_noNans = np.where(np.isnan(flat_cube_noEdges_mask) == 0)

        mf_map = np.ones((ny,nx)) + np.nan
        cc_map = np.ones((ny,nx)) + np.nan
        flux_map = np.ones((ny,nx)) + np.nan

        # Calculate the criterion map.
        # For each pixel calculate the dot product of a stamp around it with the PSF.
        # We use the PSF cube to consider also the spectrum of the planet we are looking for.
        stamp_PSF_x_grid, stamp_PSF_y_grid = np.meshgrid(np.arange(0,nx_PSF,1)-nx_PSF//2,
                                                         np.arange(0,ny_PSF,1)-ny_PSF//2)
        aper_radius = np.min([ny_PSF,nx_PSF])*aprad_frac
        r_PSF_stamp = (stamp_PSF_x_grid)**2 +(stamp_PSF_y_grid)**2
        where_sky_mask = np.where(r_PSF_stamp < (aper_radius**2))
        stamp_PSF_sky_mask = np.ones((ny_PSF,nx_PSF))
        stamp_PSF_sky_mask[where_sky_mask] = np.nan
        where_aper_mask = np.where(r_PSF_stamp > (aper_radius**2))
        stamp_PSF_aper_mask = np.ones((ny_PSF,nx_PSF))
        stamp_PSF_aper_mask[where_aper_mask] = np.nan
        if (len(PSF_cube_arr.shape) == 3):
            # Duplicate the mask to get a mask cube.
            # Caution: No spectral widening implemented here
            stamp_PSF_aper_mask = np.tile(stamp_PSF_aper_mask,(nl,1,1))

        N_pix = flat_cube_noNans[0].size
        if N_threads > 0:
            chunk_size = N_pix//N_threads

        if N_threads > 0 and chunk_size != 0:
            pool = mp.Pool(processes=N_threads)

            ## cut images in N_threads part
            N_chunks = N_pix//chunk_size

            # Get the chunks
            chunks_row_indices = []
            chunks_col_indices = []
            for k in range(N_chunks-1):
                chunks_row_indices.append(flat_cube_noNans[0][(k*chunk_size):((k+1)*chunk_size)])
                chunks_col_indices.append(flat_cube_noNans[1][(k*chunk_size):((k+1)*chunk_size)])
            chunks_row_indices.append(flat_cube_noNans[0][((N_chunks-1)*chunk_size):N_pix])
            chunks_col_indices.append(flat_cube_noNans[1][((N_chunks-1)*chunk_size):N_pix])

            outputs_list = pool.map(calculate_matchedfilter_star, zip(chunks_row_indices,
                                                       chunks_col_indices,
                                                       itertools.repeat(image_pad),
                                                       itertools.repeat(PSF_cube_arr),
                                                       itertools.repeat(stamp_PSF_sky_mask),
                                                       itertools.repeat(stamp_PSF_aper_mask)))

            for row_indices,col_indices,out in zip(chunks_row_indices,chunks_col_indices,outputs_list):
                mf_map[(row_indices,col_indices)] = out[0]
                cc_map[(row_indices,col_indices)] = out[1]
                flux_map[(row_indices,col_indices)] = out[2]
            pool.close()
        else:
            out = calculate_matchedfilter(flat_cube_noNans[0],
                                                       flat_cube_noNans[1],
                                                       image_pad,
                                                       PSF_cube_arr,
                                                       stamp_PSF_sky_mask,
                                                       stamp_PSF_aper_mask)

            mf_map[flat_cube_noNans] = out[0]
            cc_map[flat_cube_noNans] = out[1]
            flux_map[flat_cube_noNans] = out[2]

        if not maskedge:
            mf_map = mf_map[ny_PSF//2:(ny-ny_PSF//2),nx_PSF//2:(nx-nx_PSF//2)]
            cc_map = cc_map[ny_PSF//2:(ny-ny_PSF//2),nx_PSF//2:(nx-nx_PSF//2)]
            flux_map = flux_map[ny_PSF//2:(ny-ny_PSF//2),nx_PSF//2:(nx-nx_PSF//2)]
        metricMap = (mf_map,cc_map,flux_map)
        return metricMap