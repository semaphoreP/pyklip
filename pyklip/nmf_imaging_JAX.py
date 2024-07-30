# This code is the nmf_imaging.py adjusted for pyKLIP at https://bitbucket.org/pyKLIP/pyklip/src/master/pyklip/nmf_imaging.py
# Another version is kept at https://github.com/seawander/nmf_imaging/blob/master/nmf_imaging_for_pyKLIP.py
# Altered to match nmf_imagingJAX.py

import numpy as np
import os
from astropy.io import fits
import pyklip.NonnegMFJAX as nmf

def data_masked_only(data, mask = None):
    """ Return the data where the same regions are ignored in all the data
    Args:
        data: (p, N) where N is the number of references, p is the number of pixels in each reference
        mask: 1d array containing only 1 and 0, with 0 will be ignored in the output
    Returns:
        data_focused: (p_focused, N) where p_focused is the number of 1's in mask
    """ 
    p_focused = int(np.nansum(mask)) # number of pixels that will be included
    if len(data.shape) == 2:
        data_focused = np.zeros((p_focused, data.shape[1])) # output
        for i in range(data.shape[1]):
            data_focused[:, i] = data[:, i][np.where(mask == 1)]
    elif len(data.shape) == 1:
        data_focused = data[mask == 1] # output
        data_focused = data_focused[:, np.newaxis]
    
    return data_focused
    
def data_masked_only_revert(data, mask = None):
    """ Return the data where the same regions were ignored in all the data
    Args:
        data: (p_focused, N) where N is the number of references, p_focused is the number of 1's in mask
        mask: 1d array containing only 1 and 0, with 0 was previously ignored in the input
    Returns:
        data_focused_revert: (p, N) where p is the number of pixels in each reference
    """ 
    data_focused_revert = np.zeros((len(mask), data.shape[1])) * np.nan
    
    for i in range(data.shape[1]):
        data_focused_revert[np.where(mask == 1), i] = data[:, i]
    
    return data_focused_revert
     
def NMFcomponents(ref, ref_err = None, mask = None, n_components = None, maxiters = 1e3, oneByOne = False, path_save = None):
    """ref and ref_err should be (n * height * width) where n is the number of references. Mask is the region we are interested in.
    if mask is a 3D array (binary, 0 and 1), then you can mask out different regions in the ref.
    if path_save is provided, then the code will start from there.
    """
    ref = ref.T # matrix transpose to comply with statistician standards on storing data
    
    if ref_err is None:
        ref_err = np.sqrt(ref)
    else:
        ref_err = ref_err.T # matrix transpose for the error map as well

    if (n_components is None) or (n_components > ref.shape[0]):
        n_components = ref.shape[0]

    if mask is None:
        mask = np.ones_like(ref)
    
    # ignore certain values in component construction
    mask[ref <= 0] = 0 # 1. negative values
    mask[~np.isfinite(ref)] = 0 # 2. infinite values
    mask[np.isnan(ref)] = 0 # 3. nan values
    
    mask[ref_err <= 0] = 0 # 1. negative values in input error map
    mask[~np.isfinite(ref_err)] = 0 # 2. infinite values in input error map
    mask[np.isnan(ref_err)] = 0 # 3. nan values in input error map

    # speed up component calculation by ignoring the commonly-ignored elements across all references
    mask_mark = np.nansum(mask, axis = 1)
    mask_mark[mask_mark != 0] = 1 # 1 means that there is coverage in at least one of the refs

    ref_columnized = data_masked_only(ref, mask = mask_mark)
    ref_err_columnized = data_masked_only(ref_err, mask = mask_mark)
    mask_columnized = data_masked_only(mask, mask = mask_mark)
    mask_columnized_boolean = np.array(data_masked_only(mask, mask = mask_mark), dtype = bool)
    #ref_columnized[mask_columnized == 0] = 0 # assign 0 to ignored values, should not impact the final result given the usage of mask_columnized_boolean
    ref_err_columnized[mask_columnized == 0] = np.nanmax(ref_err_columnized) # assign max uncertainty to ignored values, should not impact the final result

    
    # component calculation
    components_column = 0
    if not oneByOne:
        print("Building components NOT one by one... If you want the one-by-one method (suggested), please set oneByOne = True.")
        g_img = nmf.NMF(ref_columnized, V=1.0/ref_err_columnized**2, M = mask_columnized_boolean, n_components=n_components)
        #chi2, time_used = g_img.SolveNMF(maxiters=maxiters)
        chi2,time_used, g_img.H, g_img.W = nmf.SolveNMF(g_img.maxiters, g_img.W, g_img.H, g_img.tol, g_img.X, g_img.V, g_img.V_size, maxiters=maxiters)
        print(f'chi square of {round(chi2)}')
        components_column = g_img.W/np.sqrt(np.nansum(g_img.W**2, axis = 0)) #normalize the components        
        components = data_masked_only_revert(components_column, mask = mask_mark) 
    else:
        print("Building components one by one...")
        if path_save is None:
            for i in range(n_components):
                print("\t" + str(i+1) + " of " + str(n_components))
                n = i + 1
                if (i == 0):
                    g_img = nmf.NMF(ref_columnized, V = 1.0/ref_err_columnized**2, M = mask_columnized_boolean, n_components= n)
                else:
                    W_ini = np.random.rand(ref_columnized.shape[0], n)
                    W_ini[:, :(n-1)] = np.copy(g_img.W)
                    W_ini = np.array(W_ini, order = 'F') #Fortran ordering, column elements contiguous in memory.
                
                    H_ini = np.random.rand(n, ref_columnized.shape[1])
                    H_ini[:(n-1), :] = np.copy(g_img.H)
                    H_ini = np.array(H_ini, order = 'C') #C ordering, row elements contiguous in memory.                
                    g_img = nmf.NMF(ref_columnized, V = 1.0/ref_err_columnized**2, W = W_ini, H = H_ini, n_components= n)
                #chi2 = g_img.SolveNMF(maxiters=maxiters)
                chi2, time_used, g_img.H, g_img.W = nmf.SolveNMF(g_img.maxiters, g_img.W, g_img.H, g_img.tol, g_img.X, g_img.V, g_img.V_size, maxiters=maxiters)
                print(f'chi square of {round(chi2)}')
                components_column = g_img.W/np.sqrt(np.nansum(g_img.W**2, axis = 0)) #normalize the components
                components = data_masked_only_revert(components_column, mask = mask_mark) 
        else:
            print('\t path_save provided, you might want to load data and continue previous component calculation')
            print('\t\t loading from ' + path_save + '_comp.fits for components.')
            if not os.path.exists(path_save + '_comp.fits'):
                print('\t\t ' + path_save + '_comp.fits does not exist, calculating from scratch.')
                for i in range(n_components):
                    print("\t" + str(i+1) + " of " + str(n_components))
                    n = i + 1
                    if (i == 0):
                        g_img = nmf.NMF(ref_columnized, V = 1.0/ref_err_columnized**2, M = mask_columnized_boolean, n_components= n)
                    else:
                        W_ini = np.random.rand(ref_columnized.shape[0], n)
                        W_ini[:, :(n-1)] = np.copy(g_img.W)
                        W_ini = np.array(W_ini, order = 'F') #Fortran ordering, column elements contiguous in memory.
            
                        H_ini = np.random.rand(n, ref_columnized.shape[1])
                        H_ini[:(n-1), :] = np.copy(g_img.H)
                        H_ini = np.array(H_ini, order = 'C') #C ordering, row elements contiguous in memory.
            
                        g_img = nmf.NMF(ref_columnized, V = 1.0/ref_err_columnized**2, W = W_ini, H = H_ini, n_components= n)
                    #chi2 = g_img.SolveNMF(maxiters=maxiters)
                    chi2, time_used, g_img.H, g_img.W = nmf.SolveNMF(g_img.maxiters, g_img.W, g_img.H, g_img.tol, g_img.X, g_img.V, g_img.V_size, maxiters=maxiters)
                    print(f'chi square of {round(chi2)}')
                    print('\t\t\t Calculation for ' + str(n) + ' components done, overwriting raw 2D component matrix at ' + path_save + '_comp.fits')
                    fits.writeto(path_save + '_comp.fits', g_img.W, overwrite = True)
                    print('\t\t\t Calculation for ' + str(n) + ' components done, overwriting raw 2D coefficient matrix at ' + path_save + '_coef.fits')
                    fits.writeto(path_save + '_coef.fits', g_img.H, overwrite = True)
                    components_column = g_img.W/np.sqrt(np.nansum(g_img.W**2, axis = 0)) #normalize the components
                    components = data_masked_only_revert(components_column, mask = mask_mark)
            else:
                W_assign = fits.getdata(path_save + '_comp.fits')
                H_assign = fits.getdata(path_save + '_coef.fits')
                if W_assign.shape[1] >= n_components:
                    print('You have already had ' + str(W_assign.shape[1]) + ' components while asking for ' + str(n_components) + '. Returning to your input.')
                    components_column = W_assign/np.sqrt(np.nansum(W_assign**2, axis = 0))
                    components = data_masked_only_revert(components_column, mask = mask_mark)
                else:
                    print('You are asking for ' + str(n_components) + ' components. Building the rest based on the ' + str(W_assign.shape[1]) + ' provided.')

                    for i in range(W_assign.shape[1], n_components):
                        print("\t" + str(i+1) + " of " + str(n_components))
                        n = i + 1
                        if (i == W_assign.shape[1]):
                            W_ini = np.random.rand(ref_columnized.shape[0], n)
                            W_ini[:, :(n-1)] = np.copy(W_assign)
                            W_ini = np.array(W_ini, order = 'F') #Fortran ordering, column elements contiguous in memory.
            
                            H_ini = np.random.rand(n, ref_columnized.shape[1])
                            H_ini[:(n-1), :] = np.copy(H_assign)
                            H_ini = np.array(H_ini, order = 'C') #C ordering, row elements contiguous in memory.
            
                            g_img = nmf.NMF(ref_columnized, V = 1.0/ref_err_columnized**2, W = W_ini, H = H_ini, M = mask_columnized_boolean, n_components= n)
                        else:
                            W_ini = np.random.rand(ref_columnized.shape[0], n)
                            W_ini[:, :(n-1)] = np.copy(g_img.W)
                            W_ini = np.array(W_ini, order = 'F') #Fortran ordering, column elements contiguous in memory.
            
                            H_ini = np.random.rand(n, ref_columnized.shape[1])
                            H_ini[:(n-1), :] = np.copy(g_img.H)
                            H_ini = np.array(H_ini, order = 'C') #C ordering, row elements contiguous in memory.
            
                            g_img = nmf.NMF(ref_columnized, V = 1.0/ref_err_columnized**2, W = W_ini, H = H_ini, n_components= n)
                        #chi2 = g_img.SolveNMF(maxiters=maxiters)
                        chi2,time_used, g_img.H, g_img.W = nmf.SolveNMF(g_img.maxiters, g_img.W, g_img.H, g_img.tol, g_img.X, g_img.V, g_img.V_size, maxiters=maxiters)
                        print(f'chi square of {round(chi2)}')
                        print('\t\t\t Calculation for ' + str(n) + ' components done, overwriting raw 2D component matrix at ' + path_save + '_comp.fits')
                        fits.writeto(path_save + '_comp.fits', g_img.W, overwrite = True)
                        print('\t\t\t Calculation for ' + str(n) + ' components done, overwriting raw 2D coefficient matrix at ' + path_save + '_coef.fits')
                        fits.writeto(path_save + '_coef.fits', g_img.H, overwrite = True)
                        components_column = g_img.W/np.sqrt(np.nansum(g_img.W**2, axis = 0)) #normalize the components
                        components = data_masked_only_revert(components_column, mask = mask_mark)                  
    return components.T

def NMFmodelling(trg, components, n_components = None, trg_err = None, mask_components = None, maxiters = 1e3, returnChi2 = False, projectionsOnly = False, coefsAlso = False, cube = False, trgThresh = 1.0, mask_data_imputation = None):
    """
    trg: height * width
    components: n * height * width, calculated using NMFcomponents.
        mask_components: height * width, the mask used in NMFcomponents.
    n_components: how many components do you want to use. If None, all the components will be used.
    
    projectionsOnly: output the individual projection results.
    cube: whether output a cube or not (increasing the number of components).
    trgThresh: ignore the regions with low photon counts. Especially when they are ~10^-15 or smaller. I chose 1 in this case.
    mask_data_imputation: a 2D mask to model the planet-/disk-less regions (0 means there are planets/disks). The reconstructed model will still model the planet-/disk- regions, but without any input from them.
    """
    if n_components is None:
        n_components = components.shape[0]
    
    if trg_err is None:
        trg_err = np.sqrt(trg)
    
    if mask_components is not None:
        mask_components = np.nansum(mask_components, axis = 1)
        mask_components[mask_components != 0] = 1   

    if mask_components is None:
        mask_components = np.ones(trg.shape)
        mask_components[np.where(np.isnan(components[0]))] = 0

    components_column_all = data_masked_only(components[:n_components].T, mask = mask_components)
    components_column_all = components_column_all/np.sqrt(np.nansum(components_column_all**2, axis = 0))

    if mask_data_imputation is None:
        flag_di = 0
        mask_data_imputation = np.ones(trg.shape)
    else:
        flag_di = 1
        print('Data Imputation!')
    
    mask = mask_components*mask_data_imputation

    trg[trg < trgThresh] = 0.
    trg_err[trg == 0] = np.nanmax(trg_err)

    mask[trg <= 0] = 0
    mask[np.isnan(trg)] = 0
    mask[~np.isfinite(trg)] = 0
    
    #Columnize the target and its error.
    trg_column = data_masked_only(trg, mask = mask)
    trg_err_column = data_masked_only(trg_err, mask = mask)
    components_column = data_masked_only(components.T, mask = mask)

    if not cube:
        trg_img = nmf.NMF(trg_column, V=1/(trg_err_column**2), W=components_column, n_components = n_components)
        #(chi2, time_used) = trg_img.SolveNMF(H_only=True, maxiters = maxiters)
        chi2, time_used, trg_img.H, trg_img.W = nmf.SolveNMF(trg_img.maxiters, trg_img.W, trg_img.H, trg_img.tol, trg_img.X, trg_img.V, trg_img.V_size, H_only=True, maxiters = maxiters)
        print(f'chi square of {round(chi2)}')
        coefs = trg_img.H
        if flag_di == 0:
            model_column = np.dot(components_column, coefs)
            model = data_masked_only_revert(model_column, mask)
            model[np.where(mask == 0)] = np.nan
        elif flag_di == 1:
            model_column = np.dot(components_column_all, coefs)
            model = data_masked_only_revert(model_column, mask_components)
            model[np.where(mask_components == 0)] = np.nan
    else:
        print("Building models one by one...")
        for i in range(n_components):
            print("\t" + str(i+1) + " of " + str(n_components))
            trg_img = nmf.NMF(trg_column, V=1/trg_err_column**2, W=components_column[:, :i+1], n_components = i + 1)
            
            chi2, time_used, trg_img.H, trg_img.W = nmf.SolveNMF(trg_img.maxiters, trg_img.W, trg_img.H, trg_img.tol, trg_img.X, trg_img.V, trg_img.V_size, H_only=True, maxiters = maxiters)
            print(f' chi square of {round(chi2)}')
            coefs = trg_img.H

            if flag_di == 0:
                model_column = np.dot(components_column[:, :i+1], coefs)
                model_slice = data_masked_only_revert(model_column, mask)
                model_slice[np.where(mask == 0)] = np.nan
            elif flag_di == 1:
                model_column = np.dot(components_column_all[:, :i+1], coefs)
                model_slice = data_masked_only_revert(model_column, mask)
                model_slice[np.where(mask == 0)] = np.nan
            if i == 0:
                model = np.zeros((n_components, ) + model_slice.shape)
            model[i] = model_slice
            
    if returnChi2:
        return model, chi2
    if coefsAlso:
        return model, coefs
    
    return model.flatten()

def NMFsubtraction(trg, model, mask = None, frac = 1):
    """NMF subtraction with a correction factor, frac."""
    if mask is not None:
        trg = trg*mask      
        model = model*mask      
    if np.shape(np.asarray(frac)) == ():
        return trg-model*frac
    result = np.zeros((len(frac), ) + model.shape)
    for i, fraction in enumerate(frac):
        result[i] = trg-model*fraction
    return result
    
def NMFbff(trg, model, fracs = None):
    """BFF subtraction.
    Input: trg, model, mask (if need to be), fracs (if need to be).
    Output: best frac
    """
    if fracs is None:
        fracs = np.arange(0.80, 1.001, 0.001)
    
    std_infos = np.zeros(fracs.shape)
    
    for i, frac in enumerate(fracs):
        data_slice = trg - model*frac
        while 1:
            if np.nansum(data_slice > np.nanmedian(data_slice) + 3*np.nanstd(data_slice)) == 0 or np.nansum(data_slice < np.nanmedian(data_slice) -3*np.nanstd(data_slice)) == 0: 
                break
            data_slice[data_slice > np.nanmedian(data_slice) + 3*np.nanstd(data_slice)] = np.nan
            data_slice[data_slice < np.nanmedian(data_slice) - 3*np.nanstd(data_slice)] = np.nan # Modified from -10 on 2018/07/12
        std_info = np.nanstd(data_slice)
        std_infos[i] = std_info
    return fracs[np.where(std_infos == np.nanmin(std_infos))]    
 
def nmf_func(trg, refs, trg_err = None, refs_err = None, mask = None, componentNum = [5], maxiters = 1e5, oneByOne = True, trg_type = 'disk',mask_data_imputation = None):
    """
    Main NMF function for high contrast imaging.
    Args:  
        trg (1D array): target image, dimension: height * width.
        refs (2D array): reference cube, dimension: referenceNumber * height * width.
        trg_err, ref_err: uncertainty for trg and refs, repectively. If None is given, the squareroot of the two arrays will be adopted.
    
        componentNum (integer): number of components to be used. Default: 5. Caution: choosing too many components will slow down the computation.
        maxiters (integer): number of iterations needed. Default: 10^5.
        trg_type (string,  default: "disk" or "d" for circumsetllar disks by Bin Ren, the user can use "planet" or "p" for planets): are we aiming at finding circumstellar disks or planets?
    Returns: 
        result (1D array): NMF modeling result. Only the final subtraction result is returned.
    """
    # Imitating nmf_imaging.py
    badpix = np.where(np.isnan(trg))
    trg[badpix] = 0

    trg[trg<0]=0
    refs[refs<0]=0
    refs[np.isnan(refs)]=0

    if refs_err is None:
        refs_err = np.ones(refs.shape)
    if trg_err is None:
        trg_err = np.ones(trg.shape)

    maxcomponents = max(componentNum)

    if maxcomponents > refs.shape[0]:
            maxcomponents = refs.shape[0]
    
    components = NMFcomponents(refs, ref_err = refs_err, n_components = maxcomponents, maxiters = maxiters, 
                                oneByOne=oneByOne, mask = mask)
    
    results = []
    for num in componentNum:
        model = NMFmodelling(trg = trg, components = components[:num], n_components = num, trg_err = trg_err, mask_components = mask,
                             maxiters=maxiters, trgThresh=0.0, mask_data_imputation=mask_data_imputation)
        
        #Bff Procedure below: for planets, it will not be implemented.
        if trg_type == 'p' or trg_type == 'planet': # planets
            best_frac = 1
        elif trg_type == 'd' or trg_type == 'disk': # disks
            best_frac = NMFbff(trg = trg, model = model)

        result = NMFsubtraction(trg = trg, model = model, frac = best_frac)
        result = result.flatten()
        result[badpix] = np.nan
        results.append(result)

    results = np.array(results).T

    return results