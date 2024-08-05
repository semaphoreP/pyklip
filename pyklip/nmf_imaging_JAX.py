__license__ = "MIT"
__author__ = "Guangtun Ben Zhu (BGT) @ Johns Hopkins University"
__startdate__ = "2014.11.30"
__name__ = "nmf"
__module__ = "NonnegMFJAX"
__python_version__ = "3.5.1"
__numpy_version__ = "1.11.0"
__scipy_version__ = "0.17.0"

__lastdate__ = "2024.07.08"
__version__ = "0.1.1"


__all__ = ['NMF']

""" 
nmf.py

    This piece of software is developed and maintained by Guangtun Ben Zhu, 
    It is designed to solve nonnegative matrix factorization (NMF) given a dataset with heteroscedastic 
    uncertainties and missing data with a vectorized multiplicative update rule (Zhu 2016).
    The un-vectorized (i.e., indexed) update rule for NMF without uncertainties or missing data was
    originally developed by Lee & Seung (2000), and the un-vectorized update rule for NMF
    with uncertainties or missing data was originally developed by Blanton & Roweis (2007).

    As all the codes, this code can always be improved and any feedback will be greatly appreciated.

    Note:
      -- Between W and H, which one is the basis set and which one is the coefficient
         depends on how you interpret the data, because you can simply transpose everything
         as in X-WH versus X^T - (H^T)(W^T)
      -- Everything needs to be non-negative

    Here are some small tips for using this code:
      -- The algorithm can handle heteroscedastic uncertainties and missing data.
         You can supply the weight (V) and the mask (M) at the instantiation:

         >> g = nmf.NMF(X, V=V, M=M)

         This can also be very useful if you would like to iterate the process
         so that you can exclude certain new data by updating the mask.
         For example, if you want to perform a 3-sigma clipping after an iteration
         (assuming V is the inverse variance below):

         >> chi2_red, time_used = g.SolveNMF()
         >> New_M = np.copy(M)
         >> New_M[np.fabs(np.sqrt(V)*(X-np.dot(g.W, g.H)))>3] = False
         >> New_g = nmf.NMF(X, V=V, M=New_M)

         Caveat: Currently you need to re-instantiate the object whenever you update
         the weight (V), the mask (M), W, H or n_components.
         At the instantiation, the code makes a copy of everything.
         For big jobs with many iterations, this could be a severe bottleneck.
         For now, I think this is a safer way.

      -- It has W_only and H_only options. If you know H or W, and would like
         to calculate W or H. You can run, e.g.,

         >> chi2_red, time_used = g.SolveNMF(W_only=True)

         to get the other matrix (H in this case).


    Copyright (c) 2015-2016 Guangtun Ben Zhu

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
    and associated documentation files (the "Software"), to deal in the Software without 
    restriction, including without limitation the rights to use, copy, modify, merge, publish, 
    distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the 
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or 
    substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
    BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from jax import lax
from jax import random
from scipy import sparse
from time import time
from functools import partial
from jax import config
import os
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
  #Can uncomment above line to allow for dynamic memory allocation and deallocation
  #Seems like it is needed for calculating the whole image with high componenet numbers
config.update("jax_enable_x64", True)

# Some magic numbes
_largenumber = 1E100
_smallnumber = 1E-5

class NMF:
    """
    Nonnegative Matrix Factorization - Build a set of nonnegative basis components given 
    a dataset with Heteroscedastic uncertainties and missing data with a vectorized update rule.

    Algorithm:
      -- Iterative multiplicative update rule

    Input: 
      -- X: m x n matrix, the dataset

    Optional Input/Output: 
      -- n_components: desired size of the basis set, default 5

      -- V: m x n matrix, the weight, (usually) the inverse variance
      -- M: m x n binary matrix, the mask, False means missing/undesired data
      -- H: n_components x n matrix, the H matrix, usually interpreted as the coefficients
      -- W: m x n_components matrix, the W matrix, usually interpreted as the basis set
    
    (See README for how to retrieve the test data)
    
    Construct a new basis set with 12 components
    Instantiation: 
        >> g = nmf.NMF(flux, V=ivar, n_components=12)
    Run the solver: 
        >> chi2_red, time_used = g.SolveNMF()

    If you have a basis set, say W, you would like to calculate the coefficients:
        >> g = nmf.NMF(flux, V=ivar, W=W_known)
    Run the solver: 
        >> chi2_red, time_used = g.SolveNMF(H_only=True)

    Comments:
      -- Between W and H, which one is the basis set and which one is the coefficient 
         depends on how you interpret the data, because you can simply transpose everything
         as in X-WH versus X^T - (H^T)(W^T)
      -- Everything needs to be non-negative

    References:
      -- Guangtun Ben Zhu, 2016
         A Vectorized Algorithm for Nonnegative Matrix Factorization with 
         Heteroskedastic Uncertainties and Missing Data
         AJ/PASP, (to be submitted)
      -- Blanton, M. and Roweis, S. 2007
         K-corrections and Filter Transformations in the Ultraviolet, Optical, and Near-infrared
         The Astronomical Journal, 133, 734
      -- Lee, D. D., & Seung, H. S., 2001
         Algorithms for non-negative matrix factorization
         Advances in neural information processing systems, pp. 556-562

    To_do:
      -- 

    History:
        -- 22-May-2016, Documented, BGT, JHU
        -- 13-May-2016, Add projection mode (W_only, H_only), BGT, JHU
        -- 30-Nov-2014, Started, BGT, JHU
    """

    def __init__(self, X, W=None, H=None, V=None, M=None, n_components=5):
        """
        Initialization
        
        Required Input:
          X -- the input data set

        Optional Input/Output:
          -- n_components: desired size of the basis set, default 5

          -- V: m x n matrix, the weight, (usually) the inverse variance
          -- M: m x n binary matrix, the mask, False means missing/undesired data
          -- H: n_components x n matrix, the H matrix, usually interpreted as the coefficients
          -- W: m x n_components matrix, the W matrix, usually interpreted as the basis set
        """

        # I'm making a copy for the safety of everything; should not be a bottleneck
        key = random.PRNGKey(42)

        self.X = jnp.copy(X) 
        if (jnp.count_nonzero(self.X<0)>0):
            print("There are negative values in X. Setting them to be zero...", flush=True)
            self.X = self.X.at[self.X<0].set(0.0)
            #self.X[self.X<0] = 0.

        self.n_components = n_components
        self.maxiters = 1000
        self.tol = _smallnumber

        #print(self.X.shape[0], self.X.shape[1], self.n_components)

        if (W is None):
            key, subkey = random.split(key)
            self.W = random.uniform(subkey, shape=(self.X.shape[0],self.n_components))
            #self.W = np.random.rand(self.X.shape[0], self.n_components) ##
        else:
            if (W.shape != (self.X.shape[0], self.n_components)):
                raise ValueError("Initial W has wrong shape.")
            self.W = jnp.copy(W)
        if (jnp.count_nonzero(self.W<0)>0):
            print("There are negative values in W. Setting them to be zero...", flush=True)
            self.W = self.W.at[self.W<0].set(0.0)
            #self.W[self.W<0] = 0.

        if (H is None):
            key, subkey = random.split(key)
            self.H = random.uniform(subkey, shape=(self.n_components, self.X.shape[1]))
            #self.H = np.random.rand(self.n_components, self.X.shape[1]) ##
        else:
            if (H.shape != (self.n_components, self.X.shape[1])):
                raise ValueError("Initial H has wrong shape.")
            self.H = jnp.copy(H)
        if (jnp.count_nonzero(self.H<0)>0):
            print("There are negative values in H. Setting them to be zero...", flush=True)
            self.H = self.H.at[self.H<0].set(0.0)
            #self.H[self.H<0] = 0.

        if (V is None):
            self.V = jnp.ones(self.X.shape)
        else:
            if (V.shape != self.X.shape):
                raise ValueError("Initial V(Weight) has wrong shape.")
            self.V = jnp.copy(V)
        if (jnp.count_nonzero(self.V<0)>0):
            print("There are negative values in V. Setting them to be zero...", flush=True)
            self.V = self.V.at[self.V<0].set(0.0)
            #self.V[self.V<0] = 0.

        if (M is None):
            self.M = jnp.ones(self.X.shape, dtype=jnp.bool_)
        else:
            if (M.shape != self.X.shape):
                raise ValueError("M(ask) has wrong shape.")
            if (M.dtype != jnp.bool_):
                raise TypeError("M(ask) needs to be boolean.")
            self.M = jnp.copy(M)

        # Set masked elements to be zero
        self.V = self.V.at[(self.V*self.M)<=0].set(0)
        #self.V[(self.V*self.M)<=0] = 0
        self.V_size = jnp.count_nonzero(self.V) ####

def cost(self_X, self_W, self_H, self_V, self_V_size):
        """
        Total cost of a given set s
        """
        
        diff = self_X - jnp.dot(self_W, self_H)
        diff = jnp.array(diff, dtype=jnp.float64)
        self_V = jnp.array(self_V, dtype=jnp.float64)
        self_V_size = jnp.array(self_V_size, dtype=jnp.float64)
        chi2 = jnp.einsum('ij,ij', self_V*diff, diff)/self_V_size ####
        return chi2

@jit
def SolveNMF(self_maxiters, self_W, self_H, self_tol, self_X, self_V, self_V_size, W_only=False, H_only=False, sparsemode=False, maxiters=None, tol=None):
    """
    Construct the NMF basis

    Keywords:
        -- W_only: Only update W, assuming H is known
        -- H_only: Only update H, assuming W is known
            -- Only one of them can be set

    Optional Input:
        -- tol: convergence criterion, default 1E-5
        -- maxiters: allowed maximum number of iterations, default 1000

    Output: 
        -- chi2: reduced final cost
        -- time_used: time used in this run


    """

    t0 = time()

    if (maxiters is not None): 
        self_maxiters = maxiters
    if (tol is not None):
        self_tol = tol

    chi2 = cost(self_X, self_W, self_H, self_V, self_V_size)
    oldchi2 = _largenumber

    if (W_only and H_only):
        print("Both W_only and H_only are set to be True. Returning ...", flush=True)
        return (chi2, 0.)

    if (sparsemode == True):
        V = sparse.csr_matrix(self_V)
        VT = sparse.csr_matrix(jnp.transpose(self_V))
        multiply = sparse.csr_matrix.multiply
        dot = sparse.csr_matrix.dot
    else:
        V = jnp.copy(self_V)
        VT = jnp.transpose(V)
        multiply = jnp.multiply
        dot = jnp.dot

    #XV = self.X*self.V
    XV = multiply(V, self_X)
    XVT = multiply(VT, jnp.transpose(self_X))
    niter = 0

    def loop_body(carry):

        niter, oldchi2, chi2, self_W, self_H = carry

        def true1(self_H):
        #if (not W_only):
            H_up = jnp.dot(XVT, self_W)
            WHVT = jnp.multiply(VT, jnp.transpose(jnp.dot(self_W, self_H)))
            H_down = jnp.dot(WHVT, self_W)
            self_H = self_H*jnp.transpose(H_up)/jnp.transpose(H_down)
            return self_H
        def false1(self_H):
            return self_H

        # Update W
        #if (not H_only):
        def true2(self_W):
            W_up = jnp.dot(XV, jnp.transpose(self_H))
            WHV = jnp.multiply(V, jnp.dot(self_W, self_H))
            W_down = jnp.dot(WHV, jnp.transpose(self_H))
            self_W = self_W*W_up/W_down
            return self_W
        def false2(self_W):
            return self_W
        
        self_H = lax.cond(jnp.logical_not(W_only), true1, false1, self_H)
        self_W = lax.cond(jnp.logical_not(H_only), true2, false2, self_W)

        # chi2
        oldchi2 = chi2
        chi2 = cost(self_X, self_W, self_H, self_V, self_V_size)

        # Some quick check. May need its error class ...
        #value = lax.cond(jnp.isfinite(chi2), jnp.array(0), ValueError("NMF construction failed, likely due to missing data"), chi2)
        
        def print_err(niter):
            raise ValueError("NMF construction failed, likely due to missing data")
            return niter

        def print_new(niter):
            print("Current Chi2="+str(chi2)+", Previous Chi2="+str(oldchi2)+", Change="+str((oldchi2-chi2)/oldchi2*100.)+"% @ niters="+str(niter), flush=True)
            return niter

        def print_niter(niter):
            print("Iteration in re-initialization reaches maximum number = "+str(niter), flush=True)
            return niter

        def do_nothing(niter):
            return niter

        #value = lax.cond(jnp.isfinite(chi2), do_nothing, print_err, chi2)
        #value2 = lax.cond(jnp.mod(niter, 20)==0,print_new, do_nothing, niter)

        new_niter = niter + 1
        
        #value3 = lax.cond(new_niter == self_maxiters, print_niter, do_nothing, new_niter)
    
        return new_niter, oldchi2, chi2, self_W, self_H

    def loop_condition(carry):
        niter, oldchi2, chi2, self_W, self_H = carry
        return jnp.logical_and(niter < self_maxiters, ((oldchi2 - chi2) / oldchi2) > self_tol)

    carry_init = (niter, oldchi2, chi2, self_W, self_H)
    
    # Perform the loop
    niter, oldchi2, chi2, self_W, self_H = lax.while_loop(loop_condition, loop_body, carry_init)


    time_used = (time()-t0)/3600.
    print("Took "+str(time_used)+" seconds to reach current solution --> to solve NMF.", flush=True)
        #This is really uneeded since JAX JIT doesn't work well with time

    return chi2, niter, self_H, self_W

#############################################################################################################################################

# This code is the nmf_imaging.py adjusted for pyKLIP at https://bitbucket.org/pyKLIP/pyklip/src/master/pyklip/nmf_imaging.py
# Another version is kept at https://github.com/seawander/nmf_imaging/blob/master/nmf_imaging_for_pyKLIP.py
# Altered to match nmf_imagingJAX.py

import numpy as np
import os
from astropy.io import fits

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
        g_img = NMF(ref_columnized, V=1.0/ref_err_columnized**2, M = mask_columnized_boolean, n_components=n_components)
        #chi2, time_used = g_img.SolveNMF(maxiters=maxiters)
        chi2,time_used, g_img.H, g_img.W = SolveNMF(g_img.maxiters, g_img.W, g_img.H, g_img.tol, g_img.X, g_img.V, g_img.V_size, maxiters=maxiters)
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
                    g_img = NMF(ref_columnized, V = 1.0/ref_err_columnized**2, M = mask_columnized_boolean, n_components= n)
                else:
                    W_ini = np.random.rand(ref_columnized.shape[0], n)
                    W_ini[:, :(n-1)] = np.copy(g_img.W)
                    W_ini = np.array(W_ini, order = 'F') #Fortran ordering, column elements contiguous in memory.
                
                    H_ini = np.random.rand(n, ref_columnized.shape[1])
                    H_ini[:(n-1), :] = np.copy(g_img.H)
                    H_ini = np.array(H_ini, order = 'C') #C ordering, row elements contiguous in memory.                
                    g_img = NMF(ref_columnized, V = 1.0/ref_err_columnized**2, W = W_ini, H = H_ini, n_components= n)
                #chi2 = g_img.SolveNMF(maxiters=maxiters)
                chi2, time_used, g_img.H, g_img.W = SolveNMF(g_img.maxiters, g_img.W, g_img.H, g_img.tol, g_img.X, g_img.V, g_img.V_size, maxiters=maxiters)
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
                        g_img = NMF(ref_columnized, V = 1.0/ref_err_columnized**2, M = mask_columnized_boolean, n_components= n)
                    else:
                        W_ini = np.random.rand(ref_columnized.shape[0], n)
                        W_ini[:, :(n-1)] = np.copy(g_img.W)
                        W_ini = np.array(W_ini, order = 'F') #Fortran ordering, column elements contiguous in memory.
            
                        H_ini = np.random.rand(n, ref_columnized.shape[1])
                        H_ini[:(n-1), :] = np.copy(g_img.H)
                        H_ini = np.array(H_ini, order = 'C') #C ordering, row elements contiguous in memory.
            
                        g_img = NMF(ref_columnized, V = 1.0/ref_err_columnized**2, W = W_ini, H = H_ini, n_components= n)
                    #chi2 = g_img.SolveNMF(maxiters=maxiters)
                    chi2, time_used, g_img.H, g_img.W = SolveNMF(g_img.maxiters, g_img.W, g_img.H, g_img.tol, g_img.X, g_img.V, g_img.V_size, maxiters=maxiters)
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
            
                            g_img = NMF(ref_columnized, V = 1.0/ref_err_columnized**2, W = W_ini, H = H_ini, M = mask_columnized_boolean, n_components= n)
                        else:
                            W_ini = np.random.rand(ref_columnized.shape[0], n)
                            W_ini[:, :(n-1)] = np.copy(g_img.W)
                            W_ini = np.array(W_ini, order = 'F') #Fortran ordering, column elements contiguous in memory.
            
                            H_ini = np.random.rand(n, ref_columnized.shape[1])
                            H_ini[:(n-1), :] = np.copy(g_img.H)
                            H_ini = np.array(H_ini, order = 'C') #C ordering, row elements contiguous in memory.
            
                            g_img = NMF(ref_columnized, V = 1.0/ref_err_columnized**2, W = W_ini, H = H_ini, n_components= n)
                        #chi2 = g_img.SolveNMF(maxiters=maxiters)
                        chi2,time_used, g_img.H, g_img.W = SolveNMF(g_img.maxiters, g_img.W, g_img.H, g_img.tol, g_img.X, g_img.V, g_img.V_size, maxiters=maxiters)
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
        trg_img = NMF(trg_column, V=1/(trg_err_column**2), W=components_column, n_components = n_components)
        #(chi2, time_used) = trg_img.SolveNMF(H_only=True, maxiters = maxiters)
        chi2, time_used, trg_img.H, trg_img.W = SolveNMF(trg_img.maxiters, trg_img.W, trg_img.H, trg_img.tol, trg_img.X, trg_img.V, trg_img.V_size, H_only=True, maxiters = maxiters)
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
            trg_img = NMF(trg_column, V=1/trg_err_column**2, W=components_column[:, :i+1], n_components = i + 1)
            
            chi2, time_used, trg_img.H, trg_img.W = SolveNMF(trg_img.maxiters, trg_img.W, trg_img.H, trg_img.tol, trg_img.X, trg_img.V, trg_img.V_size, H_only=True, maxiters = maxiters)
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