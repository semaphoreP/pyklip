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