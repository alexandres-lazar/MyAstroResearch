#!/usr/bin/env python3

import os
import sys
import h5py
import numpy as np

if sys.version_info < (3, 6):
    sys.exit("!!! Please use Python 3.6+ to execute script!!!")

# ---------------------------------------------------------------------------

def main() -> None:
    compute_mass_function

# ---------------------------------------------------------------------------

def compute_mass_function(mass_catalog: float, volume: float=1.0, log_bounds: int = None) -> float:
    """Computes the stellar mass functions (or cumulative mass function) 
       from imported `catalog` within specified tupled bounds (else will compute own)
       and normalized by some volume (default is float 1.0 which just be the cumulative 
       number count).
    """
    results = dict()

    # keep only non-zero masses (Msol)
    mass_catalog = mass_catalog[np.nonzero(mass_catalog)]

    # generate a spanned array of masses and compute the cumulative mass 
    # function of the same shape
    nbins = 35
    if log_bounds is None:
        power_arr = np.linspace(np.log10(mass_catalog.min()), np.log10(mass_catalog.max()), nbins)
    elif isinstance(log_bounds, tuple):
        power_arr = np.linspace(log_bounds[0], log_bounds[1], nbins)
    else:
        sys.exit("!!! log_bounds must be a tuple: (min, max) !!!")
    mass_arr = 10.0 ** power_arr 
    cumul_numb = np.zeros(nbins) 
    for i, mass in enumerate(mass_arr): 
        cond = mass_catalog > mass
        cumul_numb[i] += np.where(cond)[0].shape[0]

    # keep non-zero values of arrays
    nonzero_index_1 = np.nonzero(cumul_numb)
    mass_arr = mass_arr[nonzero_index_1] 
    cumul_numb = cumul_numb[nonzero_index_1]

    # compute number density and gradient equivalent
    num_dens = cumul_numb / volume
    num_dens_grad = np.abs(np.gradient(num_dens) / np.gradient(np.log10(mass_arr)))
    nonzero_index_2 = np.nonzero(num_dens_grad)

    # save results to dictionary
    results['mass.bins:n'] = mass_arr
    results['number.density:n'] = num_dens 
    results['mass.bins:dlogM'] = mass_arr[nonzero_index_2]
    results['number.density:dlogM'] = num_dens_grad[nonzero_index_2] 

    return results


if __name__ == "__main__":
    pass
