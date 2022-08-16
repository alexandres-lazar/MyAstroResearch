#!/usr/bin/env python3

import os
import sys
import h5py
import numpy as np

if sys.version_info < (3, 6):
    sys.exit("!!! Please use Python 3.6+ to execute script!!!")

# ---------------------------------------------------------------------------

def main() -> None:
    load_subhalo_sample()
    load_star_catalog()
    load_halo_catalog()
    load_ahf_catalog()

# ---------------------------------------------------------------------------

def load_subhalo_sample(path: str) -> None:
    """
    """
    results = dict()

    # first check if sample is located (or exists) in passed path
    path_check = os.path.isfile(path)
    if not path_check:
        sys.exit(f"No subhalo sample located in: '{path}'")
    else:
        pass

    # Extract hdf5 contents and import to a dictionary
    with h5py.File(path, 'r') as h5:
        key_list = [key for key in h5.keys()]
        for key in key_list: 
            results[key] = h5[key][()]

    return results


def load_star_catalog(path: str, key_dict=None) -> None:
    """Load relavent quantities from Andrew Wetzel's hdf5 star catalogs
    """
    return load_halo_catalog(path, key_dict)


def load_halo_catalog(path: str, key_dict=None) -> None:
    """Load relavent quantities from Andrew Wetzel's hdf5 rockstar catalogs
    """
    results = dict()

    # load in relavent hdf5 quantities through `key_dict`
    with h5py.File(path, 'r') as h5:
        if key_dict == None:
            for key in h5.keys():
                results[key] = h5[key][()]
        else:
            for key in key_dict:
                results[key] = h5[key][()]

    return results


def load_ahf_catalog(path: str = None) -> float:
    """Loads and opens specific contents of AHF halo catalogs based on
       given catalog location (`path`). Note that this is for specific 
       hydrodynamic simulation, so be mindful of the columns being 
       iterated from (see header comments in *.AHF_halos file).
    """
    results = dict()
    
    try:
        print("... Fetching stellar masses from AHF catalog")
        catalog = np.loadtxt(path, usecols=64, comments='#')
        results['stellar.mass'] = catalog[:] # Msol 
    except ValueError:
        print("!!! No stellar masses")
 
    return results


if __name__ == "__main__":
    pass
