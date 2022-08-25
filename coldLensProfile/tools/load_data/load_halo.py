#!/usr/bin/env python3

import os
import sys
import h5py
import numpy as np

if sys.version_info < (3, 6):
    sys.exit("!!! Please use Python 3.6+ to execute script!!!")

# ---------------------------------------------------------------------------

def main() -> None:
    load_ahf_catalog()
    sample_from_mass_range()
    convert_factors()

# ---------------------------------------------------------------------------

def load_ahf_catalog(haloPath: str, 
                    cosmo_box: bool = False,
                     high_res_only: bool = False, 
                     field_halo_only: bool = False) -> float:
    """Loads and opens specific contents of AHF halo catalogs based on
       given catalog location (`path`). Note that this is for specific 
       hydrodynamic simulation, so be mindful of the columns being 
       iterated from (see header comments in *.AHF_halos file).
    """
    results = dict()

    print("... Fetching data from AHF catalog")
    
    # important to load halo IDs seperatly for correct formating
    results['halo.id'] = np.loadtxt(haloPath, comments='#', usecols=0, dtype=np.uint64)
    
    catalog = np.loadtxt(haloPath, comments='#')
    results['virial.mass'] = catalog[:, 3]           # Msol/h
    results['virial.radius'] = catalog[:, 11]        # comoving kpc/h
    results['max.radius'] = catalog[:, 12]           # comoving kpc/h
    results['max.circ'] = catalog[:,16]              # km/s
    results['position'] = np.zeros((results['halo.id'].shape[0], 3)) # comoving kpc/h
    results['position'][:, 0] += catalog[:, 5]
    results['position'][:, 1] += catalog[:, 6]
    results['position'][:, 2] += catalog[:, 7]

    # partitiion halo populations based on box or zoom simulations
    if cosmo_box:
        pid = catalog[:, 1]
        subFlag = np.zeros(results['halo.id'].shape[0])
        subFlag[pid>0] = 1

    # logic pass to sample conditions
    if high_res_only or field_halo_only: 
        if high_res_only:
            cond = (catalog[:, 37] > 0.99)
            if field_halo_only:
                if cosmo_box:
                    cond = cond & (subFlag == 0)
                else:
                    cond = cond & (catalog[:, 1] == -1)
        elif field_halo_only: 
            if cosmo_box:
                cond = (subFlag == 0)
            else:
                cond = (catalog[:, 1] == -1)

        for skey, sval in results.items():
            results[skey] = sval[cond]

    return results


def sample_from_mass_range(catalog, min_mass: float, max_mass: float):
    """Masks catalog for halos only within `min_mass` and `max_mass`"""
    new_catalog = dict()
    cond1 = min_mass < catalog['virial.mass'] 
    cond2 = catalog['virial.mass'] < max_mass 
    cond = cond1 & cond2
    for skey, sval in catalog.items():
        new_catalog[skey] = sval[cond]
    return new_catalog


def convert_factors(catalog, hubble: float, scale_factor: float) -> None:
    """Transform properties to physcial coordinates and removes hubble factors"""
    new_catalog = catalog.copy() 
    new_catalog['virial.mass'] /= hubble
    new_catalog['virial.radius'] *= scale_factor / hubble
    new_catalog['max.radius'] *= scale_factor / hubble
    new_catalog['position'] *= scale_factor / hubble
    return new_catalog


if __name__ == "__main__":
    pass
