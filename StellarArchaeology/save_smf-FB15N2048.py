#! usr/bin/env python3

# system ----
import os
import sys
import h5py
import numpy as np

# local ----
simPath = '/data18/brunello/aalazar/FIREbox/FB15N2048'
outPath = '/data17/grenache/aalazar/projects/researchproject_010/output'
sys.path.append("/export/home/aalazar/code/")

if sys.version_info < (3, 6):
    sys.exit("!!! Please use Python 3.6+ to execute script!!!")

# Global FIREbox parameters (Planck-15)
HUBBLE = 0.6774
BOX_LENGTH = 15.0 / HUBBLE     # comoving Mpc
BOX_VOLUME = BOX_LENGTH**3     # comoving Mpc^3

# AHF file paths for FIREbox based on snapshot number. Dictionary `snapList` 
# is set up to specify which AHF file paths are, as well as dictionary key to
# adopt as this will be saved to a hdf5 file. Same for the redshift list, `zlist`.
snapList = dict()
snapList['snapshot_026'] = f"{simPath}/halo/AHF/catalog/026/FB15N2048.z6.000.AHF_halos"
snapList['snapshot_020'] = f"{simPath}/halo/AHF/catalog/020/FB15N2048.z7.113.AHF_halos"
snapList['snapshot_016'] = f"{simPath}/halo/AHF/catalog/016/FB15N2048.z8.130.AHF_halos"

zList = dict()
zList['snapshot_026'] = 6.000
zList['snapshot_020'] = 7.113
zList['snapshot_016'] = 8.130

# import self-made analysis modules
from tools import lss

# ---------------------------------------------------------------------------

def main() -> None:
    print("-" * 75)
    print("*** Beginning analysis ***")
    print("-" * 75)

    results = dict()
    for snap, catalog_path in snapList.items(): 

        print(f">>> Redshift {zList[snap]}")

        # load in desired AHF halo catalog and compute resulting stellar 
        # mass function
        catalog = load_ahf_catalog(catalog_path) 
        print("... Computing mass function")
        smf_results = lss.compute_mass_function(catalog['stellar.mass'], 
                                                volume=BOX_VOLUME,
                                                log_bounds=(3, 12))
      
        # results are save to snapshot specific dictionary which is then  
        # saved to all encompasing dictionary
        results[snap] = dict()
        results[snap]['redshift'] = zList[snap]
        results[snap] = smf_results.copy()

        print(">>> Done!")
        print("-" * 75)

    save_to_hdf5(results)

    print("*** Results saved to HDF5 ***")
    print("-" * 75)

# ---------------------------------------------------------------------------

def save_to_hdf5(local_dict=None) -> None:
    """Creates all encompasing hdf5 file based on contents in passed 
       `local_dict`. 
    """
    h5 = h5py.File(f"{outPath}/smf_FB15N2048.hdf5", 'w')

    for snap, snap_dict in local_dict.items():
        h5s = h5.create_group(snap)
        for skey, sdata in snap_dict.items():
            h5s.create_dataset(skey, data=sdata)

    h5.close()

    return None


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
        results['stellar.mass'] = catalog[:] / HUBBLE # Msol 
    except ValueError:
        print("!!! No stellar masses")
 
    return results


if __name__ == "__main__":
    main()
