#!/usr/bin/env python3

import os
import sys
import h5py
import numpy as np

if sys.version_info < (3, 6):
    sys.exit("!!! Please use Python 3.6+ to execute script!!!")

# ---------------------------------------------------------------------------

def main() -> None:
    load_particles()
    construct_kdtree()
    particles_in_halo_kd_tree()
    in_halo_via_kdtree()

# ---------------------------------------------------------------------------

def in_halo_via_kdtree(kdtree, halo_pos: float, halo_rad: float) -> int:
    """Returns particle index for a given halo position and radius
       (`halo_pos` and `halo_rad`) via ball in point method of the 
       kdtree (`kdtree`).
    """ 
    return kdtree.query_ball_point(halo_pos, halo_rad)


def construct_kdtree(pos: float) -> None:
    """Constructs KD-Tree based on passed particles positions"""
    print(f"... Constructing KDtree for {pos.shape[0]} particles")
    from scipy.spatial import KDTree
    return KDTree(pos)


def load_particles(simPath: str, snapNum: int = 600, partType: str = "PartType4", key_list=None) -> None:
    """Load relavent quantities of simulation particle data based on 
       specified particle type and associated keys
    """
    results = dict()

    # from simulation path to snapshot directory, determine if snapshots 
    # are partitioned into integer chunks
    snapdirPath = f"{simPath}/snapdir_{snapNum}"
    num_chunks = len(os.listdir(snapdirPath))
        
    # load in relavent hdf5 quantities through `key_dict`.
    if num_chunks == 1:
        completePath = f"{snapdirPath}/snapshot_{snapNum}.hdf5"
        with h5py.File(completePath, 'r') as h5:
            hubble = h5['Header'].attrs.__getitem__('HubbleParam') 
            h5p = h5[partType]
            if key_dict == None:
                for key in h5p.keys():
                    results[key] = h5p[key][()]
            else:
                for key in key_list:
                    results[key] = h5p[key][()]
        
    elif num_chunks > 1:
        # before looping each chunk, first initilize empty lists for each 
        # key to append particle data to.
        if key_list == None: 
            with h5py.File(f"{snapdirPath}/snapshot_{snapNum}.0.hdf5", 'r') as h5:
                h5p = h5[partType]
                for key in h5p.keys():
                    results[key] = []
        else:
            for key in key_list:
                results[key] = []

        # empty lists initialized, now begin loop.
        for chunk in range(num_chunks):
            completePath = f"{snapdirPath}/snapshot_{snapNum}.{chunk}.hdf5"
            with h5py.File(completePath, 'r') as h5:
                hubble = h5['Header'].attrs.__getitem__('HubbleParam') 
                h5p = h5[partType]
                if key_list == None: 
                    for key in h5p.keys():
                        results[key] = h5p[key][()]
                else:
                    for key in key_list:
                        results[key].append(h5p[key][()])
        
        # loop to concatenate all arrays for key items
        for key, values in results.items():
            results[key] = np.concatenate(results[key])

    # convernt relevant quantities to code units
    for key, values in results.items():
        if key == "Masses":
            results[key] *=  1e10 / hubble # convert to Msol
        elif key == "Coordinates":
            results[key] /= hubble # convert to comoving coordinates

    return results


if __name__ == "__main__":
    pass
