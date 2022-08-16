#! usr/bin/env python3

# system ----
import os
import sys
import h5py
import time
import numpy as np

if sys.version_info < (3, 6):
    sys.exit("!!! Please use Python 3.6+ to execute script!!!")

# local ----
simPath='/data18/brunello/aalazar/FIRE'
outPath='/data17/grenache/aalazar/projects/researchproject_010/output'
sys.path.append("/export/home/aalazar/code/")

# define key scale factors for the different key redshifts to analyze at.
a2 = 1.0 / (1.0 + 2.0) 
a3 = 1.0 / (1.0 + 3.0) 
a4 = 1.0 / (1.0 + 4.0) 
a5 = 1.0 / (1.0 + 5.0) 
a6 = 1.0 / (1.0 + 6.0)
a7 = 1.0 / (1.0 + 7.113)
a8 = 1.0 / (1.0 + 8.130)

# define relevant halo catalog keys to extract
halo_keys = ['cosmology:hubble', 
             'cosmology:omega_matter', 
             'cosmology:omega_lambda', 
             'id', 
             'position', 
             'mass.lowres', 
             'mass.vir'];

# define relevant star catalog keys to extract
star_keys = ['star.indices'];

# import self-made functions
from sample import simulations
from tools import particles
from tools import halos
from tools import lss
    
# Grab Currrent Time Before Running the Code
start = time.time()

# ---------------------------------------------------------------------------

def main() -> None:
    
    print("-" * 75)
    print("*** Beginning analysis ***")
    print("-" * 75)
    
    # will change this if including M31 analog samples...
    haloList = simulations.sample_list

    # initialize hdf5 file
    resultFile = f"{outPath}/smf-archaeology.hdf5"
    with h5py.File(resultFile, "w") as h5:
        for sim in haloList:
            sh5 = h5.create_group(sim)
            for cstr, _ in star_sample_criteria().items():
                sh5.create_group(cstr)

    # analysis for each simulation halo starts here
    for sim in haloList:
        print(f">>> Starting analysis for {sim}")

        # first import the post-processed subhalo sample
        samplePath = f"results/local_halo_sample-catalog/{sim}.hdf5"
        print("... Loading subhalo sample")
        sub_sample = halos.load_subhalo_sample(samplePath)

        # load in original halo catalog and complimentary star catalogs
        print("... Loading halo and star catalog")
        haloPath = f"/data17/grenache/aalazar/FIRE/GVB/{sim}/halo/rockstar_dm/hdf5/halo_600.hdf5"
        starPath = f"/data17/grenache/aalazar/FIRE/GVB/{sim}/halo/rockstar_dm/hdf5/star_600.hdf5"
        h5_halo = halos.load_halo_catalog(haloPath, halo_keys)
        h5_star = halos.load_star_catalog(starPath, star_keys)

        # load in simulation star particles
        print("... Loading star particles")
        simPath = f"/data17/grenache/aalazar/FIRE/GVB/{sim}/output/hdf5"
        partType4_keys = ['Masses', 'Coordinates', 'StellarFormationTime']
        h5p_star = particles.load_particles(simPath, key_list=partType4_keys)

        # construct KDtree, partition out subhalo stars, and perform age computation
        archeo_results = perform_archaeology(h5p_star, sub_sample, 
                                             criteria_dict=star_sample_criteria())
       
        # compute stellar mass functions and save results into master hdf5 file
        print("... Computing stellar mass functions")
        alist = ['a0', 'a6', 'a7', 'a8']
        for cstr, _ in star_sample_criteria().items():
            with h5py.File(resultFile, "a") as h5:
                sh5 = h5[sim][cstr]
                for astr in alist:
                    ah5 = sh5.create_group(astr)
                    smf = lss.compute_mass_function(archeo_results[cstr][astr], 
                                                             volume=1.0, 
                                                             log_bounds=(4, 8))
                    for smf_str, smf_val in smf.items():
                        ah5.create_dataset(smf_str, data=smf_val)

        print(">>> Done!")
        print("-" * 75)
        print("*** Results saved to HDF5 ***")
        print("-" * 75)

    #Subtract start time from the end time
    total_time = time.time() - start
    print(f"Wall-clock time execution: {total_time:0.3f} sec.")
    print("-" * 75)

# ---------------------------------------------------------------------------

def perform_archaeology(partType4, sub_sample, criteria_dict) -> float:
    """Computes the stellar mass based on cumulative stellar ages 
       of z=0 halos based on selection criteria (this would be the
       stars within fractional values of the subhalo virial radius and 
       bound particles).
    """
    results = dict()
   
    # construct kdtree of star particles
    kdtree = particles.construct_kdtree(partType4['Coordinates'])

    # set subhalo sample size and quantities
    subhalo_sample_size = sub_sample['sub:catalog.index:z=0'].shape[0]
    subhalo_position = sub_sample['sub:comoving.position:z=0']
    subhalo_radius = sub_sample['sub:virial.radius:z=0']
        
    print(f"... Performing stellar excavation of {subhalo_sample_size} halos")
    # for each criteria, loop through each halo and compute the stars 
    # within specific criteria conditios and save to set numpy arrays
    for cstr, carg in criteria_dict.items():
        
        mstar_dict = dict()
        mstar_dict['a0'] = np.zeros(subhalo_sample_size)
        mstar_dict['a2'] = np.zeros(subhalo_sample_size)
        mstar_dict['a3'] = np.zeros(subhalo_sample_size)
        mstar_dict['a4'] = np.zeros(subhalo_sample_size)
        mstar_dict['a5'] = np.zeros(subhalo_sample_size)
        mstar_dict['a6'] = np.zeros(subhalo_sample_size)
        mstar_dict['a7'] = np.zeros(subhalo_sample_size)
        mstar_dict['a8'] = np.zeros(subhalo_sample_size)
        
        # kdtree iteration
        for ind, (hpos, hrad) in enumerate(zip(subhalo_position, subhalo_radius)):
 
            # compute mass with star ages greater than zref (\propto 1/aref)
            particle_index = particles.in_halo_via_kdtree(kdtree, hpos, carg*hrad)
            mass = partType4['Masses'][particle_index]
            age = partType4['StellarFormationTime'][particle_index]
            mstar_dict['a0'][ind] += np.sum(mass)
            mstar_dict['a2'][ind] += np.sum(mass[age < a2])
            mstar_dict['a3'][ind] += np.sum(mass[age < a3])
            mstar_dict['a4'][ind] += np.sum(mass[age < a4])
            mstar_dict['a5'][ind] += np.sum(mass[age < a5])
            mstar_dict['a6'][ind] += np.sum(mass[age < a6])
            mstar_dict['a7'][ind] += np.sum(mass[age < a7])
            mstar_dict['a8'][ind] += np.sum(mass[age < a8])
        
        # save criteria results to main results dictionary
        results[cstr] = mstar_dict

    # remove kdtree address for memory purposes
    del kdtree 

    return results


def star_sample_criteria() -> None:
    criteria = dict()
    #criteria['bounds'] = True
    criteria['0.1r'] = 0.1
    criteria['0.2r'] = 0.2
    criteria['0.5r'] = 0.5
    criteria['1.0r'] = 1.0
    return criteria


if __name__ == "__main__": 
    main()
