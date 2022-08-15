#!/usr/bin/env python3

import os
import sys
import numpy as np
import h5py

if sys.version_info < (3, 6):
    sys.exit("!!! Please use Python 3.6+ to execute script!!!")

hdf5_keys = ['cosmology:hubble', 
             'cosmology:omega_matter', 
             'cosmology:omega_lambda', 
             'id', 
             'position', 
             'mass.lowres', 
             'mass.vir'];

# import self-made analysis modules
import sample_simulations
import analysis_halos
import analysis_cosmology

# ---------------------------------------------------------------------------

def main() -> None:
    
    haloList = sample_simulations.sample_list
    for sim in haloList:
        print("-" * 75)
        print(f">>> Starting sample collection for {sim}")
        
        # import halo catalog
        haloPath = f"/data17/grenache/aalazar/FIRE/GVB/{sim}/halo/rockstar_dm/hdf5/halo_600.hdf5"
        h5 = analysis_halos.load_halo_catalog(haloPath, hdf5_keys)

        # load cosmology class with specific parameters
        cosmo = analysis_cosmology.CosmologyFunctions(h0=h5['cosmology:hubble'], 
                                                      Om0=h5['cosmology:omega_matter'], 
                                                      Ol0=h5['cosmology:omega_lambda'])
        
        # determine MW-mass target systems and partition subhalo sample
        rlim = 300.0
        mw_system = determine_milky_way_halo(sim_name=sim, h5_file=h5, cosmology=cosmo)
        sub_sample = determine_subhalos(mw_results=mw_system, h5_file=h5, cosmology=cosmo, rlim=rlim)
        
        # save final results
        save_sample_to_hdf5(sim_name=sim, mw_results=mw_system, sub_sample=sub_sample)

        print(">>> Done!")
    print("-" * 75)

# ---------------------------------------------------------------------------

def save_sample_to_hdf5(sim_name: str = None, mw_results=None, sub_sample=None) -> None:
    """Save sample to hdf5 in some directory
    """
    # check if directory exists, else create it
    dir_name = "./results/local_halo_sample-catalog" 
    dir_check = os.path.isdir(dir_name)
    if not dir_check:
        os.makedirs(dir_name)
        print(f"created directory: '{dir_name}'")
    else:
        pass

    # wite results to hdf5 via relative path
    saveName = f"{dir_name}/{sim_name}.hdf5"
    with h5py.File(saveName, 'w') as sh5:
        sh5.create_dataset('host:catalog.index:z=0', data=mw_results['index']) 
        sh5.create_dataset('host:comoving.position:z=0', data=mw_results['comoving.position']) 
        sh5.create_dataset('host:virial.mass:z=0', data=mw_results['virial.mass'])
        sh5.create_dataset('host:virial.radius:z=0', data=mw_results['virial.radius'])
        
        sh5.create_dataset('sub:catalog.index:z=0', data=sub_sample['catalog.index.z=0']) 
        sh5.create_dataset('sub:comoving.position:z=0', data=sub_sample['comoving.position']) 
        sh5.create_dataset('sub:virial.mass:z=0', data=sub_sample['virial.mass'])
        sh5.create_dataset('sub:virial.radius:z=0', data=sub_sample['virial.radius'])

        len_check = sh5['sub:comoving.position:z=0'][:].shape[0] == sh5['sub:catalog.index:z=0'][:].shape[0] \
                    == sh5['sub:virial.radius:z=0'][:].shape[0] 
        print(f"... Sanity check that data is the same shape: {len_check}") 

    return None


def determine_subhalos(mw_results=None, h5_file=None, cosmology=None, rlim: float = 300.0) -> None:
    """Feed MW/M31 halo function `mw_results` to parse out subhalo 
       population of interest that is within sphere of radius `rlim` [physical kpc].
    """
    results = dict()

    print(f"... Selecting halos within {rlim} physical kpc")

    # load MW halo quantities
    h0 = h5_file['cosmology:hubble'] 
    host_pos = mw_results['comoving.position']
    wn0 = mw_results['descending.index:wout.MW'] 

    # load subhalo quantities
    sub_id = h5_file['id'][wn0]
    sub_pos = h5_file['position'][wn0]  # comoving kpc
    sub_mass = h5_file['mass.vir'][wn0] # Msol
    sub_rad = np.fromiter((cosmology.virial_radius(0.0, m) for m in sub_mass), dtype=sub_mass.dtype)

    # compute galactocentric seperation
    rsep = compute_seperation(host_pos, sub_pos)

    # select out subhalos within rlim from
    sub_mask = (rsep < rlim) & (sub_mass > 0.0)
    wsub = np.where(sub_mask)[0]
    print(f'... Number of halos within {rlim} physical kpc: {wsub.shape[0]}')

    results['virial.mass'] = sub_id[wsub]
    results['virial.radius'] = sub_rad[wsub]
    results['comoving.position'] = sub_pos[wsub]
    results['catalog.index.z=0'] = wn0[wsub]
    
    return results


def determine_andromeda_halo(sim_name: str = None, h5_file=None, cosmology=None) -> None:
    """Determine M31-mass halo from isolated and ELVIS simulations.
    """
    results = dict()

    h0 = h5_file['cosmology:hubble'] 
    
    # obtain z = 0 halos and sort to descending mass order
    wn0 = np.argsort(h5_file['mass.vir'])[::-1]
    
    # partition out MW halo (most mass halo in isolated), check if ELVIS
    str_split = sim_name.split('_')
    if 'elvis' in str_split:
        cond1 = h5_file['mass.vir'][wn0] > 0.8e12 
        cond2 = h5_file['mass.lowres'][wn0] == 0.0 
        mass_mask = (cond1) & (cond2)
        
        if(sum(mass_mask[wn0]) == 2): 
            # compute galactocentric seperation
            pos = h5_file['position'][wn0][mass_mask]
            dist = compute_seperation(pos[0], pos[1])

            # if two halos fit mass criteria, simply pick first one 
            host_index = wn0[mass_mask][0]
            w = np.where(host_index == wn0)[0]
            wn0 = np.delete(arr=wn0, obj=w, axis=None) # remove host from subhalo sample
            print(f"... Andromeda halo mass: {h5_file['mass.vir'][host_index]:.3e} Msol")   
            print(f'... Pair seperation: {dist} physical kpc') 
        else:
            # no other simulation behaves otherwise, will leave alone until error arises
            pass 
    else:
        pass

    results['virial.mass'] = h5_file['mass.vir'][host_index]
    results['virial.radius'] = cosmology.virial_radius(0.0, h5_file['mass.vir'][host_index])
    results['comoving.position'] = h5_file['position'][host_index]
    results['index'] = host_index
    results['descending.index:wout.M31'] = wn0 
    
    return results


def determine_milky_way_halo(sim_name: str = None, h5_file=None, cosmology=None) -> None:
    """Determine MW-mass halo from isolated and ELVIS simulations.
    """
    results = dict()

    h0 = h5_file['cosmology:hubble'] 
    
    # obtain z = 0 halos and sort to descending mass order
    wn0 = np.argsort(h5_file['mass.vir'])[::-1]
    
    # partition out MW halo (most mass halo in isolated), check if ELVIS
    str_split = sim_name.split('_')
    if 'elvis' in str_split:
        cond1 = h5_file['mass.vir'][wn0] > 0.8e12 
        cond2 = h5_file['mass.lowres'][wn0] == 0.0 
        mass_mask = (cond1) & (cond2)
        
        if(sum(mass_mask[wn0]) == 2): 
            # compute galactocentric seperation
            pos = h5_file['position'][wn0][mass_mask]
            dist = compute_seperation(pos[0], pos[1])

            # if two halos fit mass criteria, simply pick last one 
            new_sim_name = sim_name
            host_index = wn0[mass_mask][1]
            w = np.where(host_index == wn0)[0]
            wn0 = np.delete(arr=wn0, obj=w, axis=None) # remove host from subhalo sample
            print(f"... Milky-Way halo mass: {h5_file['mass.vir'][host_index]:.3e} Msol")   
            print(f'... Pair seperation: {dist} physical kpc') 
        else:
            # no other simulation behaves otherwise, will leave alone until error arises
            pass 
    else:
        # for isolated halo, most massive would be first index of descending ordered array
        new_sim_name = sim_name
        host_index = wn0[0]                              
        wn0 = np.delete(arr=wn0, obj=0, axis=None) # remove host from entire sample
        print(f"... Milky-Way halo mass: {h5_file['mass.vir'][host_index]:.3e} Msol")

    results['sim.name'] = new_sim_name
    results['virial.mass'] = h5_file['mass.vir'][host_index]
    results['virial.radius'] = cosmology.virial_radius(0.0, h5_file['mass.vir'][host_index])
    results['comoving.position'] = h5_file['position'][host_index]
    results['index'] = host_index
    results['descending.index:wout.MW'] = wn0 
    
    return results


def compute_seperation(pos1: float, pos2: float) -> float:
    """Quick and dirty method for computing position seperation.
       Be very careful with what you are evaluating.
    """
    sep = pos1 - pos2
    if pos1.shape != pos2.shape:
        result = np.sqrt(sep[:, 0]**2 + sep[:, 1]**2 + sep[:, 2]**2) 
    else:
        result = np.sqrt(sep[0]**2 + sep[1]**2 + sep[2]**2)
    return result


if __name__ == "__main__":
    main()
