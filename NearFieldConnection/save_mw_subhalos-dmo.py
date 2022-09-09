#! usr/bin/env python3

# system ----
import os
import sys
import time
import h5py
import numpy as np

start = time.time()

import sample.FB15N2048_DM
import sample.cosmologies
from tools.cosmology import preset_cosmologies

outPath = '/data17/grenache/aalazar/projects/researchproject_008/output'
BOX_SIDE_LENGTH = 15.0 # cMpc/h
G_MSOL = 4.3e-6 

# ---------------------------------------------------------------------------

def main() -> None:

    print('-' * 75)
    print(f"*** Starting analysis ***")
    print('-' * 75)

    # import sample dictionaries
    cosmo_dict = preset_cosmologies.models['planck15']
    dmo_dict = sample.FB15N2048_DM.snapInfo['103']
    box_side_length = BOX_SIDE_LENGTH * 1000.0 / cosmo_dict['h0']  # comoving kpc
    
    """
    # load halo catalog
    print("... importing halo catalog")
    catPath = dmo_dict['halo.catalog.path']
    catalog = load_rockstar_catalog(catalog_path=catPath, cosmo_dict=cosmo_dict)
   
    # partition out MW-mass halo sample
    print("... partitioning out MW-mass halos")
    mw_catalog = sample_isolated_MW_halos(catalog, box_side_length, 
                                         verbose=True)
   
    # construct MW-mass halo mmp trees
    print("... constructing and saving MW halo main progenitor trees")
    save_isolated_MW_mpb(catalog=mw_catalog, cosmo_dict=cosmo_dict, 
                        verbose=True)
    
    # save subhalo sample based on MW-mass halo positions
    print("... saving MW-system subhalo sample")
    save_subhalo_sample(halo_catalog=catalog, mw_catalog=mw_catalog, 
                        box_length=box_side_length, 
                        verbose=True)

    # save subhalo main prog branch to hdf5 
    print("... constructing and saving subhhalo main progenitor trees")
    save_subhalo_mpb(cosmo_dict=cosmo_dict, verbose=True)

    # construct subhalo mass functions
    print("... Constructing subhalo mass functions for MW-mass hosts")
    save_mw_mass_function()
    """
    
    #
    saveref_halo_progenitors()

    print(">>> Done!")
    print("-" * 75)

    total_time = time.time() - start
    print(f"Wall-clock time execution: {total_time:0.3f} sec.")
    print("-" * 75)

    return None

# ---------------------------------------------------------------------------

from tools.cosmology import cosmology_analysis

# -------------------------------------

def saveref_halo_progenitors() -> None:
    """
    """
    results = dict()
    
    # initialize lists
    prg_types = ['maj', 'min']
    z_list = ['z6', 'z7']
    param_list = ['hids', 'labels', 'rad', 'x', 'y', 'z']
    for prg in prg_types:
        for z in z_list:
            for param in param_list:
                results[f'{prg}:{z}:{param}'] = list()
        
    with h5py.File(f"{outPath}/mpb-res_subs.hdf5", 'r') as h5:
        # start by saving MW halo parameters
        mwIDs = h5["halo.id:MW:list"][()]
        for mwID in mwIDs:            
            mwh5 = h5[f"halo.id:{mwID}"]
            tmwh5 = mwh5["tree.quantities"]
            for z in z_list:
                mamihid = tmwh5[f'ma.mi.prg:halo.id:{z}'][()]
                mamirad = tmwh5[f'ma.mi.prg:r.vir:com.h0:{z}'][()]
                mamipos = tmwh5[f'ma.mi.prg:pos:com.h0:{z}'][()]
                for prg in prg_types:
                    if prg == "maj":
                        results[f'{prg}:{z}:hids'].append(mamihid[0])
                        results[f'{prg}:{z}:labels'].append(0)
                        results[f'{prg}:{z}:rad'].append(mamirad[0])
                        results[f'{prg}:{z}:x'].append(mamipos[0, 0])
                        results[f'{prg}:{z}:y'].append(mamipos[0, 1])
                        results[f'{prg}:{z}:z'].append(mamipos[0, 2])
                    if prg == "min":
                        results[f'{prg}:{z}:hids'].append(mamihid[1:])
                        results[f'{prg}:{z}:labels'].append(np.zeros(mamihid[1:].shape[0]))
                        results[f'{prg}:{z}:rad'].append(mamirad[1:])
                        results[f'{prg}:{z}:x'].append(mamipos[1:, 0])
                        results[f'{prg}:{z}:y'].append(mamipos[1:, 1])
                        results[f'{prg}:{z}:z'].append(mamipos[1:, 2])

            # start subhalos
            rsIDs = mwh5['halo.id:subs:list:resolved'][()]
            for sID in rsIDs:
                sh5 = mwh5[f"halo.id:{sID}"]
                lab = sh5['galaxy.label'][()]
                try:
                    # first check if both z6 and z7 values exist
                    for z in z_list:
                        mamihid = sh5[f'ma.mi.prg:halo.id:{z}'][()]
                        mamirad = sh5[f'ma.mi.prg:r.vir:com.h0:{z}'][()]
                        mamipos = sh5[f'ma.mi.prg:pos:com.h0:{z}'][()]
                except KeyError:
                    # ignore those missing halos
                    pass
                else:
                    # redo loop and append values
                    for z in z_list:
                        mamihid = sh5[f'ma.mi.prg:halo.id:{z}'][()]
                        mamirad = sh5[f'ma.mi.prg:r.vir:com.h0:{z}'][()]
                        mamipos = sh5[f'ma.mi.prg:pos:com.h0:{z}'][()]
                        for prg in prg_types:
                            if prg == "maj":
                                results[f'{prg}:{z}:hids'].append(mamihid[0])
                                results[f'{prg}:{z}:labels'].append(lab)
                                results[f'{prg}:{z}:rad'].append(mamirad[0])
                                results[f'{prg}:{z}:x'].append(mamipos[0, 0])
                                results[f'{prg}:{z}:y'].append(mamipos[0, 1])
                                results[f'{prg}:{z}:z'].append(mamipos[0, 2])
                            if prg == "min":
                                results[f'{prg}:{z}:hids'].append(mamihid[1:])
                                labs = np.full(shape=mamihid[1:].shape[0], fill_value=lab)
                                results[f'{prg}:{z}:labels'].append(labs)
                                results[f'{prg}:{z}:rad'].append(mamirad[1:])
                                results[f'{prg}:{z}:x'].append(mamipos[1:, 0])
                                results[f'{prg}:{z}:y'].append(mamipos[1:, 1])
                                results[f'{prg}:{z}:z'].append(mamipos[1:, 2]) 

    for skey, sval in results.items():
        try:
            results[skey] = np.concatenate(sval)
        except ValueError:
            results[skey] = np.array(sval)

    # save major and minor progenitor sample
    fmt = '%i %i %s %s %s %s'
    for z in z_list:
        for prg in prg_types:
            rfstr = f"{prg}:{z}"
            data = np.c_[results[f'{rfstr}:hids'], 
                         results[f'{rfstr}:labels'],
                         results[f'{rfstr}:rad'],
                         results[f'{rfstr}:x'],
                         results[f'{rfstr}:y'],
                         results[f'{rfstr}:z'],
                         ]                        
            saveFile = f"results/subhalo_progs-rockstar/{prg}-{z}.txt"
            np.savetxt(saveFile, data, fmt=fmt)


def label_galaxy_type(vpeak: float) -> int:
    """Peak circualr velocity criterian for galaxy halos"""
    if vpeak > 150.0:  # Milky Way-mass halo
        result = 0
    elif (60.0 < vpeak) & (vpeak < 150.0):  # bright dwarfs
        result = 1
    elif (30.0 < vpeak) & (vpeak < 60.0):  # classical dwarfs
        result = 2
    elif (8.0 < vpeak) & (vpeak < 30.0):  # ultra-faint dwarfs
        result = 3
    else:
        result = -1
    return result

# -------------------------------------

def save_mw_mass_function() -> None:
    """Generates MW-halo mass functions on interestes. 
       | Saves redshift z=0 N(>vpeak) for each MW-mass halo.
       | Saves N(>Mhalo) at redshift z=0 for each MW-mass halo 
       | Saves N(>Mhalo) at redshift Z~7 for the five most-massive 
         progenitors of each MW-mass host.
    """
    
    # load reference files
    try:
        mwIDs = np.loadtxt("results/mw_final_sample-z0.txt", dtype=np.int64)
        asIDs = np.loadtxt('results/all_subs_in_MW.txt', dtype=np.int64)
    except IOError:
        raise IOError("!!! Reference files not available to use, function obselete!!!")
    else:
        pass
   
    # 
    outFile = f"{outPath}/mw_mass_functions.hdf5"
    with h5py.File(outFile, 'w') as h5:
        with h5py.File(f"{outPath}/mpb-res_subs.hdf5", 'r') as th5:
            mwIDs = th5["halo.id:MW:list"][()]
            h5.create_dataset("halo.id:MW:list", data=mwIDs)

            for mwID in mwIDs:
                mwh5 = h5.create_group(f"halo.id:{mwID}")
                
                # load relevant parameters
                mwth5 = th5[f"halo.id:{mwID}"]
                rsIDs = mwth5['halo.id:subs:list:resolved'][()]
                usIDs = mwth5['halo.id:subs:list:unresolved'][()]
               
                # save all vpeak values inMW  halo at z=0
                vh5 = mwh5.create_group('v.peak')
                vpeak = np.array(
                            [mwth5[f"halo.id:{sID}"]['v.peak'][()] for sID in usIDs]
                            )
                vmf = mass_function(vpeak, log_bounds=(np.log10(5.0), np.log10(vpeak.max())))
                for skey, sval in vmf.items():
                    vh5.create_dataset(skey, data=sval)

                # save all mhalo values in MW halo at z=0
                m0h5 = mwh5.create_group('m.halo:z0')
                mhalo_z0 = np.array(
                                   [mwth5[f"halo.id:{sID}"]['m.vir'][()][0] for sID in usIDs]
                                   )
                m0mf = mass_function(mhalo_z0, log_bounds=(6.0, np.log10(mhalo_z0.max())))
                for skey, sval in m0mf.items():
                    m0h5.create_dataset(skey, data=sval)

                # not all halos will have z7 minor progenitors, so will not 
                # use list comprehension in order to catch excpetions
                m7h5 = mwh5.create_group('m.halo:z7')
                mlist = list()
                for sID in usIDs:
                    try:
                        mlist.append(mwth5[f"halo.id:{sID}"]['ma.mi.prg:m.vir:z7'][()])
                    except KeyError:
                        pass
                mhalo_all_z7 = np.concatenate(mlist)
                m7mf = mass_function(mhalo_all_z7, log_bounds=(6.0, np.log10(mhalo_all_z7.max())))
                for skey, sval in m7mf.items():
                    m7h5.create_dataset(skey, data=sval)


def mass_function(mass_catalog: float, volume: float = 1.0, log_bounds: int = None) -> float:
    """Computes the mass functions (or cumulative mass function) 
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
    """
    nonzero_index_1 = np.nonzero(cumul_numb)
    mass_arr = mass_arr[nonzero_index_1] 
    cumul_numb = cumul_numb[nonzero_index_1]
    """

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

# -------------------------------------

def save_subhalo_mpb(cosmo_dict, verbose: bool = True) -> None:
    """Will generate a hdf5 file to save the subhalo heirarchy:
    1) List of MW-mass halo ids as root group
    2) For each MW-mass halo [root group], save list of subhalo ids 
       [sub group]
    3) Withing each MW root grouping, save trees of each resolved subhalos
       [sub group]
    4) Each subhalo halo sub group tree will have relevent tree quantitites within
    """
    # set up import parameters to reference
    treePath = "/data18/brunello/aalazar/FIREbox/FB15N2048_DM/halo/rockstar_dm/mmp/snapdir_103"
 
    # load MW reference file of final sample 
    mwIDs = np.loadtxt("results/mw_final_sample-z0.txt", dtype=np.int64)
    
    if verbose:
        print("> Creating subhalo tree hdf5 for each MW-mass system")

    with h5py.File(f"{outPath}/mpb-res_subs.hdf5", 'w') as h5:
        # save MW id list
        h5.create_dataset("halo.id:MW:list", data=mwIDs)  

        # now iterate through each MW ID        
        for mwID in mwIDs:  
            if verbose:
                print(f"| Starting now with halo {mwID}")
            # create MW ID subgroup
            mwh5 = h5.create_group(f"halo.id:{mwID}")  

            # save MW tree results
            mwth5 = mwh5.create_group(f"tree.quantities")
            treeFile = f"{treePath}/tree_{mwID}.hdf5"
            tree_res = load_mmp_tree(treeFile, mwID, cosmo_dict)        
            for skey, sval in tree_res.items():
                mwth5.create_dataset(skey, data=sval)
            mwth5.create_dataset("v.peak", data=np.max(mwth5['v.max'][()]))
            mwth5.create_dataset('m.peak', data=np.max(mwth5['m.vir'][()]))
            if verbose:
                print(f"+ Saved the MW halo tree")
            
            # load each subhalo ID based on MW halo ID
            usIDs = np.loadtxt(f"results/mw_subhalos-rockstar/halo_{mwID}-unresolved.txt", dtype=np.int64)
            rsIDs = np.loadtxt(f"results/mw_subhalos-rockstar/halo_{mwID}-resolved.txt", dtype=np.int64)
            if verbose:
                print(f"+ Iterating through trees for {usIDs.shape[0]} subhalos")
            
            # list containing indices of halos with constructed trees
            usind = []
            rsind = []
            # open relevent tree file parameters
            for ind, sID in enumerate(usIDs):
                treeFile = f"{treePath}/tree_{sID}.hdf5"
                try:
                    tree_res = load_mmp_tree(treeFile, sID, cosmo_dict)   
                except IOError: 
                    # halo does not have a tree file!
                    pass
                else:
                    # check if halo extends back to z6 and z7
                    cond1 = tree_res["ind:z6"] == -1
                    cond2 = tree_res["ind:z7"] == -1
                    #cond = cond1 & cond2
                    cond = False
                    if cond:
                        pass
                    else:
                        # index of unresolved halo w/ constructed tree
                        usind.append(ind)

                        # create subgroup containing index of resolved halos
                        w = np.where(sID == rsIDs)[0]
                        if w.shape[0] > 0:
                            rsind.append(w)

                        sh5 = mwh5.create_group(f"halo.id:{sID}")
                        # save results to sub group hdf5
                        for skey, sval in tree_res.items():
                            sh5.create_dataset(skey, data=sval) 

                        # save mpeak and vpeak
                        sh5.create_dataset("v.peak", data=np.max(sh5['v.max'][()]))
                        sh5.create_dataset('m.peak', data=np.max(sh5['m.vir'][()]))
                       
                        # assign classification label
                        label = label_galaxy_type(sh5['v.peak'][()])
                        sh5.create_dataset("galaxy.label", data=label)
            
            # save subhalo ID list
            rsind = np.concatenate(rsind)
            mwh5.create_dataset("halo.id:subs:list:resolved", data=rsIDs[rsind])
            mwh5.create_dataset("halo.id:subs:list:unresolved", data=usIDs[usind])
            
            if verbose:
                ushape = mwh5["halo.id:subs:list:unresolved"][()].shape[0]
                print(f"+ Final sample contains {ushape} unresolved halos with trees")
                rshape = mwh5["halo.id:subs:list:resolved"][()].shape[0]
                print(f"+ Final sample contains {rshape} resolved halos with trees")

            if verbose:
                print(f"| Saved MW system")
                print(":" * 50)

# -------------------------------------

def save_subhalo_sample(halo_catalog, mw_catalog, box_length, verbose: bool = True) -> None:
    """
    """
    # generate subhalo dictionary catalog
    scatalog = halo_catalog.copy() 

    # remove MW halos from rest of the sample
    halo_index = np.array([i for i in range(halo_catalog['halo.id'].shape[0])])
    sub_index = np.delete(halo_index, mw_catalog['index'], 0) 
    for skey, sval in scatalog.items():
        scatalog[skey] = sval[sub_index]
    
    res_sub_inMW_list = []
    all_sub_inMW_list = []
    all_sub_outMWinLV_list = []
    all_sub_inLV_list = []
    for hind, hpos in enumerate(mw_catalog['position']):
        sep = hpos - scatalog['position']
        dx = periodic(sep[:, 0], box_length)
        dy = periodic(sep[:, 1], box_length)
        dz = periodic(sep[:, 2], box_length)
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        # condition masks based on distance (physical kpc)
        dist_cond1 = dist < 400.0
        dist_cond2 = dist > 400.0
        dist_cond3 = dist < 1200.0

        # condition masks based on mass and vmax
        mass_cond1 = 1e8 < scatalog['virial.mass']
        mass_cond2 = scatalog['virial.mass'] < 5e9
        vmax_cond1 = 5.0 < scatalog['max.vcirc']
        vmax_cond2 = scatalog['max.vcirc'] < 30.0

        res_sub_in_mw = np.where(dist_cond1 & mass_cond1 & vmax_cond1)[0]
        all_sub_in_mw = np.where(dist_cond1)[0]
        all_sub_in_lgv = np.where(dist_cond3)[0]
        all_sub_in_lgv_out_mw = np.where(dist_cond2 & dist_cond3)[0]
        
        if verbose:
            hid = mw_catalog['halo.id'][hind]
            sm = max(scatalog['virial.mass'][res_sub_in_mw])
            print(f"||| max satellite mass within MW halo {hid}: {sm:.2e}")
        
        res_sub_inMW_list.append(scatalog['halo.id'][res_sub_in_mw])
        all_sub_inMW_list.append(scatalog['halo.id'][all_sub_in_mw])
        all_sub_outMWinLV_list.append(scatalog['halo.id'][all_sub_in_lgv_out_mw])
        all_sub_inLV_list.append(scatalog['halo.id'][all_sub_in_lgv])
        
        # save resolved subhalo ids for each MW in results folder
        np.savetxt(f"results/mw_subhalos-rockstar/halo_{mw_catalog['halo.id'][hind]}-resolved.txt",
                   scatalog['halo.id'][res_sub_in_mw], fmt='%s')
        np.savetxt(f"results/mw_subhalos-rockstar/halo_{mw_catalog['halo.id'][hind]}-unresolved.txt",
                   scatalog['halo.id'][all_sub_in_mw], fmt='%s')
        if verbose:
            hid = mw_catalog['halo.id'][hind]
            print(f"::: Saved resolved and unresolved subhalo IDs for host {hid} in `results` folder")

    def uc_list(list_):
        return np.unique(np.concatenate(list_), axis=0)
    
    np.savetxt('results/res_subs_in_MW.txt', uc_list(res_sub_inMW_list), fmt='%s')
    np.savetxt('results/all_subs_in_MW.txt', uc_list(all_sub_inMW_list), fmt='%s')
    np.savetxt('results/all_subs_out_MW_in_LGV.txt', uc_list(all_sub_outMWinLV_list), fmt='%s')
    np.savetxt('results/all_subs_in_LGV.txt', uc_list(all_sub_inLV_list), fmt='%s')

# -------------------------------------

def save_isolated_MW_mpb(catalog, cosmo_dict, verbose: bool = True) -> None:
    """
    """
    # set up import parameters to reference
    treePath = '/data18/brunello/aalazar/FIREbox/FB15N2048_DM/halo/rockstar_dm/mmp/snapdir_103'

    # will want to save seperate file in results to reference later...
    hid_0, hid_6, hid_7 = [], [], []
    mpeak, vpeak = [], []
    with h5py.File(f'{outPath}/mpb-isolated_MW.hdf5', 'w') as h5:
        for hid in catalog['halo.id']: 
            # mmp tree file name
            treeFile = f"{treePath}/tree_{hid}.hdf5"

            # open relevent tree file parameters
            tree_res = load_mmp_tree(treeFile, hid, cosmo_dict)        
                
            # append list for most massive progenitor of MW
            hid_0.append(hid)
            mpeak.append(max(tree_res['m.vir']))
            vpeak.append(max(tree_res['v.max']))
            if tree_res['ind:z6'] == -1:
                hid_6.append(-1)
            else:
                hid_6.append(int(tree_res['rockstar.id'][tree_res['ind:z6']]))
            if tree_res['ind:z7'] == -1:
                hid_7.append(-1)
            else:
                hid_7.append(int(tree_res['rockstar.id'][tree_res['ind:z7']]))
                   
            # save information to hdf5
            h5h = h5.create_group(f'tree.sample/id:{hid}')
            for skey, sval in tree_res.items():
                h5h.create_dataset(skey, data=sval)

        # save z=0 rockstar ids to hdf5 to easily call
        h5.create_dataset('halo.id:list', data=np.array(hid_0))

    if verbose:
        print("::: Saved mmp trees to hdf5")
    
    # save resulting most massive progenitors of MW halos 
    data = np.c_[[int(i) for i in hid_0], 
                 [int(i) for i in hid_6], 
                 [int(i) for i in hid_7], 
                 vpeak, mpeak]
    np.savetxt('./results/mw_final_sample-mmp.txt', data, fmt='%i %i %i %s %s')
    if verbose:
        print("::: Saved MW mmp reference file to `results` folder")

# -------------------------------------

def sample_isolated_MW_halos(halo_catalog, box_length, verbose: bool = False):
    """
    """
    results = dict()

    # parition out MW-mass halos within this mass range:
    mass_range = (8.0e+11, 2.4e+12)
    
    # index positiona of these MW-mass halos, which contains objects
    # isolated and near massive neighbors
    cond1 = mass_range[0] < halo_catalog['virial.mass']
    cond2 = halo_catalog['virial.mass'] < mass_range[1]
    wh1 = np.where(cond1 & cond2)[0] 
    if verbose:
        print(f"| Number of MWs within set mass range: {wh1.shape[0]}")

    # now remove MW-mass halos with mass neighbors
    rlim1 = 2000.0  # 2 physical Mpc 
    wh2 = []    # new MW-mass halo indices
    for hind in wh1:
        # define quantities of target halo
        target_mass = halo_catalog['virial.mass'][hind]
        target_pos = halo_catalog['position'][hind]

        # define quantities of more massive halos
        cond1 = halo_catalog['virial.mass'] > target_mass
        wm = np.where(cond1)[0]
        massive_mass = halo_catalog['virial.mass'][wm]
        massive_pos = halo_catalog['position'][wm]

        # compute physical seperation between objects
        sep = target_pos - massive_pos[:] 
        dx = periodic(sep[0], box_length)
        dy = periodic(sep[1], box_length)
        dz = periodic(sep[2], box_length)
        dist = np.sqrt(dx**2 + dy**2 + dx**2)

        # remove halos with massive neighbors within rlim1
        cond2 = any(dist < rlim1)
        if cond2:
            pass
        else:
            wh2.append(hind)    
    wh2 = np.array(wh2)
    if verbose:
        print(f"| Number of MWs without a massive neighbor within {rlim1} kpc: {wh2.shape[0]}")

    # now remove MW-mass halos with major mergers!
    rlim2 = 600
    wh3 = []
    for i, ind in enumerate(wh2):
        target_pos = halo_catalog['position'][hind]
         
        # parition out MW halos from position sample
        other_i = np.delete(wh2, i, 0); 
        other_pos = halo_catalog['position'][other_i]

        # compute physical seperation between objects
        sep = target_pos - other_pos[:] 
        dx = periodic(sep[0], box_length)
        dy = periodic(sep[1], box_length)
        dz = periodic(sep[2], box_length)
        dist = np.sqrt(dx**2 + dy**2 + dx**2)

        cond1 = any(dist < rlim2)
        if cond1:
            pass
        else:
            wh3.append(ind)
    wh3 = np.array(wh3) 
    if verbose:
        print(f"| Number of MWs without a major merger: {wh3.shape[0]}")
   
    # save final sample quantities of isolated MW-mass halos
    wf = wh3
    results['index'] = wf
    for skey, sval in halo_catalog.items():
        results[skey] = sval[wf]
    
    if verbose:
        print("::: Saved MW sample reference file to `results` folder")
    np.savetxt('./results/mw_final_sample-z0.txt', results['halo.id'], fmt='%s')

    return results

# -------------------------------------

def load_mmp_tree(treePath: str, halo_id: int, cosmo_dict) -> None:
    """Loads post-process most massive progenitor tree given a rockstar halo id"""
    results = dict()

    # set up import parameters to reference
    setCosmo = cosmology_analysis.SetCosmology(**cosmo_dict)
    comp_rvir = setCosmo.virial_radius
    conv = 1.0 - cosmo_dict['Ob0']/cosmo_dict['Om0']

    # check if halo file exist 
    if not os.path.isfile(treePath):
        raise IOError(f"missing {halo_id}")
    else:
        # open hdf5 saved mpb tree
        with h5py.File(treePath, 'r') as th5:

            # load mpb quantitied
            snap_h5 = th5['Snap_idx'][()]
            hid_h5 = th5['Orig_halo_ID'][()]
            scale_h5 = th5['scale'][()] 
            redshift_h5 = 1.0/scale_h5 - 1.0
            mvir_h5 = th5['Mvir'][()] * conv 
            rvir_h5 = th5['Rvir'][()]
            vmax_h5 = th5['vmax'][()] * np.sqrt(conv);
            rmax_h5 = th5['rmax'][()]
            x_h5 = th5['x'][()]
            y_h5 = th5['y'][()]  
            z_h5 = th5['z'][()]
            vx_h5 = th5['vx'][()]
            vy_h5 = th5['vy'][()] 
            vz_h5 = th5['vz'][()]
            rvir_h5_fixed = np.fromiter( # physical kpc
                            (comp_rvir(z, m) for z, m in zip(redshift_h5, mvir_h5/cosmo_dict['h0'])),
                            dtype=np.float64)
            rvir_h5_fixed *= cosmo_dict['h0']/scale_h5  
            
            # remove duplicate scale factors and switch to ascending order
            results['scale.factor'] = np.unique(scale_h5, axis=0)[::-1]
            nsize = results['scale.factor'].shape[0]
            # prepare new arrays to save
            results['m.vir'] = np.zeros(nsize)
            results['r.vir:phy'] = np.zeros(nsize)
            results['r.vir:com.h0'] = np.zeros(nsize)
            results['v.max'] = np.zeros(nsize)
            results['snapshot.number'] = np.zeros(nsize)
            results['rockstar.id'] = np.zeros(nsize)
            results['pos:phy'] = np.zeros((nsize, 3))
            results['pos:com.h0'] = np.zeros((nsize, 3))
            results['vel'] = np.zeros((nsize, 3))

            # default index factor to apply to retrieve quantities at 
            # some redshift
            results['ind:z6'], results['ind:z7'], results['ind:z8'] = -1, -1, -1
            results['ind:z9'], results['ind:z10'] = -1, -1
            
            # loop through each forest at specific redshift (or scale factor)
            for i, a in enumerate(results['scale.factor']):
                z = 1.0/a - 1.0
                # quantity locations of matching redshift
                wa = np.where(a == scale_h5)[0]
                amass = mvir_h5[wa] / cosmo_dict['h0']; 
                arvir = rvir_h5_fixed[wa]
                avmax = vmax_h5[wa]; 
                armax = rmax_h5[wa]
                asnap = snap_h5[wa]
                ahid = hid_h5[wa]
                ax, ay, az = x_h5[:][wa], y_h5[:][wa], z_h5[:][wa]
                avx, avy, avz = vx_h5[:][wa], vy_h5[:][wa], vz_h5[:][wa];
                
                # search random forest for most massive progenitor, make 
                # exception if one halo in forest exists
                if amass.shape[0] == 1: 
                    # only one?
                    results['m.vir'][i] += amass[0]
                    results['r.vir:com.h0'][i] += arvir[0]  
                    results['r.vir:phy'][i] += arvir[0] * a / cosmo_dict['h0']
                    results['v.max'][i] += avmax[0]
                    results['snapshot.number'][i] += int(asnap[0])
                    results['rockstar.id'][i] += int(ahid[0]) 
                    results['pos:com.h0'][i] += np.array([ax[0], ay[0], az[0]])
                    results['pos:phy'][i] += results['pos:com.h0'][i] * a / cosmo_dict['h0']
                    results['vel'][i] += np.array([avx[0], avy[0], avz[0]])

                else:
                    # sample out most massive halo
                    wm = np.where(amass == max(amass))[0]
                    results['m.vir'][i] += amass[wm][0]
                    results['r.vir:phy'][i] += arvir[wm][0] * a / cosmo_dict['h0']
                    results['r.vir:com.h0'][i] += arvir[wm][0]
                    results['v.max'][i] += avmax[wm][0]
                    results['snapshot.number'][i] += int(asnap[wm][0])
                    results['rockstar.id'][i] += int(ahid[wm][0])
                    results['pos:com.h0'][i] += np.array([ax[wm][0], ay[wm][0], az[wm][0]])
                    results['pos:phy'][i] += results['pos:com.h0'][i] * a / cosmo_dict['h0']
                    results['vel'][i] += np.array([avx[wm][0], avy[wm][0], avz[wm][0]])

                    # check if progenitor extends down to this redshift/snapshot
                    if results['snapshot.number'][i] == 29:
                        results['ind:z6'] = i
                        # save 5 most masssive progenitors
                        wap = np.argsort(amass)[::-1][:5]  # mass in descending order
                        results['ma.mi.prg:halo.id:z6'] = ahid[wap].astype(np.int64)
                        results['ma.mi.prg:m.vir:z6'] = amass[wap]
                        results['ma.mi.prg:r.vir:phy:z6'] = arvir[wap] * a / cosmo_dict['h0']
                        results['ma.mi.prg:r.vir:com.h0:z6'] = arvir[wap]
                        pos = np.vstack((ax[wap], ay[wap], az[wap])).T
                        results['ma.mi.prg:pos:com.h0:z6'] = pos
                        results['ma.mi.prg:pos:phy:z6'] = pos * a / cosmo_dict['h0']
                    if results['snapshot.number'][i] == 22:
                        results['ind:z7'] = i
                        # save 5 most masssive progenitors
                        wap = np.argsort(amass)[::-1][:5]  # mass in descending order
                        results['ma.mi.prg:halo.id:z7'] = ahid[wap].astype(np.int64)
                        results['ma.mi.prg:m.vir:z7'] = amass[wap]
                        results['ma.mi.prg:r.vir:phy:z7'] = arvir[wap] * a / cosmo_dict['h0']
                        results['ma.mi.prg:r.vir:com.h0:z7'] = arvir[wap]
                        pos = np.vstack((ax[wap], ay[wap], az[wap])).T
                        results['ma.mi.prg:pos:com.h0:z7'] = pos
                        results['ma.mi.prg:pos:phy:z7'] = pos * a / cosmo_dict['h0']
                    if results['snapshot.number'][i] == 17:
                        results['ind:z8'] = i
                    if results['snapshot.number'][i] == 14:
                        results['ind:z9'] = i
                    if results['snapshot.number'][i] == 11:
                        results['ind:z10'] = i
            
            # create resolution flag array
            cond1 = results['v.max'] > 5.0
            cond2 = results['m.vir'] > 1e7
            res_arr = np.zeros(nsize) 
            res_arr[cond1 & cond2] = 1
            results['resolved.flag'] = res_arr

    return results

# -------------------------------------

def load_rockstar_catalog(catalog_path: str, cosmo_dict, 
                          conv_dmo_to_hyd: bool = True, 
                          conv_units: bool = True,
                          *args, **kwargs) -> None:
    """
    """
    results = dict()
   
    # if dmo results and comparing with hydro analogs, 
    # conversion factor takes difference based on baryon
    # fraction of the universe.
    conv = 1.0
    if conv_dmo_to_hyd:
        conv -= cosmo_dict['Ob0'] / cosmo_dict['Om0']
    
    # load halo id seperatly to save uint64 data types
    results['halo.id'] = np.loadtxt(catalog_path, comments='#', usecols=0, dtype=np.uint64)
    
    # load in other halo quantities
    catalog = np.loadtxt(catalog_path, comments='#')
    results['numb.parts'] = catalog[:, 7]
    results['virial.mass'] = catalog[:, 2] * conv
    results['virial.radius'] = catalog[:, 5]
    results['max.vcirc'] = catalog[:, 3] * np.sqrt(conv)
    results['max.radius'] = catalog[:, 18]
    results['position'] = np.vstack(
                            (catalog[:, 8], 
                             catalog[:, 9],
                             catalog[:, 10])
                            ).T
    
    # convert quantities to appropriate units
    if conv_units:
        results['virial.mass'] /= cosmo_dict['h0'] 
        results['virial.radius'] /= cosmo_dict['h0']
        results['max.radius'] /= cosmo_dict['h0']
        results['position'] *= 1000.0 / cosmo_dict['h0']

    return results

# -------------------------------------

def periodic(sep: float, box_side_length: float) -> float:
    """Transform halo coordinates based box periodic conditions"""
    sep[sep < -box_side_length/2.0] += box_side_length
    sep[sep > box_side_length/2.0] -= box_side_length
    return sep

# -------------------------------------

if __name__ == "__main__":
    main()
