#!/usr/bin/env python2

# system ----
import os
import sys
import h5py
import time
import numpy as np
from scipy.interpolate import interp1d

import warnings
warnings.filterwarnings("ignore")

# local ----

# ... Process of loading snapshot and halo catalogs at each different redshift.
snapNum = str(int(sys.argv[1]))
log_mass_bounds = (int(sys.argv[2]), int(sys.argv[3]))    

# import simulation sample parameters
from sample import cosmologies
from sample.FB15N2048 import snapInfo

from tools.cosmology import preset_cosmologies

# ---------------------------------------------------------------------------


def main() -> None:
    
    print('-' * 75)
    print(f"*** Starting analysis for snapshot {snapNum} ***")
    print('-' * 75)

    start = time.time()

    global log_mass_bounds

    # load in important parameters (e.g. quantities, cosmology, paths...)
    cosmo_param = preset_cosmologies.models['planck15']
    sim_dict = snapInfo[snapNum]
    scale_factor = sim_dict['scale.factor']
    saveName = sim_dict['save.name'] + f"{log_mass_bounds[0]}-{log_mass_bounds[1]}"
    catalogPath = sim_dict['halo.catalog.path']
    partPath = sim_dict['halo.particle.path']

    # load in halo catalog and apply post-processing functions
    from tools.load_data import load_halo
    unproc_catalog = load_halo.load_ahf_catalog(catalogPath, cosmo_box=True, field_halo_only=True)
    proc_catalog = load_halo.convert_factors(unproc_catalog, cosmo_param['h0'], scale_factor) 

    # partition out halo masses
    if log_mass_bounds[0] == 7:
        log_mass_bounds = (np.log10(4e7), log_mass_bounds[1])
    if log_mass_bounds[1] == 11:
        log_mass_bounds = (log_mass_bounds[0], np.log10(3e11))

    mass_bounds = (10.0**log_mass_bounds[0], 10.0**log_mass_bounds[1])
    halo_catalog = load_halo.sample_from_mass_range(proc_catalog, mass_bounds[0], mass_bounds[1])
    del unproc_catalog; del proc_catalog
    print(f"... Halo virial masses within {mass_bounds[0]:0.2e} to {mass_bounds[1]:0.2e} Msol ")
    print(f"... Sample contains {halo_catalog['halo.id'].shape[0]} halos ")

    # perform analysis halo properties
    print("... Starting profile analysis")
    kwargs = {
            "halo_dict": halo_catalog, "cosmo_dict": cosmo_param, "scale_factor": scale_factor,
            "partPath": partPath
            }       
    results = start_analysis(**kwargs, verbose=True)

    print(">>> Done!")
    print("-" * 75)

    # save results to hdf5
    outPath = f"/data17/grenache/aalazar/projects/researchproject_007/output/field-massbins/{saveName}.hdf5"
    with h5py.File(outPath, 'w') as h5:
        for skey, sval in results.items():
            h5[skey] = sval

    print("*** Results saved to HDF5 ***")
    print("-" * 75)

    # subtract start time from the end time
    total_time = time.time() - start
    print(f"Wall-clock time execution: {total_time:0.3f} sec.")
    print("-" * 75)

# ---------------------------------------------------------------------------

from tools.cosmology import cosmology_analysis

from tools.profiles import analytical
from tools.profiles import deprojected
from tools.profiles import projected

from tools.geometry import rotations
from tools.geometry import shapes

def start_analysis(halo_dict, cosmo_dict, scale_factor: float, 
                   partPath: str, verbose: bool = False, 
                   *args, **kwargs) -> None:
    """Main analysis done with the halo catalogs, snapshots, and cosmology. 
       Halo virial mass and virial radius are recomputed using all bound and 
       unbound particles. Moreover, The deprojected and projected profile are 
       constructed here and are fitted with analytical profiles.
    """
    results = dict()

    nsample = halo_dict['halo.id'].shape[0]
    profile_bins = 25
   
    # initialize deprojected quantities
    results['mass.vir'] = np.zeros(nsample)
    results['mass.200c'] = np.zeros(nsample)
    results['rad.vir'] = np.zeros(nsample)
    results['rad.200c'] = np.zeros(nsample)
    results['rad.pow'] = np.zeros(nsample)
    results['nfw:rho.s'] = np.zeros(nsample)
    results['nfw:r.s'] = np.zeros(nsample)
    results['nfw:c.vir'] = np.zeros(nsample)
    results['nfw:c.200c'] = np.zeros(nsample)
    results['ein:rho.2'] = np.zeros(nsample)
    results['ein:r.2'] = np.zeros(nsample)
    results['ein:c.vir'] = np.zeros(nsample)
    results['ein:c.200c'] = np.zeros(nsample)
    results['ein:alpha'] = np.zeros(nsample)
    #results['deproj:radius'] = np.zeros((nsample, profile_bins))
    #results['deproj:density'] = np.zeros((nsample, profile_bins))
    
    # initialize projected quantities
    axs_list = [str(n) for n in range(5)]
    for axs in axs_list:
        results[f'E.1:vir:{axs}'] = np.zeros(nsample)
        results[f'R.1:vir:{axs}'] = np.zeros(nsample)
        results[f'C.1:vir:{axs}'] = np.zeros(nsample)
        results[f'M.1kpc:vir:{axs}'] = np.zeros(nsample)
        results[f'M.enc:vir:{axs}'] = np.zeros(nsample)
        results[f'E.1:200c:{axs}'] = np.zeros(nsample)
        results[f'R.1:200c:{axs}'] = np.zeros(nsample)
        results[f'C.1:200c:{axs}'] = np.zeros(nsample)
        results[f'M.1kpc:200c:{axs}'] = np.zeros(nsample)
        results[f'M.enc:200c:{axs}'] = np.zeros(nsample)
    
    # actual analysis starts here
    successful_ind = []
    for ind, hid in enumerate(halo_dict['halo.id'][:]): 
        if verbose:
            print(":" * 50)
            print(f'::: Analyzing halo {hid} :::')
        # load particle data
        part_dict = dict()
        haloPartPath = f"{partPath}/halo_{hid}.hdf5"
        with h5py.File(haloPartPath, 'r') as h5:
            p1 = h5['PartType1']
            part_dict['Masses'] = p1['Masses'][:] * 1e10 / cosmo_dict['h0']
            part_dict['Coordinates'] = p1['Coordinates'][:] * scale_factor / cosmo_dict['h0']
        try:   
            kwargs = {
                    'halo_index': ind, 'halo_dict': halo_dict, 
                    'particle_dict': part_dict, 'cosmo_dict': cosmo_dict, 
                    'scale_factor': scale_factor, 'profile_bins': profile_bins
                    }
            deproj = compute_deprojected_properties(**kwargs, verbose=verbose)
        except RuntimeError:
            pass
        except ValueError:
            pass
        else:
            # once deprojection profiles are successfuly computed, start analysis 
            # for projected profiles
            try:
                # internal exception catching is implimented for random projections,
                # this is to really catch ValueError exceptions from the density 
                # axis projection, which just tells that these halos cannot 
                # be fit well
                kwargs = {
                        'deproj_dict': deproj, 'cosmo_dict': cosmo_dict, 
                        'scale_factor': scale_factor, 'profile_bins': profile_bins
                        }
                proj = compute_projected_properties(**kwargs, verbose=verbose)
            except ValueError:
                if verbose:
                    print("!!! Cannot be fitted to, not counted in analysis !!!")
                pass
            else:
                # If no errors when fitting, mark this as a successful result
                # and save projected quantitie
                successful_ind.append(ind)
                for skey, sval in deproj.items():
                    try:
                        results[skey][ind] = sval
                    except KeyError:
                        pass 
    
    # loop over to return successive iterations 
    print(f" Final sample contains {len(successful_ind)} halos")
    for skey, sval in results.items():
        try:
            results[skey] = sval[successful_ind]
        except KeyError:
            pass
    return results


def compute_projected_properties(deproj_dict, cosmo_dict, scale_factor: float, 
                                 profile_bins: int = 25, verbose: bool = False, 
                                 *args, **kwargs) -> None:
    """Analysis here constructs projected halo profile from the deprojected 
       results. Profiles are constructed and fitted in random projections and 
       alongs the three density axii.
    """
    results = deproj_dict.copy()
    
    # obtain depth criteria
    xi = 1.5
    cond1 = (results['part.sep:mag'] < xi*results['rad.vir'])
    cond2 = (results['part.sep:mag'] < xi*results['rad.200c'])
    dw1 = np.where(cond1)[0]
    dw2 = np.where(cond2)[0]

    if verbose:
        print("...Computing projected properties")

    # pass halo-centered particle coordinates through 3D rotation class
    rot_coords_vir = rotations.CoordinateRotation3D(results['part.sep'][dw1])
    rot_coords_200 = rotations.CoordinateRotation3D(results['part.sep'][dw2])
    
    # performing 5 random coordinate rotations of a single halo
    n_rotation_success = 0
    n_rotation_reattempts = 0
    n_rotations = 5
    while(n_rotation_success < n_rotations):
        try:
            # evaluate for virial definition
            kwargs1 = {
                    'result_dict': results, 
                    'rot_coords_class': rot_coords_vir, 
                    'part_mass': results['part.mass'][dw1], 
                    'profile_bins': profile_bins,
                    'save_key': str(n_rotation_success)
                    }
            eval_along_axis(**kwargs1, halo_def='vir') 
            # evaluate for 200c definition
            kwargs2 = {
                    'result_dict': results, 
                    'rot_coords_class': rot_coords_200, 
                    'part_mass': results['part.mass'][dw2], 
                    'profile_bins': profile_bins,
                    'save_key': str(n_rotation_success)
                    }
            eval_along_axis(**kwargs2, halo_def='200c') 
        except ValueError:
            n_rotation_reattempts += 1 
        else:            
            if verbose:
                mv = results[f'M.enc:vir:{n_rotation_success}'] 
                cv = results[f'C.1:vir:{n_rotation_success}'] 
                print(f"| Rand rotation {n_rotation_success} Menc and C [vir]: {mv:0.3e} and {cv:0.3f}")
                m2 = results[f'M.enc:200c:{n_rotation_success}'] 
                c2 = results[f'C.1:200c:{n_rotation_success}'] 
                print(f"| Rand rotation {n_rotation_success} Menc and C [200c]: {m2:0.3e} and {c2:0.3f}")
            n_rotation_success += 1  
    if verbose:
        print(f"Random rotation re-attempts: {n_rotation_reattempts}")
            
    return results


def eval_along_axis(result_dict, rot_coords_class, part_mass: float, 
                    profile_bins: int = 25, save_key: str = None, 
                    halo_def: str = 'vir', axis: float = None, 
                    *args, **kwargs) -> None:
    """Compute the surface density profile along a axis orientation or along
        a random projected of the halo particles. Must pass `rot_coord` class.
    """

    if axis is None:
        nz_pos = rot_coords_class.random_rotation()
    else:
        # transfrom coordinates for z-axis to align with passed `axis`
        nz_pos = rot_coords_class.rotation_new_z_axis(axis) 
 
    # compute projected profile with rotated coordinates and interpolate 
    # within desired radial range (rpower, rvir), then save results to dictionary
    lrmin = np.log10(0.75*result_dict['rad.pow']) 
    lrmax = np.log10(1.5*result_dict[f'rad.{halo_def}']) 
    kwargs = {
            'part_sep': nz_pos, 'part_mass': part_mass,
            'log_bounds': (lrmin, lrmax), 'nbins': 35, 'clean_up': True
            }
    surf_dens = projected.compute_surface_profiles(**kwargs)
    rad_arr = 10.0 ** np.linspace(np.log10(result_dict['rad.pow']), 
                                  np.log10(result_dict[f'rad.{halo_def}']), profile_bins)
    dens_arr = interpolate_array(surf_dens['density.bins:local'], 
                                 surf_dens['radial.bins'], 
                                 rad_arr)
    #result_dict['proj:radius:{save_key}'] = rad_arr
    #result_dict['proj:density:{save_key}'] = dens_arr

    kwargs = {'rad_arr': rad_arr, 'sigma_arr': dens_arr}
    params_laz = analytical.Lazar().curve_fit(**kwargs, set_beta_03=True)
    result_dict[f'E.1:{halo_def}:{save_key}'] = params_laz['sigma1']
    result_dict[f'R.1:{halo_def}:{save_key}'] = params_laz['R1']
    result_dict[f'C.1:{halo_def}:{save_key}'] = result_dict[f'rad.{halo_def}'] / params_laz['R1']
    
    # Interpolate enclosed mass profile and interpolate out 1 kpc
    result_dict[f'M.1kpc:{halo_def}:{save_key}'] = interpolate_array(surf_dens['mass.bins:enclosed'], 
                                                          surf_dens['radial.bins'], 
                                                          1.0)

    # Interpolate enclosed mass profile and interpolate out rdelta (M_enc)
    result_dict[f'M.enc:{halo_def}:{save_key}'] = interpolate_array(surf_dens['mass.bins:enclosed'], 
                                                            surf_dens['radial.bins'], 
                                                            result_dict[f'rad.{halo_def}'])
    
    return None


def compute_deprojected_properties(halo_index: int, halo_dict, particle_dict, 
                                   cosmo_dict, scale_factor: float, profile_bins: int = 25,  
                                   verbose: bool = False, *args, **kwargs) -> None:
    """Analysis constructs the three-dimensional halo profile and quantifies 
       properties via analytical profile fits. The power radius and new virial
       halo quantities are also computed.
    """
    results = dict()

    # set up local parameters
    halo_mass = halo_dict['virial.mass'][halo_index]
    halo_rad = halo_dict['virial.radius'][halo_index]
    halo_pos = halo_dict['position'][halo_index]
    _part_mass = particle_dict['Masses'] 
    _part_pos = particle_dict['Coordinates']
    _part_sep = _part_pos
    _part_sep_mag = deprojected.compute_seperation(_part_pos, 0.0)
    
    # obtain depth criteria
    xi = 1.5
    cond1 = (_part_sep_mag < xi*halo_rad)
    dw = np.where(cond1)[0]
    results['part.sep:mag'] = _part_sep_mag[dw]
    results['part.mass'] = _part_mass[dw]
    results['part.sep'] = _part_sep[dw]
    
    if verbose:
        print("... Computing deprojected properties")
   
    # compute power radius 
    lrmin = np.log10(0.01)
    lrmax = 1.0 
    kwargs = {
            'part_sep': results['part.sep'], 'part_mass': results['part.mass'],
            'log_bounds': (lrmin, lrmax), 'nbins':100, 'clean_up': True
            }
    mass_dens_1 = deprojected.compute_mass_profiles(**kwargs)
    kwargs = {
            'power_criteria': 0.6, 'h': cosmo_dict['h0'], 
            'rad_enc': mass_dens_1['radial.bins'], 
            'num_enc': mass_dens_1['number.bins:enclosed'],
            'rho_enc': mass_dens_1['density.bins:enclosed']
            }
    results['rad.pow'] = deprojected.compute_power_radius(**kwargs)
    if verbose:
        print(f"| Power radius: {results['rad.pow']:0.3f} kpc")

    # re-compute overdensities by calculating present density profile
    # and pass that result through the particle cosmology module
    lrmin = np.log10(0.75*results['rad.pow'])
    lrmax = np.log10(1.2*halo_rad) 
    kwargs = {
            'part_sep': results['part.sep'], 'part_mass': results['part.mass'],
            'log_bounds': (lrmin, lrmax), 'nbins': 35, 'clean_up': True
            }
    mass_dens_2 = deprojected.compute_mass_profiles(**kwargs)
    cosmo_func = cosmology_analysis.ParticleCosmology(**cosmo_dict)
    kwargs = {
            'rho_enc_arr': mass_dens_2['density.bins:enclosed'], 
            'rad_arr': mass_dens_2['radial.bins'],
            'a': scale_factor
            }
    dvir_quants = cosmo_func.compute_halo_mass_and_radius(**kwargs, halo_def='vir')
    d200c_quants = cosmo_func.compute_halo_mass_and_radius(**kwargs, halo_def='200c')
    results['mass.vir'] = dvir_quants['mass']
    results['mass.200c'] = d200c_quants['mass']
    results['rad.vir'] = dvir_quants['radius'] 
    results['rad.200c'] = d200c_quants['radius'] 
    if verbose:
        print(f"| New halo mass [vir]: {dvir_quants['mass']:0.3e} Msol")
        print(f"| New halo radius [vir]: {dvir_quants['radius']:0.3f} kpc")
        print(f"| New halo mass [200c]: {d200c_quants['mass']:0.3e} Msol")
        print(f"| New halo radius [200c]: {d200c_quants['radius']:0.3f} kpc")

    # with new virial quantities quantified compute the final mass profile 
    # and interpolate the density between rpower and rvir
    lrmin = np.log10(0.75*results['rad.pow']) 
    lrmax = np.log10(1.2*results['rad.vir']) 
    kwargs = {
            'part_sep': results['part.sep'], 'part_mass': results['part.mass'],
            'log_bounds': (lrmin, lrmax), 'nbins': 35, 'clean_up': True
            }
    mass_dens = deprojected.compute_mass_profiles(**kwargs)
    rad_arr = 10.0 ** np.linspace(np.log10(results['rad.pow']), 
                                  np.log10(results['rad.vir']), profile_bins)
    dens_arr = interpolate_array(mass_dens['density.bins:local'], 
                                 mass_dens['radial.bins'], 
                                 rad_arr)
    #results['deproj:radius'] = rad_arr
    #results['deproj:density'] = dens_arr
    
    # curve fit analysis for Einasto profile
    kwargs = {'rad_arr': rad_arr, 'rho_arr': dens_arr}
    params_ein = analytical.Einasto().curve_fit(**kwargs, set_alpha_017=True)
    results['ein:rho.2'] = params_ein['rho2']
    results['ein:r.2'] = params_ein['r2']
    results['ein:alpha'] = params_ein['alpha']
    results['ein:c.vir'] = results['rad.vir'] / params_ein['r2']
    results['ein:c.200c'] = results['rad.200c'] / params_ein['r2']
    if verbose:
        print(f"| c_vir [ein]: {results['ein:c.vir']:.3f}")
        print(f"| c_200c [ein]: {results['ein:c.200c']:.3f}")

    # curve fit analysis for NFW profile
    kwargs = {'rad_arr': rad_arr, 'rho_arr': dens_arr}
    params_nfw = analytical.NFW().curve_fit(**kwargs) 
    results['nfw:rho.s'] = params_nfw['rhos']
    results['nfw:r.s'] = params_nfw['rs']
    results['nfw:c.vir'] = results['rad.vir'] / params_nfw['rs']
    results['nfw:c.200c'] = results['rad.200c'] / params_nfw['rs']
    if verbose:
        print(f"| c_vir [nfw]: {results['nfw:c.vir']:.3f}")
        print(f"| c_200c [nfw]: {results['nfw:c.200c']:.3f}")

    return results


def interpolate_array(ydata: float, xdata: float, nxdata: float) -> float:
    """Pass x and y data to return interpolation result for `nxdata`"""
    intp = interp1d(xdata, ydata, fill_value='extrapolate')
    return intp(nxdata)


if __name__ == "__main__":
    main()
