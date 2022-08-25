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

# Grab Currrent Time Before Running the Code
start = time.time()

# local ----

# import simulation sample parameters
from sample import cosmologies
from sample import zoom_simulations

from tools.cosmology import preset_cosmologies

# ---------------------------------------------------------------------------

def main() -> None:
    print('-' * 75)
    print('*** Starting analysis ***')
    print('-' * 75)
    
    for sim in zoom_simulations.sample:
        print(f">>> {sim}")

        # load cosmological parameters
        cosmo_model = cosmologies.sample[sim]
        cosmo_param = preset_cosmologies.models[cosmo_model]
        
        # load in important parameters (e.g. quantities, cosmology, paths...)
        sim_dict = zoom_simulations.snapInfo[sim]
        snapNum = sim_dict['snapshot.number']
        scale_factor = sim_dict['scale.factor']
        redshift = sim_dict['redshift']
        catalogPath = sim_dict['ahf.catalog.path']
        snapPath = sim_dict['snapshot.path']
        
        # set halo mass (Msol) range to perform analysis on
        split_str = sim.split('_')
        matches = ['m11d', 'm11e', 'm11h', 'm11i']
        if any(halo in split_str for halo in matches):
            # m11 bounds
            mass_bounds = (7e8, 4e11)
        else:
            # m10 bounds
            mass_bounds = (9e6, 5e10)
        
        # load in halo catalog and apply post-processing functions
        from tools.load_data import load_halo
        unproc_catalog = load_halo.load_ahf_catalog(catalogPath, high_res_only=True, field_halo_only=True)
        proc_catalog = load_halo.convert_factors(unproc_catalog, cosmo_param['h0'], scale_factor) 
        halo_catalog = load_halo.sample_from_mass_range(proc_catalog, mass_bounds[0], mass_bounds[1])
        del unproc_catalog; del proc_catalog
        print(f"... Sample contains {halo_catalog['halo.id'].shape[0]} halos")
        
        # load particle data
        from tools.load_data import load_particles
        kwargs = {
                'simPath': snapPath, 'snapNum': snapNum, 'partType': 'PartType1',
                'key_list': ['Masses', 'Coordinates'],
                'hubble': cosmo_param['h0'], 'scale_factor': scale_factor,
                'coord_conv': zoom_simulations.coord_conv[sim]
                }
        part1 = load_particles.load_particles(**kwargs)
           
        # perform analysis of halo properties
        print("... Starting profile analysis")
        kwargs = {
                "halo_dict": halo_catalog, "particle_dict": part1, 
                "cosmo_dict": cosmo_param, "scale_factor": scale_factor
                }
        results = start_analysis(**kwargs, verbose=True)

        print(">>> Done!")
        print("-" * 75)

        # save results to hdf5
        outPath = f"/data17/grenache/aalazar/projects/researchproject_007/output/{sim}.hdf5"
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

def start_analysis(halo_dict, particle_dict, cosmo_dict, 
                   scale_factor: float, verbose: bool = False, 
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
    results['virial.mass'] = np.zeros(nsample)
    results['virial.radius'] = np.zeros(nsample)
    results['power.radius'] = np.zeros(nsample)
    results['nfw:rhos'] = np.zeros(nsample)
    results['nfw:rs'] = np.zeros(nsample)
    results['nfw:cvir'] = np.zeros(nsample)
    results['ein:rho2'] = np.zeros(nsample)
    results['ein:r2'] = np.zeros(nsample)
    results['ein:cvir'] = np.zeros(nsample)
    results['ein:alpha'] = np.zeros(nsample)
    results['deproj:radius'] = np.zeros((nsample, profile_bins))
    results['deproj:density'] = np.zeros((nsample, profile_bins))
    
    # initialize projected quantities
    axs_list = ['maj', 'int', 'min'] + [str(n) for n in range(5)]
    for axs in axs_list:
        results[f'E1:{axs}'] = np.zeros(nsample)
        results[f'R1:{axs}'] = np.zeros(nsample)
        results[f'Cvir:{axs}'] = np.zeros(nsample)
        results[f'M1kpc:{axs}'] = np.zeros(nsample)
        results[f'proj:radius:{axs}'] = np.zeros((nsample, profile_bins))
        results[f'proj:density:{axs}'] = np.zeros((nsample, profile_bins))
    
    # actual analysis starts here
    successful_ind = []
    for ind, hid in enumerate(halo_dict['halo.id'][:]): 
        if verbose:
            print(f'::: Analyzing halo {hid} :::')
        try:   
            kwargs = {
                    'halo_index': ind, 'halo_dict': halo_dict, 
                    'particle_dict': particle_dict, 'cosmo_dict': cosmo_dict, 
                    'scale_factor': scale_factor, 'profile_bins': profile_bins
                    }
            deproj = compute_deprojected_properties(**kwargs, verbose=False)
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
                proj = compute_projected_properties(**kwargs, verbose=False)
            except ValueError:
                if verbose:
                    print("!!! Cannot be fitted to, not counted in analysis !!!")
                pass
            else:
                # If no errors when fitting, mark this as a successful result
                # and save projected quantitie
                successful_ind.append(ind)
                for skey, sval in proj.items():
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
    xi = 1.0
    cond1 = (results['part.sep:mag'] < xi*results['virial.radius'])
    dw = np.where(cond1)[0]
    results['part.sep:mag'] = results['part.sep:mag'][dw]
    results['part.mass'] = results['part.mass'][dw]
    results['part.sep'] = results['part.sep'][dw]

    if verbose:
        print("...Computing projected properties")

    # pass halo-centered particle coordinates through 3D rotation class
    rot_coords = rotations.CoordinateRotation3D(results['part.sep'])
    
    # performing 5 random coordinate rotations of a single halo
    n_rotation_success = 0
    n_rotation_reattempts = 0
    n_rotations = 5
    while(n_rotation_success < n_rotations):
        try:
            kwargs = {
                    'result_dict': results, 
                    'rot_coords_class': rot_coords, 
                    'part_mass': results['part.mass'], 
                    'profile_bins': profile_bins,
                    'save_key': str(n_rotation_success)
                    }
            eval_along_axis(**kwargs) 
        except ValueError:
            n_rotation_reattempts += 1 
        else:            
            if verbose:
                print(f"| Rand rotation {n_rotation_success} C_vir: \
                        {results[f'Cvir:{n_rotation_success}']:0.3f}")
            n_rotation_success += 1  
    if verbose:
        print(f"Random rotation re-attempts: {n_rotation_reattempts}")
            
    # now compute unit vectors pointing the density axis        
    kwargs = {
            'arr_in': results['part.sep'], 'rad': np.array([0.5, 1.5]),
            'shell': True, 'axes_out': False, 'fix_volume': True, 'quiet': True
            }
    axs = shapes.axis(**kwargs)

    # evaluate halo major, intermediate, and minor axis
    kwargs = {
            'result_dict': results, 'rot_coords_class': rot_coords, 
            'part_mass': results['part.mass'], 'profile_bins': profile_bins
            }
    eval_along_axis(**kwargs, save_key='maj', axis=axs['axis:major'])
    eval_along_axis(**kwargs, save_key='int', axis=axs['axis:intermediate'])
    eval_along_axis(**kwargs, save_key='min', axis=axs['axis:minor'])
    if verbose:
        print(f"| Maj C_vir: {results['Cvir:maj']:0.3f}")
        print(f"| Int C_vir: {results['Cvir:int']:0.3f}")
        print(f"| Min C_vir: {results['Cvir:min']:0.3f}")

    return results


def eval_along_axis(result_dict, rot_coords_class, part_mass: float, 
                    profile_bins: int = 25, save_key: str = None, 
                    axis: float = None, *args, **kwargs) -> None:
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
    lrmin = np.log10(0.75*result_dict['power.radius']) 
    lrmax = np.log10(1.5*result_dict['virial.radius']) 
    kwargs = {
            'part_sep': nz_pos, 'part_mass': part_mass,
            'log_bounds': (lrmin, lrmax), 'nbins': 35, 'clean_up': True
            }
    surf_dens = projected.compute_surface_profiles(**kwargs)
    rad_arr = 10.0 ** np.linspace(np.log10(result_dict['power.radius']), 
                                  np.log10(result_dict['virial.radius']), profile_bins)
    dens_arr = interpolate_array(surf_dens['density.bins:local'], 
                                 surf_dens['radial.bins'], 
                                 rad_arr)
    result_dict['proj:radius:{save_key}'] = rad_arr
    result_dict['proj:density:{save_key}'] = dens_arr

    kwargs = {'rad_arr': rad_arr, 'sigma_arr': dens_arr}
    params_laz = analytical.Lazar().curve_fit(**kwargs, set_beta_03=True)
    result_dict[f'E1:{save_key}'] = params_laz['sigma1']
    result_dict[f'R1:{save_key}'] = params_laz['R1']
    result_dict[f'Cvir:{save_key}'] = result_dict['virial.radius'] / params_laz['R1']
    
    # Interpolate enclosed mass profile and interpolate out 1 kpc
    mass_1kpc = interpolate_array(surf_dens['mass.bins:enclosed'], 
                                 surf_dens['radial.bins'], 
                                 1.0)
    result_dict[f'M.1kpc:{save_key}'] = mass_1kpc

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
    _part_sep = halo_pos - _part_pos
    _part_sep_mag = deprojected.compute_seperation(halo_pos, _part_pos)
    
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
    results['power.radius'] = deprojected.compute_power_radius(**kwargs)
    if verbose:
        print(f"| Power radius: {results['power.radius']:0.3f} kpc")

    # re-compute overdensities by calculating present density profile
    # and pass that result through the particle cosmology module
    lrmin = np.log10(0.75*results['power.radius'])
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
    vir_quants = cosmo_func.overdensity_def_vir(**kwargs)
    results['virial.mass'] = vir_quants['mass']
    results['virial.radius'] = vir_quants['radius'] 
    if verbose:
        print(f"| New virial mass: {vir_quants['mass']:0.3e} Msol")
        print(f"| New virial radius: {vir_quants['radius']:0.3f} kpc")

    # with new virial quantities quantified compute the final mass profile 
    # and interpolate the density between rpower and rvir
    lrmin = np.log10(0.75*results['power.radius']) 
    lrmax = np.log10(1.2*results['virial.radius']) 
    kwargs = {
            'part_sep': results['part.sep'], 'part_mass': results['part.mass'],
            'log_bounds': (lrmin, lrmax), 'nbins': 35, 'clean_up': True
            }
    mass_dens = deprojected.compute_mass_profiles(**kwargs)
    rad_arr = 10.0 ** np.linspace(np.log10(results['power.radius']), 
                                  np.log10(results['virial.radius']), profile_bins)
    dens_arr = interpolate_array(mass_dens['density.bins:local'], 
                                 mass_dens['radial.bins'], 
                                 rad_arr)
    results['deproj:radius'] = rad_arr
    results['deproj:density'] = dens_arr
    
    # curve fit analysis for Einasto profile
    kwargs = {'rad_arr': rad_arr, 'rho_arr': dens_arr}
    params_ein = analytical.Einasto().curve_fit(**kwargs, set_alpha_017=True)
    results['ein:rho2'] = params_ein['rho2']
    results['ein:r2'] = params_ein['r2']
    results['ein:alpha'] = params_ein['alpha']
    results['ein:cvir'] = results['virial.radius'] / params_ein['r2']
    if verbose:
        print(f"| c_vir [ein]: {results['ein:cvir']:.3f}")

    # curve fit analysis for NFW profile
    kwargs = {'rad_arr': rad_arr, 'rho_arr': dens_arr}
    params_nfw = analytical.NFW().curve_fit(**kwargs) 
    results['nfw:rhos'] = params_nfw['rhos']
    results['nfw:rs'] = params_nfw['rs']
    results['nfw:cvir'] = results['virial.radius'] / params_nfw['rs']
    if verbose:
        print(f"| c_vir [nfw]: {results['nfw:cvir']:.3f}")

    return results


def interpolate_array(ydata: float, xdata: float, nxdata: float) -> float:
    """Pass x and y data to return interpolation result for `nxdata`"""
    intp = interp1d(xdata, ydata, fill_value='extrapolate')
    return intp(nxdata)


if __name__ == "__main__":
    main()
