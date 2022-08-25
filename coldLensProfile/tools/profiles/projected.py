#!/usr/bin/env python 2

import numpy as np

# ---------------------------------------------------------------------------

def main() -> None:
    compute_surface_profiles()

# ---------------------------------------------------------------------------

def compute_surface_profiles(part_sep: float, part_mass: float, nbins_equal: bool = False, 
                          log_bounds: float = None, nbins: int = 25, 
                          clean_up: bool = True, *args, **kwargs) -> float:
    """Computes the circularly averaged surface-density profiles from halo 
       centered particle positions [(3, N) numpy array] and particle mass
    """
    results = dict()

    # LOS is fixed to arbituary z-axis pass through
    part_sep_mag = np.sqrt(part_sep[:, 0]**2 + part_sep[:, 1]**2)

    # specificy which radial array to use
    if nbins_equal:
        pass 
    else:
        if log_bounds == None:
            power_arr = 0
        elif isinstance(log_bounds, tuple):
            power_arr = np.linspace(log_bounds[0], log_bounds[1], nbins)
        else:
            sys.exit("!!! Either dont put anything here or specify a tuple !!!")
        radial_arr = 10.0 ** power_arr


    # intialize data arrays
    results['radial.bins'] = np.zeros(nbins)
    results['density.bins:local'] = np.zeros(nbins)
    results['mass.bins:enclosed'] = np.zeros(nbins)
    results['density.bins:enclosed'] = np.zeros(nbins)

    for ind, rad in enumerate(radial_arr):
        # want to maintain same shape as nbins, so catch first index and set to zero 
        # when taking the averaged centered radius.
        if(ind == 0):
            lower_rad = 0.0
        else:
            lower_rad = radial_arr[ind-1]
        higher_rad = radial_arr[ind]
        
        # center the radial bins
        rad_centered = (lower_rad + higher_rad) / 2.0 
        results['radial.bins'][ind] += rad_centered

        # sample particles within circular shells (and enclosing) radius r 
        rcond1 = lower_rad < part_sep_mag 
        rcond2 = part_sep_mag < higher_rad 
        part_in_shell = (rcond1) & (rcond2)
        rcond3 = (part_sep_mag < rad_centered) 
        part_enc_rad = np.where(rcond3)[0]

        # compute local profile
        area_in_shell = np.pi * (higher_rad**2 - lower_rad**2) 
        numb_part_in_shell = np.sum(part_in_shell)
        tot_mass_in_shell = np.sum(part_mass[part_in_shell])
        results['density.bins:local'][ind] += tot_mass_in_shell / area_in_shell 

        # compute enclosed profile
        enclosed_area = np.pi * rad_centered**2
        enclosed_mass = np.sum(part_mass[part_enc_rad]) 
        results['mass.bins:enclosed'][ind] += enclosed_mass
        results['density.bins:enclosed'][ind] += enclosed_mass / enclosed_area

    if clean_up:
        nonzero_dens = (results['density.bins:local'] != 0.0)
        for skey, sval in results.items():
            results[skey] = sval[nonzero_dens]

    return results

if __name__ == "__main__":
    pass
