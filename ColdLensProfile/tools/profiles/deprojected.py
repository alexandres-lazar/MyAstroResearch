#!/usr/bin/env python3

import numpy as np
from scipy.optimize import curve_fit

G_CONSTANT = 4.3e-6 

# ---------------------------------------------------------------------------

def main() -> None:
    compute_mass_profiles()
    histedges_equal_number()
    weighted_dispersion_inshell()
    mass_weighted_dispersion()
    velocity_conversion_from_cartesian_to_spherical()
    shrinking_spheres_center()
    compute_power_radius()
    compute_seperation()

# ---------------------------------------------------------------------------

def compute_mass_profiles(part_sep: float, part_mass: float, nbins_equal: bool = False, 
                          log_bounds: float = None, nbins: int = 25, 
                          clean_up: bool = True, *args, **kwargs) -> float:
    """Computes the spherically averaged density profiles from halo centered 
       particle positions [(3, N) numpy array] and particle mass
    """
    results = dict()
   
    # compute magnitude of particle seperation
    part_sep_mag = compute_seperation(part_sep, 0.0)

    # specificy which radial array to use
    if nbins_equal:
        radial_arr = histedges_equal_number(part_sep_mag, nbins) 
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
    results['number.bins:local'] = np.zeros(nbins)
    results['number.bins:enclosed'] = np.zeros(nbins)
    results['mass.bins:enclosed'] = np.zeros(nbins)
    results['density.bins:enclosed'] = np.zeros(nbins)
    results['circular.velocity.bins'] = np.zeros(nbins)

    for ind, rad in enumerate(radial_arr):
        # want to maintain same shape as nbins, so catch first index and set to zero 
        # when taking the averaged centered radius.
        if(ind == 0):
            lower_rad = 0.0
        else:
            lower_rad = radial_arr[ind-1]
        higher_rad = rad
        
        # center the radial bins
        rad_centered = (lower_rad + higher_rad) / 2.0 
        results['radial.bins'][ind] += rad_centered
       
        # sample particles within spherical shells (and enclosing) radius r 
        rcond1 = lower_rad < part_sep_mag 
        rcond2 = part_sep_mag < higher_rad 
        rcond3 = part_sep_mag < rad_centered 
        part_in_shell = np.where((rcond1) & (rcond2))[0]
        part_enc_rad = np.where(rcond3)[0]

        # compute local profile
        vol_in_shell = (4.0*np.pi/3.0) * (higher_rad**3 - lower_rad**3) 
        numb_part_in_shell = part_in_shell.shape[0]
        tot_mass_in_shell = np.sum(part_mass[part_in_shell])
        results['density.bins:local'][ind] += tot_mass_in_shell / vol_in_shell 
        results['number.bins:local'][ind] += numb_part_in_shell / vol_in_shell 

        # compute enclosed profile
        enclosed_vol = (4.0*np.pi/3.0) * rad_centered**3
        enclosed_numb = part_enc_rad.shape[0] 
        enclosed_mass = np.sum(part_mass[part_enc_rad]) 

        results['number.bins:enclosed'][ind] += enclosed_numb
        results['mass.bins:enclosed'][ind] += enclosed_mass
        results['density.bins:enclosed'][ind] += enclosed_mass / enclosed_vol
        results['circular.velocity.bins'][ind] += np.sqrt(enclosed_mass * G_CONSTANT / rad_centered)

    if clean_up:
        nonzero_dens = (results['density.bins:local'] != 0.0)
        for skey, sval in results.items():
            results[skey] = sval[nonzero_dens]

    return results 


def histedges_equal_number(array, n_partitions: int) -> None:
    """Generates equal bin sizes for desired number of particles"""
    npt = array.shape[0]
    nbins = int(npt/n_partitions)
    return np.interp(np.linspace(0, npt, nbins+1), np.arange(npt), np.sort(array))


def weighted_dispersion_inshell(ppos: float, pvel: float, pmass: float, 
                                log_bounds = None, nbins: int = 25,
                                *args, **kwargs) -> float:
    """Compute the mass weighted radial, tangential, and anisotropy of a 
        halo or galaxy based on centered particle positions and velocites.
    """
    results = dict()

    radial_mag = compute_seperation(ppos, 0.0)
    rad_vel, theta_vel, phi_vel = velocity_conversion_cartesian_to_spherical(ppos, pvel)
    
    if log_bounds == None:
        power_arr = 0
    elif isinstance(log_bounds, tuple):
        power_arr = np.linspace(log_bounds[0], log_bounds[1], nbins)
    else:
        sys.exit("!!! Either dont put anything here or specify a tuple !!!")
    radial_arr = 10.0 ** power_arr

    results['radial.bins'] = np.zeros(rad_bins.size)
    results['sigma.rad.bins'] = np.zeros(rad_bins.size)
    results['sigma.tan.bins'] = np.zeros(rad_bins.size)
    results['beta.bins'] = np.zeros(rad_bins.size)

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

        # sample particles within spherical shells and prep for mass weighing 
        rcond1 = lower_rad < part_sep_mag 
        rcond2 = part_sep_mag < higher_rad 
        part_in_shell = (rcond1) & (rcond2)
        
        # compute mass weighted quantites
        sigma_rad_vel = mass_weighted_dispersion(rad_vel, pmass, part_in_shell)
        sigma_theta_vel = mass_weighted_dispersion(theta_vel, pmass, part_in_shell)
        sigma_phi_vel = mass_weighted_dispersion(phi_vel, pmass, part_in_shell)
        sigma_tan_vel = np.sqrt(sigma_phi_vel**2 + sigma_theta_vel**2)
        beta = 1.0 - sigma_tan_vel**2/(2.0*sigma_rad_vel**2)
       
        # just pass through very divergent quantities
        cond1 = np.isnan(sigma_rad_vel) 
        cond2 = sigma_rad_vel == 0.0
        if (cond1 & cond2):
            pass
        else:
            nonzero_rad = rad_centered != 0.0
            results['radial.bins'][ind] += rad_centered[nonzero_rad]
            results['sigma.rad.bins'][ind] += sigma_rad_vel[nonzero_rad]
            results['sigma.tan.bins'][ind] += sigma_tan_vel[nonzero_rad]
            results['beta.bins'][ind] += beta[non_zerorad]

    return results


def mass_weighted_dispersion(quantity: float, pmass: float, inshell_cond: bool,
                             *args, **kwargs) -> float:
    """Computes the mass weighted dispersion (assuming normal distribution) 
       of a given quantity.
    """
    # 
    mass_in_shell = pmass[inshell_cond]
    tot_mass_in_shell = np.sum(mass_in_shell) 
    quantity_in_shell = quantity[inshell_cond]

    # compute normal weight quantity
    quant_squared_w = np.sum(quantity_in_shell**2 * mass_in_shell) / tot_mass_in_shell
    quant_w = np.sum(quantity_in_shell * mass_in_shell) / tot_mass_in_shell
    sigma_quant_sq = quant_squared_w - quant_w**2
    
    return np.sqrt(sigma_quant_sq)


def velocity_conversion_from_cartesian_to_spherical(ppos: float, pvel: float) -> float:
    """Convert (3, N) numpy array of cartesian velocities to velocities to spherical coordinates
       of position and velocity centered particle data.
    """
    # radial component of velocity
    radial_mag = compute_seperation(pos, 0.0)
    radial_vel = compute_seperation(pos*vel, 0.0) / radial_mag

    # theta component of velocity
    theta_num = ppos[:, 2]*radial_vel - radial_vel*pvel[:, 2] 
    theta_den = np.sqrt(radial_vel**2 - ppos[:, 2]**2) 
    theta_vel = theta_num / theta_den

    # phi component of velocity
    phi_num = ppos[:, 0]*pvel[:, 1] - ppos[:, 1]*pvel[:, 0] 
    phi_den = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2) 
    phi_vel = phi_num / phi_den

    return np.stack([radial_vel, theta_vel, phi_vel], axis=-1)


def shrinking_spheres_center(partr, partv, partm, sout=9):
    """Computes the new halo center using the shrinking spheres method,
       which iteratively computes a new center for every successive 
       shrunken sphere until specific criteria is met.
    """
    # Easier to work if we stack partr and partv: x, y, z, vx, vy, vz
    partrv = np.hstack((partr, partv)) 

    # Calculate the phase-space center of the particles
    mtot = np.sum(partm)
    ps_center = np.dot(partm, partrv) / mtot
   
    # begin iteration until ps_center converges
    sepcen2 = 1.0
    while sepcen2 > 0: 
      # calculate difference between each particle and the center at 
      # each component
      sigma_sq = np.dot(pmass, (partrv-ps_center)**2)
      sigma_sq = sigma_sq / (mtot - (mtot/float(partm.shape[0])))

      # set the limit of exclusion at `sout` times the `sigma_sq` value found
      # at each component
      exc_sq = sout * sigma_sq

      # now look which particles are `sout` times away and exclude them 
      sep_sq = np.sum((partrv-ps_center)**2/exc_sq, axis=1)
      cond = sep_sq < 1.0
      partm = partm[cond]
      partrv = partrv[cond]

      if partm.shape[0] == 0:
          # in case no particles, we keep last center
          sep_sq = 0.0 
      else:
          # calculate the new phase-space center for the particles
          mtot = np.sum(partm)
          new_center = np.dot(partm, partrv) / mtot
          
          # Compare the new center with the old one
          sepcen2 = np.sum((ps_center-new_center)**2)
          ps_center = new_center
    
    return ps_center


def compute_power_radius(power_criteria: float = 0.6, h: float = 0.7, 
                         rad_enc: float = None, num_enc: float = None, 
                         rho_enc: float = None, *args, **kwargs) -> float:
    """computes dark matter halo power radius using Eq (20) 
       from Power et al. (2003)
    """
    H0 = 100.0 * h / 1000.0 # km/s/kpc
    rho_crit = 3.0*H0**2 / (8.0*np.pi*G_CONSTANT) # Msol/kpc^3
    criteria = np.sqrt(200.0)/8.0 * num_enc/np.log(num_enc) * np.power(rho_enc/rho_crit, -0.5)

    argw = np.argwhere(criteria < power_criteria)[:,0]

    return np.max(rad_enc[argw])


def compute_seperation(pos1: float, pos2: float) -> float:
    """Quick and dirty method for computing position seperation.
       Be very careful with what you are evaluating.
    """
    sep = pos1 - pos2

    if isinstance(pos2, float):
        result = np.sqrt(sep[:, 0]**2 + sep[:, 1]**2 + sep[:, 2]**2)
    elif pos1.shape != pos2.shape:
        result = np.sqrt(sep[:, 0]**2 + sep[:, 1]**2 + sep[:, 2]**2) 
    else:
        result = np.sqrt(sep[0]**2 + sep[1]**2 + sep[2]**2)

    return result


if __name__ == "__main__":
    pass
