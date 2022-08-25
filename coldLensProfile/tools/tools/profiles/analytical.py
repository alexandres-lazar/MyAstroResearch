#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------

def main() -> None:
    Lazar()
    Einasto()
    NFW()

# ---------------------------------------------------------------------------

class Lazar():

    def __init__(self):
        pass

    def log_gradient_profile(self, r: float, R1: float, beta: float, 
                             *args, **kwargs) -> float:
        return - np.power(r/R1, beta)

    def profile(self, r: float, sigma1: float, R1: float, beta: float,
                *args, **kwargs) -> float:
        log_comp = -(1.0/beta) * (np.power(r/R1, beta) - 1.0)
        return sigma1 * np.exp(loc_comp)
    
    def log_profile(self, r: float, logsigma1: float, R1: float, beta: float, 
                    *args, **kwargs) -> float:
        comp = -(1.0/beta) * (np.power(r/R1, beta) - 1.0)
        return logsigma1 + comp
    
    def log_profile_03(self, r: float, logsigma1: float, R1: float, 
                       *args, **kwargs) -> float:
        beta = 0.30
        comp = (1.0/beta) * (np.power(r/R1, beta) - 1.0)
        return logsigma1 - comp
    
    def curve_fit(self, rad_arr: float, sigma_arr: float, set_beta_03: bool = False, 
                  *args, **kwargs) -> None:
        """Curve-fit procedure for surface-density profiles. Can also specify 
           if beta is a free parameter or not. Results save to dictionary
        """
        results = dict()

        if not set_beta_03:
            bfp = _curve_fit_routine(self.log_profile, rad_arr, sigma_arr)
            results['sigma1'] = np.exp(bfp[0])
            results['R1'] = bfp[1]
            results['beta'] = bfp[2]
        else:
            bfp = _curve_fit_routine(self.log_profile_03, rad_arr, sigma_arr)
            results['sigma1'] = np.exp(bfp[0])
            results['R1'] = bfp[1]
            results['beta'] = 0.30
    
        return results


class Einasto():

    def __init__(self):
        pass

    def log_gradient_profile(self, r: float, r2: float, alpha: float, 
                             *args, **kwargs) -> float:
        return - 2.0 * np.power(r/r2, alpha)

    def profile(self, r: float, rho2: float, r2: float, alpha: float, 
                *args, **kwargs) -> float:
        log_comp =  (2.0/alpha) * (np.power(r/r2, alpha) - 1.0)
        return rho2 * np.exp(-loc_comp)
    
    def log_profile(self, r: float, logrho2: float, r2: float, alpha: float,
                    *args, **kwargs) -> float:
        comp = (2.0/alpha) * (np.power(r/r2, alpha) - 1.0)
        return logrho2 - comp 
    
    def log_profile_017(self, r: float, logrho2: float, r2: float, 
                        *args, **kwargs) -> float:
        alpha = 0.17
        comp = (2.0/alpha) * (np.power(r/r2, alpha) - 1.0)
        return logrho2 - comp 
    
    def curve_fit(self, rad_arr: float, rho_arr: float, set_alpha_017: bool = False, 
                  *args, **kwargs) -> None:
        """Curve-fit procedure for density profiles. Can also specify if alpha 
           is a free parameter or not. Results save to dictionary.
        """
        results = dict()

        if not set_alpha_017:
            bfp = _curve_fit_routine(self.log_profile, rad_arr=rad_arr, rho_arr=rho_arr)
            results['rho2'] = np.exp(bfp[0])
            results['r2'] = bfp[1]
            results['alpha'] = bfp[2]
        else:
            bfp = _curve_fit_routine(self.log_profile_017, rad_arr=rad_arr, rho_arr=rho_arr)
            results['rho2'] = np.exp(bfp[0])
            results['r2'] = bfp[1]
            results['alpha'] = 0.17
    
        return results


class NFW():

    def __init__(self):
        pass

    def profile(self, r: float, rhos: float, rs: float, 
                *args, **kwargs) -> float:
        return rhos / (r/rs) / (1.0 + r/rs)**2
    
    def log_profile(self, r: float, logrhos: float, rs: float, 
                    *args, **kwargs) -> float:
        return logrhos - np.log(r/rs) - 2.0*np.log(1.0 + r/rs)  
    
    def log_gradient_profile(self, r: float, rs: float, 
                             *args, **kwargs) -> float:
        return (1.0 - r/rs) / (1.0 + r/rs)

    def compute_norm(self, mass: float, radius: float, rs: float, 
                     *args, **kwargs) -> float:
        return mass / (4.0 * np.pi * rs**3 * self._f(radius/rs))
    
    def mass_profile(self, r: float, tot_mass: float, radius: float, c: float,
                     *args, **kwargs) -> float:
        x = r / radius
        return mass * self._f(x*c) / self._f(c) 

    def interp_concentration(self, mass: float, vmax: float, rmax: float, 
                             *args, **kwargs) -> float:
        """NFW concentration inerpolation from Moline et al. (2017)"""
        
        # arbituary concentration array
        power_arr = np.linspace(0, 2, 100)
        conc_arr = 10.0 ** power_arr
        
        # compute interpolation
        G = 4.3e-6 # kpc/Msol (km/s)^2
        func = (rmax * vmax**2) / (G * mass) * (self._f(carr)/self._f(2.163))
        intp = interp1d(func, conc_arr, fill_value='extrapolate')

        return intp(1.0)

    def curve_fit(self, rad_arr: float, rho_arr: float, *args, **kwargs) -> None:
        """Curve-fit procedure for density profiles. Results save to dictionary."""
        results = dict()
        bfp = _curve_fit_routine(self.log_profile, rad_arr=rad_arr, rho_arr=rho_arr)
        results['rhos'] = np.exp(bfp[0])
        results['rs'] = bfp[1]
        return results

    def _f(self, x: float) -> float:
        return np.log(1.0 + x) - x/(1.0 + x)


def _curve_fit_routine(log_function, rad_arr: float, rho_arr: float, *args) -> float:
    """Curve fit routine used to recover best-fit parameters."""
    log_rho_arr = np.log(rho_arr)
    prs, __ = curve_fit(log_function, rad_arr, log_rho_arr, maxfev = 1200)
    return prs


if __name__ == "__main__":
    pass
