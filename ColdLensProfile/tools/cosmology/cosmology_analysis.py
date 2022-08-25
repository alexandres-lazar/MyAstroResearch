#!/usr/bin/env python3

import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

if sys.version_info < (3, 6):
    sys.exit("!!! Please use Python 3.6+ to execute script!!!")
    
SPEED_OF_LIGHT = 3e+5 # km/s
GRAV_CONSTANT_KG = 6.67e-11 # m^3/kg/s^2
GRAV_CONSTANT_MSOL = 4.3e-6 # kpc^2 Msol^-1 (km/s)^2

# ---------------------------------------------------------------------------

# still testing...
def main() -> None:
    ParticleCosmology() 

# ---------------------------------------------------------------------------

class SetCosmology(object):

    def __init__(self, h0: float = 0.6774, Om0: float = 0.3089, Ol0: float = 0.6911, 
            Ob0: float = 0.0486, Or0: float = 0.0, model: str = None, **kwargs):
        if model == None:
            self.h0 = float(h0)
            self.Om0 = float(Om0)
            self.Ol0 = float(Ol0)
            self.Ob0 = float(Ob0)
            self.Or0 = float(Or0)
        else:
            pass
            """
            try:
                preset = preset_cosmologies.models[model]
            except KeyError:
                raise Exception(f"{model} is not a preset")
            else:
                self.h0 = preset['h0']
                self.Om0 = preset['Om0']
                self.Ol0 = preset['Ol0']
                self.Ob0 = preset['Ob0']
                self.Or0 = preset['Or0']
            """

    def scale_factor(self, z):
        """Scale factor at redshift `z`"""
        return 1.0 / (z + 1.0)
    
    def redshift(self, a):
        """redshift at scale factor `a`"""
        return 1.0/a - 1.0

    def H(self, z):
        """The hubble constant"""
        H0 = 100 * self.h0 # km/s/Mpc
        return H0 * np.sqrt(self._E_factor(z)) # km/s/Mpc

    def _E_factor(self, z):
        return self.Om0*np.power(1.0 + z, 3) + self.Ol0

    def delta_vir(self, z): 
        """The virial overdensity from Bryan & Norman (1998)"""
        a = self.scale_factor(z)
        Omega = self.Om0*np.power(1.0 + z, 3) / (self.Om0*np.power(1.0 + z, 3) + self.Ol0)
        x = Omega - 1.0
        return (18.0*np.pi**2 + 82.0*x - 39.0*x**2)

    def rho_c(self, z):
        """Critical density of the universe at redshift z"""
        a = self.scale_factor(z)
        hubble = self.H(z) / 1000.0 # km/s/kpc
        G = GRAV_CONSTANT_MSOL 
        return 3.0*hubble**2 / (8*np.pi*G) # Msol/kpc^3

    def rho_m(self, z: float) -> float:
        """Mean matter density of the universe at redshift z"""
        return self.Om0 * self.rho_c(z)

    def reference_density(self, z: float, mass_def: str = 'vir') -> float:
        """Returns product of background density and overdensity (kg/km^3) at redshift z"""
        if mass_def == "vir":
            bg_density = self.rho_c(z)
            overdensity = self.delta_vir(z)
            ref_density = overdensity * bg_density
        elif mass_def == "200c":
            bg_density = self.rho_c(z)
            overdensity = 200.0
            ref_density = overdensity * bg_density
        elif mass_def == "200m":
            bg_density = self.rho_m(z)
            overdensity = 200.0
            ref_density = overdensity * bg_density
        else:
            raise Exception(f"!!! {mass_def} is not a available mass definition !!!")
        return ref_density

    def virial_radius(self, z: float, mass: float, mass_def: str = "vir") -> float:
        """Virial radius (kpc) of a dark matter halo mass (Msol) at redshift z"""
        ref_density = self.reference_density(z, mass_def)
        mass_conv = mass 
        r_cubed = 3.0 * mass_conv / (4.0 * np.pi * ref_density) # kpc^3
        return np.power(r_cubed, 1.0/3.0)
    
    def virial_mass(self, z: float, radius: float, mass_def: str = "vir") -> float:
        """Virial mass (Msol) of a dark matter halo radius(kpc) at redshift z"""
        ref_density = self.reference_density(z, mass_def)
        rad_conv_cubed = radius**3 # kpc^3
        mass = (4.0*np.pi/3.0) * ref_density * rad_conv_cubed
        return mass

    def _conv_Msol_to_kg(self, x: float) -> float:
        return x * 1.989e+30

    def _conv_kg_to_Msol(self, x: float) -> float:
        return x/1.989e+30

    def _conv_km_to_kpc(self, x: float) -> float:
        return x * 3.24078e-17
    
    def _conv_kpc_to_km(self, x: float) -> float:
        return x / 3.24078e-17

class ParticleCosmology(SetCosmology):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_halo_mass_and_radius(self, rho_enc_arr: float, 
                                    rad_arr: float, a: float, 
                                    halo_def: str = 'vir') -> float:
        """
        """
        if halo_def == 'vir':
            return self.overdensity_def_vir(rho_enc_arr, rad_arr, a)
        elif halo_def == '200c':
            return self.overdensity_def_200c(rho_enc_arr, rad_arr, a)
        else:
            raise Exception("!!! Only definitions available are `vir` and `200c` !!!")
        
    def overdensity_def_200c(self, rho_enc_arr: float, 
                                  rad_arr: float, a: float) -> float:
        """
        """
        results = dict()

        # load analytical quantities
        z = self.redshift(a) 
        rhoc = self.rho_c(z)
        delta = 200.0

        # interpolate enclosed density and radius to fetch virial radius
        intp = interp1d(rho_enc_arr/rhoc, rad_arr, fill_value="extrapolate")
        results['radius'] = intp(delta)
        results['mass'] = (4.0*np.pi/3.0) * delta * rhoc * results['radius']**3
        
        return results

    def overdensity_def_vir(self, rho_enc_arr: float, 
                                  rad_arr: float, a: float) -> float:
        """
        """
        results = dict()

        # load analytical quantities
        z = self.redshift(a) 
        rhoc = self.rho_c(z) 
        delta = self.delta_vir(z)

        # interpolate enclosed density and radius to fetch virial radius
        intp = interp1d(rho_enc_arr/rhoc, rad_arr, fill_value="extrapolate")
        results['radius'] = intp(delta)
        results['mass'] = (4.0*np.pi/3.0) * delta * rhoc * results['radius']**3
        
        return results

   
if __name__ == "__main__":
    main()
