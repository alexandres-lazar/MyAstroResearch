#!/usr/bin/env python3

import sys
import numpy as np
from scipy.integrate import quad

if sys.version_info < (3, 6):
    sys.exit("!!! Please use Python 3.6+ to execute script!!!")

# ---------------------------------------------------------------------------

def main() -> None:
    return CosmologyFunctions()

# ---------------------------------------------------------------------------

class CosmologyFunctions(object):

    def __init__(self, h0: float = 0.6774, Om0: float = 0.3089, Ol0: float = 0.6911, 
                       Ob0: float = 0.0486, Or0: float = 0.0, boxsize: float = 15.0, 
                       **kwargs):
        self.h0 = float(h0)
        self.Om0 = float(Om0)
        self.Ol0 = float(Ol0)
        self.Ob0 = float(Ob0)
        self.Or0 = float(Or0)
        self.boxsize = float(boxsize)

    def scale_factor(self, z):
        return 1.0 / (z + 1.0)

    def hubble(self, z):
        H0 = 100 * self.h0 # km/s/Mpc
        aa = self.scale_factor(z)
        omegaMz = (self.Om0) * np.power(1.0 + z, 3)
        omegaLz = self.Ol0
        omegaRz = self.Or0 * np.power(1.0 + z, 4)
        omegaKz = (1.0 - self.Om0 - self.Ol0 - self.Or0) * np.power(1.0+z, 2)
        return H0 * np.sqrt(omegaMz + omegaLz) # km/s/Mpc

    def Ea(self, a):
        return (self.Om0) * np.power(a, -3) + self.Ol0

    def delta_vir(self, z): 
        """The virial overdensity from Bryan & Norman (1998)
        """
        a = self.scale_factor(z)
        Omega = self.Om0*np.power(1.0 + z, 3) / (self.Om0*np.power(1.0 + z, 3) + self.Ol0)
        x = Omega - 1.0
        return (18.0*np.pi**2 + 82.0*x - 39.0*x**2)

    def rho_crit(self, z):
        a = self.scale_factor(z)
        H = self.hubble(z) / 3.09e+19  # 1/s
        G = 6.67408e-11 * 1e-9         # Converson of m^3/kg/s^2 --> km^3/kg/s^2
        return 3.0*H**2 / (8*np.pi*G)   # kg/km^3

    def virial_radius(self, z, Mvir):
        Mvir = np.float64(Mvir) * 1.989e+30 # Conversion of Msol --> kg
        r_cubed = 3.0 * Mvir / (4.0 * np.pi * self.delta_vir(z) * self.rho_crit(z))
        return np.power(r_cubed, 1.0/3.0) * 3.24078e-17 # km --> kpc

    def angular_distance(self, z):
        a = self.scale_factor(z)
        H0 = 100 * self.h0 # km/s/Mpc
        c = 3e+5 # km/s
        func = lambda x: 1.0 / np.power(x, 2) / self.Ea(x)
        return a * c/H0 * integrate.quad(func, a, 1.0)[0] * 1000.0 # kpc/arcsec

    def angular_phy_size(self, z, theta):
        return self.angular_distance(z) * theta  * 4.84814e-6 # kpc/arcsecond --> kpc

if __name__ == '__main__':
    print(main())
