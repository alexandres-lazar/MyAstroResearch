import numpy as np
import scipy as sp
import scipy.integrate as integrate

class Cosmology(object):

    def __init__(self, h0 = 0.704, omegaM = 0.2726, omegaL = 0.7274, omegaB = 0.0456, omegaR = 0.0, boxsize = 106.5, **kwargs):
        self.h0 = float(h0)
        self.omegaM = float(omegaM)
        self.omegaL = float(omegaL)
        self.omegaB = float(omegaB)
        self.omegaR = float(omegaR)
        self.boxsize = float(boxsize)

    def scale_factor(self, z):
        return 1.0 / (z + 1.0)

    def H(self, z):
        H0 = 100 * self.h0 # km/s/Mpc
        aa = self.scale_factor(z)
        omegaMz = (self.omegaM + self.omegaB) * np.power(1.0+z, 3)
        omegaLz = self.omegaL
        omegaRz = self.omegaR * np.power(1.0+z, 4)
        omegaKz = (1.0 - self.omegaM - self.omegaL - self.omegaR) * np.power(1.0+z, 2)
        return H0 * np.sqrt(omegaMz + omegaLz) # km/s/Mpc

    def delta_vir(self, z): # The virial overdensity from Bryan & Norman (1998)
        a = self.scale_factor(z)
        Omega = self.omegaM*np.power(1.0 + z,3) / (self.omegaM*np.power(1.0 + z, 3) + self.omegaL)
        x = Omega - 1.0
        return (18.0*np.pi**2.0 + 82.0 * x - 39.0 * x**2.)
      
      
    def rho_crit(self, z):
        a = self.scale_factor(z)
        H = self.H(z) / 3.09e+19 # 1/s
        G = 6.67408e-11 /1000 / 1000 / 1000 # Converson of m^3/kg/s^2 --> km^3/kg/s^2
        return 3. * H**2. /(8 * np.pi * G) # kg/km^3

    def virial_radius(self, z, Mvir):
        Mvir = np.float64(Mvir) * 1.989e+30 # Conversion of Msol --> kg
        return np.power(3.0 * Mvir / self.delta_vir(z) / self.rho_crit(z) / (4.0 * np.pi), 1.0/3.0) * 3.24078e-17 #      km --> kpc

    def virial_velocity(self, z, Mvir):
        G = 6.67408e-11 / 1000/ 1000 / 1000 # m^3/kg/s^2 --> km^3/kg/s^2
        radius = self.virial_radius(z, Mvir) * 3.086e+16 # kpc --> km
        Mvir = float(Mvir) * 1.989e+30 # Msol --> kg
        return np.sqrt(G * Mvir / radius)

    def virial_energy(self, z, Mvir):
        return self.virial_velocity(z, Mvir) * self.virial_velocity(z, Mvir)

    def virial_angularmomentum(self, z, Mvir):
        return self.virial_velocity(z, Mvir) * self.virial_radius(z, Mvir)

