import numpy as np
import scipy as sp
import scipy.integrate as integrate
import myCosmology
 
class Dynamics(myCosmology.Cosmology):
    
    def __init__(self, **kwargs):
        super(Dynamics, self).__init__(**kwargs)

    def periodic(self, z, sep):
        a = self.scale_factor(z)
        boxl = self.boxsize * a * 1000 # phyical kpc
        sep[sep < -boxl/2] += boxl
        sep[sep > boxl/2] -= boxl
        return sep

    def distance(self, z, hostpos, satpos):
        results = {}
        """
        Converts the the positions of the host and satellite to 
        to radial galactocentric distance in units of kpc. This considers
        a comoving volume as well a periodic conditions of the box.
        """
        #a = self.scale_factor(z)
        sep_vec = np.zeros(satpos.shape) # physical kpc
        sep_vec[:, 0] = (hostpos[:, 0] - satpos[:, 0])
        sep_vec[:, 1] = (hostpos[:, 1] - satpos[:, 1])
        sep_vec[:, 2] = (hostpos[:, 2] - satpos[:, 2])
        sep_mag = np.sqrt(np.power(sep_vec[:, 0],2) + np.power(sep_vec[:, 1],2) + np.power(sep_vec[:, 2],2))
        results['relative.distance:vector'] = sep_vec
        results['relative.distance:magnitude'] = sep_mag
        return results

     def velocity_with_hubble(self, z, hostvel, satvel, hostpos, satpos):
        results = {}
        """
        Takes the positions and peculiar velocities of the illustris
        halos and computes the total velocity of the satellite in the 
        galactocentric frame. This considers the comoving volume, 
        periodic conditions, the hubble flow.
        """
        #a = self.scale_factor(z)
        H = self.H(z) * 1e-3 # km/s/kpc

        #...compute distances
        dist_dict = self.distance(z, hostpos, satpos) # physical kpc
        reldis_vec = dist_dict['relative.distance:vector']
        reldis_mag = dist_dict['relative.distance:magnitude']

        #...compute total velocities
        relvel_vec = np.zeros(satvel.shape) # km/s
        relvel_vec[:, 0] = (hostvel[:, 0] - satvel[:, 0])
        relvel_vec[:, 1] = (hostvel[:, 1] - satvel[:, 1])
        relvel_vec[:, 2] = (hostvel[:, 2] - satvel[:, 2])

        hubble = np.zeros(reldis_vec.shape)
        for i in range(reldis_vec.shape[0]):
            hubble[i] += reldis_vec[i] * H[i]
        totvel_vec = hubble + relvel_vec # km/s
        totvel_mag = np.sqrt(np.power(totvel_vec[:, 0], 2) + np.power(totvel_vec[:, 1], 2) + np.power(totvel_vec[:, 2],2))
        results['total.velocity:vector'] = totvel_vec
        results['total.velocity:magnitude'] = totvel_mag

        #...compute radial velocities
        numer = totvel_vec[:,0]*reldis_vec[:, 0] + totvel_vec[:, 1]*reldis_vec[:, 1] + totvel_vec[:, 2]*reldis_vec[:, 2]
        results['radial.velocity'] = numer/reldis_mag

        #...compute tangential velocities
        results['tangential.velocity'] = np.sqrt(results['total.velocity:magnitude']**2 - results['radial.velocity']**2)

        return results
