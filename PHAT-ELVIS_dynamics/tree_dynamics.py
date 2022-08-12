#!/usr/bin/env python3

# system ----
import os
import sys
import matplotlib
import h5py
import numpy as np
from scipy.integrate import quad
import scipy.interpolate

outPath = '/data17/grenache/aalazar/phat-ELVIS/disk/merger_analysis/outputs'

sims = list()
sims.append(1107)
sims.append(1245)
sims.append(1386)
sims.append(493)
sims.append(539)
sims.append(609)
sims.append(694)
sims.append(795)
sims.append(833)
sims.append(848)
sims.append(879)
sims.append(988)

params = {
    'h0'     : 0.6774,
    'omegaM' : 0.3089,
    'omegaL' : (1.0-0.3089),
    'omegaB' : 0.0000,
    'omegaR' : 0.0000,
    'boxsize': 1,
    }
h0 = params['h0']
G = 4.3e-6

from colossus.cosmology import cosmology
cosmology.setCosmology('planck15');
from colossus.lss import peaks

import myCosmology, myDynamics
cosmo = myCosmology.Cosmology(**params)
dyn = myDynamics.Dynamics(**params)

#z=0 halo catalog
# (0)host_id, (1)tree, (2)id, 
# (3)mvir, (4)rs, (5)rvir, (6)vmax,
# (7)vx, (8)vy, (9)vz, (10)x, (11)y, (12)z,
# vpeak, scale_vpeak

alist=np.around(np.loadtxt('a_out_short.txt'), decimals=5); zlist = 1.0/alist - 1.0
nalist = np.array([0.09031,0.09695,0.11023,0.12351,0.14343,0.16999,0.2,0.25,0.33333,0.49867,0.66799,1.0])
nastring = np.array(['index:z10.0','index:z9.0','index:z8.0','index:z7.0','index:z6.0','index:z5.0','index:z4.0','index:z3.0','index:z2.0','index:z1.0','index:z0.5','index:z0.0'])

'''
merger tree info:
        host_id : simulation number. Host mass decreases as this increases
        tree : tree number (main branch) this halo belongs to
        scale : scale factor of the halo properties
        id : specific halo number (unique within `host_id` groups)
        pid : ID of the parent halo (-1 if not a subhalo)
        upids : ID of the uppermost halo (-1 if not a sub-subhalo, equals `pid` if subhalo)
        phantom : Nonzero if the halo is not found by Rockstar, but used to link snapshots
        mass : mass of the subhalo (Msun/h)
        rvir : virial radius (kpc/h comoving)
        rs : scale radius of best fit NFW profile as determined by Rockstar (kpc/h comoving)
        vmax : maximum circular velocity (km/s)
        x : x coordinate of center (Mpc/h comoving)
        y : y coordinate of center (Mpc/h comoving)
        z : z coordinate of center (Mpc/h comoving)
        vx : x bulk velocity (km/s)
        vy : y bulk velocity (km/s)
        vz : z bulk velocity (km/s)
'''
# (0)host_id, (1)tree, (2)scale, (3)id, (4)pid, (5)upid, 
# (6)phantom, (7)mvir, (8)rvir, (9)rs, (10)vmax,
# (11)x,(12)y,(13)z,(14)vx,(15)vy,(16)vz







def nfw_potential(r: float, vmax: float, rvir: float, rmax: float) -> float:
    rs = rmax / 2.163
    x = r / rvir; 
    c = rvir / rs
    A = np.power(vmax/0.465, 2); B = np.log(1.0 + c*x) / (c*x)
    return -A * B

H0 = 100.0 * h0 # km/s/Mpc
invth = H0/3.086e+19 # 1/s
th = (1.0/invth) * 3.17098e-8 # years
c = 3e5 # km/s

def time(a):
    result = 0
    integrand = lambda x: 1.0/x/np.sqrt(params['omegaM']/x/x/x + params['omegaL'])
    result += th * quad(integrand, 0.0, a)[0]
    return result / 1e9

def infall_first_crossing(sdict, rvir, vvir):
    results = {}; npoints = 500
    nsep_ = (sdict['distance']/rvir); #print(nsep_)
    scl = np.linspace(min(sdict['scale']), max(sdict['scale']), npoints); #print(scl)
    nsep = scipy.interpolate.interp1d(sdict['scale'], nsep_)(scl)

    w = np.argmax(nsep <= 1.0)
    
    results['scale'] = scl[w]
    results['redshift'] = scipy.interpolate.interp1d(sdict['scale'], sdict['redshift'])(scl)[w]
    results['lookback'] = scipy.interpolate.interp1d(sdict['scale'], sdict['lookback'])(scl)[w]
    results['distance'] = scipy.interpolate.interp1d(sdict['scale'], sdict['distance'])(scl)[w]

    results['total.velocity:noHubble'] = scipy.interpolate.interp1d(sdict['scale'], sdict['total.velocity:noHubble'])(scl)[w]
    results['radial.velocity:noHubble'] = scipy.interpolate.interp1d(sdict['scale'], sdict['radial.velocity:noHubble'])(scl)[w]
    results['tangential.velocity:noHubble'] = scipy.interpolate.interp1d(sdict['scale'], sdict['tangential.velocity:noHubble'])(scl)[w]

    results['total.velocity:Hubble'] = scipy.interpolate.interp1d(sdict['scale'], sdict['total.velocity:Hubble'])(scl)[w]
    results['radial.velocity:Hubble'] = scipy.interpolate.interp1d(sdict['scale'], sdict['radial.velocity:Hubble'])(scl)[w]
    results['tangential.velocity:Hubble'] = scipy.interpolate.interp1d(sdict['scale'], sdict['tangential.velocity:Hubble'])(scl)[w]

    if(np.isscalar(vvir) == 1):
        results['norm:virial.radius'] = rvir
        results['norm:virial.velocity'] = vvir
    else:
        results['norm:virial.radius'] = scipy.interpolate.interp1d(sdict['scale'], rvir)(scl)[w]
        results['norm:virial.velocity'] = scipy.interpolate.interp1d(sdict['scale'], vvir)(scl)[w]
    return results
  
 def infall_last_crossing(sdict, rvir, vvir):
    results = {}; npoints = 500
    nsep_ = (sdict['distance']/rvir); #print(nsep_)
    scl = np.linspace(min(sdict['scale']),max(sdict['scale']), npoints)[::-1]; #print(scl)
    nsep = scipy.interpolate.interp1d(sdict['scale'], nsep_)(scl); #print(nsep)
    w = np.argmin(nsep < 1.0)

    results['scale'] = scl[w]
    results['redshift'] = scipy.interpolate.interp1d(sdict['scale'], sdict['redshift'])(results['scale'])
    results['lookback'] = scipy.interpolate.interp1d(sdict['scale'], sdict['lookback'])(results['scale'])
    results['distance'] = scipy.interpolate.interp1d(sdict['scale'], sdict['distance'])(scl)[w]

    results['total.velocity:Hubble'] = scipy.interpolate.interp1d(sdict['scale'], sdict['total.velocity:Hubble'])(results['scale'])
    results['radial.velocity:Hubble'] = scipy.interpolate.interp1d(sdict['scale'], sdict['radial.velocity:Hubble'])(results['scale'])
    results['tangential.velocity:Hubble'] = scipy.interpolate.interp1d(sdict['scale'], sdict['tangential.velocity:Hubble'])(results['scale'])
    results['space.velocity:Hubble'] = scipy.interpolate.interp1d(sdict['scale'], sdict['space.velocity:noHubble'])(results['scale'])

    results['total.velocity:noHubble'] = scipy.interpolate.interp1d(sdict['scale'], sdict['total.velocity:noHubble'])(results['scale'])
    results['radial.velocity:noHubble'] = scipy.interpolate.interp1d(sdict['scale'], sdict['radial.velocity:noHubble'])(results['scale'])
    results['tangential.velocity:noHubble'] = scipy.interpolate.interp1d(sdict['scale'], sdict['tangential.velocity:noHubble'])(results['scale'])
    results['space.velocity:noHubble'] = scipy.interpolate.interp1d(sdict['scale'], sdict['space.velocity:noHubble'])(results['scale'])

    if(np.isscalar(vvir) == 1):
        results['norm:virial.radius'] = rvir
        results['norm:virial.velocity'] = vvir
    else:
        results['norm:virial.radius'] = scipy.interpolate.interp1d(sdict['scale'], rvir)(scl)[w]
        results['norm:virial.velocity'] = scipy.interpolate.interp1d(sdict['scale'], vvir)(scl)[w]
    return results

  
  
  
 
def main() -> None:

    for hid in sims[:]:
        h5 = h5py.File(outPath + '/tree_dynamics_' + str(hid) + '_disk.hdf5', 'w')

        sim = np.genfromtxt('/data17/grenache/aalazar/phat-ELVIS/disk/catalog/halo_catalog_disk_' + str(hid) + '.csv',
                            skip_header = 1, delimiter = ',')
        treeid = sim[:, 1]
        mvir = sim[:, 3]/h0; rvir = sim[:, 5]/h0 # ckpc
        vmax = sim[:, 6];
        pos = np.zeros((mvir.shape[0], 3))
        pos[:, 0] = sim[:,10]; pos[:, 1] = sim[:, 11]; pos[:, 2] = sim[:, 12];
        pos *= 1e3/h0 # ckpc
        vel = np.zeros((mvir.shape[0], 3)) # km/s
        vel[:, 0] = sim[:, 7]; vel[:, 1] = sim[:, 8]; vel[:, 2] = sim[:, 9];
        vpeak = sim[:, 13]; vp_scale = sim[:, 14]

        #...host halo @ z=0
        w = np.where(mvir == max(mvir))[0]
        htreeid = treeid[w][0]
        hmass = mvir[w][0]; hrvir = rvir[w][0]
        hpos = pos[w][0]; hvel = vel[w][0]

        #...satellites @ z=0
        streeid = np.delete(treeid, w, 0)
        smass = np.delete(mvir, w, 0);
        svmax = np.delete(vmax, w, 0);
        streeid = np.delete(treeid, w, 0)
        spos = np.delete(pos, w, 0); svel = np.delete(vel, w, 0)
        vpeak = np.delete(vpeak, w, 0); vp_scale = np.delete(vp_scale, w, 0)

        sep = hpos - spos; rsep = np.sqrt(sep[:, 0]**2 + sep[:, 1]**2 + sep[:, 2]**2)
        svel=hvel-svel; svel+=sep*H0*1e-3
        vrad = sep[:, 0]*svel[:, 0] + sep[:, 1]*svel[:, 1] + sep[:, 2]*svel[:, 2]; vrad /= rsep
        mask = np.where((rsep/hrvir < 1.0) & (svmax > 4.5) & (smass > 7e7))[0]
        print(mask.shape[0])
        nstree_id_ = streeid[mask]

        #...tree dynamics
        tree = np.genfromtxt('/data17/grenache/aalazar/phat-ELVIS/disk/trees/main_branches_disk_' + str(hid) + '_trimmed.csv',
                           skip_header = 1, delimiter = ',')
        tree_ids = tree[:, 1]; scale = tree[:, 2]; redshift = 1.0/scale - 1.0
        tmvir = tree[:, 7]/h0; trvir=tree[:, 8]/h0 # ckpc
        tvmax = tree[:, 10]; trmax = tree[:, 9] * 2.163/h0 # ckpc
        xpos = tree[:, 11]; ypos = tree[:, 12]; zpos = tree[:, 13]
        xvel = tree[:, 14]; yvel = tree[:, 15]; zvel = tree[:, 16]

        host_tree = {}
        wh = np.where(htreeid == tree_ids)[0]
        host_tree['virial.mass'] = tmvir[wh]
        host_tree['scale'] = scale[wh]
        host_tree['redshift'] = redshift[wh]

        age = np.zeros(host_tree['scale'].shape[0])
        for i in range(host_tree['scale'].shape[0]):
            age[i] += time(host_tree['scale'][i]) # Gyr

        host_tree['lookback'] = time(1.9) - age; #print lookback
        host_tree['virial.radius'] = host_tree['scale'] * trvir[wh]/h0 # physical kpc
        host_tree['virial.velocity'] = np.sqrt(G * host_tree['virial.mass']/host_tree['virial.radius']) # km/s
        host_tree['vel.max'] = tvmax[wh] # physical kpc
        host_tree['rad.max'] = host_tree['scale'] * trmax[wh]/h0 # physical kpc

        host_tree['position'] = np.zeros((host_tree['scale'].shape[0],3)) # physical kpc
        host_tree['position'][:, 0] = host_tree['scale'] * xpos[wh] * 1e3/h0
        host_tree['position'][:, 1] = host_tree['scale'] * ypos[wh] * 1e3/h0
        host_tree['position'][:, 2] = host_tree['scale'] * zpos[wh] * 1e3/h0

        host_tree['velocity'] = np.zeros((host_tree['scale'].shape[0], 3)) # peculiar km/s
        host_tree['velocity'][:, 0] = xvel[wh]; host_tree['velocity'][:, 1] = yvel[wh]; host_tree['velocity'][:, 2] = zvel[wh]

        host_tree['peak.height'] = np.zeros(host_tree['scale'].shape[0])
        for i in range(host_tree['peak.height'].shape[0]):
            host_tree['peak.height'][i] += peaks.peakHeight(host_tree['virial.mass'][i]*h0, host_tree['redshift'][i])

        h5.create_group('host.tree')
        h5['host.tree'].create_dataset('virial.mass', data = host_tree['virial.mass'])
        h5['host.tree'].create_dataset('virial.radius', data = host_tree['virial.radius'])
        h5['host.tree'].create_dataset('lookback.time', data = host_tree['lookback'])
        h5['host.tree'].create_dataset('redshift', data = host_tree['redshift'])
        h5['host.tree'].create_dataset('vel.max', data = host_tree['vel.max'])
        h5['host.tree'].create_dataset('rad.max', data = host_tree['rad.max'])
        h5['host.tree'].create_dataset('peak.height', data = host_tree['peak.height'])

        #...throw out orphan subhalos
        nstree_id = []
        for tid in nstree_id_:
            w = np.where(tid == tree_ids)[0]; z = redshift[w]
            if(z[-1] > 3):
                nstree_id.append(tid)
            else:
                pass
        nstree_id = np.array(nstree_id)

        #... satellite dynamical analysis
        h5.create_group('sat.tree')
        ftree_id = []
        for tid in nstree_id[:]:
            sat_tree = {}
            #...partition out satellite quantities baed on tree depth
            ws = np.where(tid == tree_ids)[0];
            sat_tree['virial.mass'] = tmvir[ws]
            sat_tree['scale'] = scale[ws]
            sat_tree['redshift'] = redshift[ws]
            sat_tree['vel.max'] = tvmax[ws] # physical kpc
            sat_tree['rad.max'] = sat_tree['scale']*trmax[ws] # physical kpc
            sat_tree['position'] = np.zeros((sat_tree['scale'].shape[0], 3)) # physical kpc
            sat_tree['position'][:, 0] = sat_tree['scale'] * xpos[ws]*1e3/h0
            sat_tree['position'][:, 1] = sat_tree['scale'] * ypos[ws]*1e3/h0
            sat_tree['position'][:, 2] = sat_tree['scale'] * zpos[ws]*1e3/h0
            sat_tree['velocity'] = np.zeros((sat_tree['scale'].shape[0], 3)) # peculiar km/s
            sat_tree['velocity'][:, 0] = xvel[ws]; sat_tree['velocity'][:, 1] = yvel[ws]; sat_tree['velocity'][:, 2] = zvel[ws]

            #...partition out matching redshift host halo quantities
            wh = np.where(np.in1d(host_tree['scale'], sat_tree['scale']))[0]
            if(wh.shape[0] != sat_tree['scale'].shape[0]):
                pass
            else:
                ftree_id.append(int(tid))

                for i in range(nalist.shape[0]):
                    w = np.where(nalist[i] == sat_tree['scale'])[0]
                    strin = nastring[i]
                    if(len(w) == 0):
                        sat_tree[strin] = -1
                    else:
                        sat_tree[strin] = w[0]

                age = np.zeros(sat_tree['scale'].shape[0])
                for i in range(sat_tree['scale'].shape[0]):
                    age[i] += time(sat_tree['scale'][i]) # Gyr

                sat_tree['lookback'] = time(1.0) - age; #print lookback

                sat_tree['distance'] = dyn.distance(sat_tree['redshift'], host_tree['position'][wh], 
                                                    sat_tree['position'])['relative.distance:magnitude']
                sat_tree['distance:pericenter'] = min(sat_tree['distance'])
                sat_tree['distance:scaled:evolving'] = host_tree['virial.radius'][wh]
                sat_tree['distance:scaled:fixed'] = host_tree['virial.radius'][0]

                sat_tree['velocity:scaled:evolving'] = host_tree['virial.velocity'][wh]
                sat_tree['velocity:scaled:fixed'] = host_tree['virial.velocity'][0]

                #...velocity w/ Hubble factor
                vel_dict_H = dyn.velocity_with_hubble(sat_tree['redshift'], host_tree['velocity'][wh], 
                                                      sat_tree['velocity'], host_tree['position'][wh], 
                                                      sat_tree['position'])
                sat_tree['total.velocity:Hubble'] = vel_dict_H['total.velocity:magnitude']
                sat_tree['radial.velocity:Hubble'] = vel_dict_H['radial.velocity']
                sat_tree['tangential.velocity:Hubble'] = vel_dict_H['tangential.velocity']
                sat_tree['infall.angle:Hubble'] = np.arccos(sat_tree['radial.velocity:Hubble'] / sat_tree['total.velocity:Hubble'])
                sat_tree['cos2.infall.angle:Hubble'] = sat_tree['radial.velocity:Hubble']**2 / sat_tree['total.velocity:Hubble']**2

                #...velocity w/out Hubble factor
                vel_dict = dyn.velocity(sat_tree['redshift'], host_tree['velocity'][wh], sat_tree['velocity'], host_tree['position'][wh], sat_tree['position'])
                sat_tree['total.velocity:noHubble'] = vel_dict['total.velocity:magnitude']
                sat_tree['radial.velocity:noHubble'] = vel_dict['radial.velocity']
                sat_tree['tangential.velocity:noHubble'] = vel_dict['tangential.velocity']
                sat_tree['infall.angle:noHubble'] = np.arccos(sat_tree['radial.velocity:noHubble'] / sat_tree['total.velocity:noHubble'])
                sat_tree['cos2.infall.angle:noHubble'] = sat_tree['radial.velocity:noHubble']**2 / sat_tree['total.velocity:noHubble']**2

                #...infall times
                # evolving virial radius
                infall_first1 = infall_first_crossing(sat_tree, host_tree['virial.radius'][wh], host_tree['virial.velocity'][wh]); #print(infall_first1)
                infall_last1 = infall_last_crossing(sat_tree, host_tree['virial.radius'][wh], host_tree['virial.velocity'][wh]); #print(infall_last1)
                abs_first_infall = np.abs(infall_first1['scale'] - infall_last1['scale']) <= 0.25

                # fixed virial radius at z=0
                infall_first2 = infall_first_crossing(sat_tree, host_tree['virial.radius'][0], host_tree['virial.velocity'][0]); #print(infall_first1)
                infall_last2 = infall_last_crossing(sat_tree, host_tree['virial.radius'][0], host_tree['virial.velocity'][0]); #print(infall_last1)

                #infall definition 3: maximum bound mass
                w3 = np.where(max(sat_tree['virial.mass']) == sat_tree['virial.mass'])[0][0]
                sat_tree['infall.index:mass'] = w3

                #infall definition 4: peak velocity
                w4 = np.where(max(sat_tree['vel.max']) == sat_tree['vel.max'])[0][0]
                sat_tree['infall.index:vmax'] = w4

                pot_list = nfw_potential(sat_tree['distance'], host_tree['vel.max'][wh], host_tree['virial.radius'][wh], host_tree['rad.max'][wh])
                vesc_list = np.sqrt(2.*np.abs(pot_list))

                bound_mask = np.full(sat_tree['scale'].shape[0], False)
                boundH_mask = np.full(sat_tree['scale'].shape[0], False)
                for i in range(bound_mask.shape[0]):
                    if(sat_tree['total.velocity:noHubble'][i] < vesc_list[i]):
                        bound_mask[i] = True
                    if(sat_tree['total.velocity:Hubble'][i] < vesc_list[i]):
                        boundH_mask[i] = True



                sh5=h5['sat.tree'].create_group('sat.id:' + str(int(tid)))
                
                sh5.create_dataset('flag:bound:noHubble', data=bound_mask)
                sh5.create_dataset('flag:bound:Hubble', data=boundH_mask)
                sh5.create_dataset('flag:abs.first.infall', data= a bs_first_infall)
                sh5.create_dataset('ratio.mass', data = sat_tree['virial.mass']/host_tree['virial.mass'][wh])
                sh5.create_dataset('redshift', data = sat_tree['redshift'])
                sh5.create_dataset('lookback.time', data = sat_tree['lookback'])
                sh5.create_dataset('vel.max', data = sat_tree['vel.max'])
                sh5.create_dataset('rad.max', data = sat_tree['rad.max'])
                sh5.create_dataset('distance', data = sat_tree['distance'])
                sh5.create_dataset('distance:pericenter', data = sat_tree['distance:pericenter'])
                sh5.create_dataset('distance:scaled:evolving', data = sat_tree['distance:scaled:evolving'])
                sh5.create_dataset('distance:scaled:fixed', data = sat_tree['distance:scaled:fixed'])

                sh5.create_dataset('radial.velocity:Hubble', data = sat_tree['radial.velocity:Hubble'])
                sh5.create_dataset('total.velocity:Hubble', data = sat_tree['total.velocity:Hubble'])
                sh5.create_dataset('tangential.velocity:Hubble', data = sat_tree['tangential.velocity:Hubble'])

                sh5.create_dataset('radial.velocity:noHubble', data = sat_tree['radial.velocity:noHubble'])
                sh5.create_dataset('total.velocity:noHubble', data = sat_tree['total.velocity:noHubble'])
                sh5.create_dataset('tangential.velocity:noHubble', data = sat_tree['tangential.velocity:noHubble'])

                sh5.create_dataset('velocity:scaled:evolving', data = sat_tree['velocity:scaled:evolving'])
                sh5.create_dataset('velocity:scaled:fixed', data = sat_tree['velocity:scaled:fixed'])

                for i in range(nalist.shape[0]):
                    strin = nastring[i]
                    sh5.create_dataset(nastring[i], data=sat_tree[nastring[i]])

                sh5.create_dataset('infall.index:mass', data=sat_tree['infall.index:mass'])
                sh5.create_dataset('infall.index:vmax', data=sat_tree['infall.index:vmax'])

                sh5.create_dataset('infall:first.crossing:evolving/redshift', data=infall_first1['redshift'])
                sh5.create_dataset('infall:first.crossing:evolving/lookback', data=infall_first1['lookback'])
                sh5.create_dataset('infall:first.crossing:evolving/distance', data=infall_first1['distance'])
                sh5.create_dataset('infall:first.crossing:evolving/total.velocity:noHubble', data=infall_first1['total.velocity:noHubble'])
                sh5.create_dataset('infall:first.crossing:evolving/radial.velocity:noHubble', data=infall_first1['radial.velocity:noHubble'])
                sh5.create_dataset('infall:first.crossing:evolving/tangential.velocity:noHubble', data=infall_first1['tangential.velocity:noHubble'])
                sh5.create_dataset('infall:first.crossing:evolving/total.velocity:Hubble', data=infall_first1['total.velocity:Hubble'])
                sh5.create_dataset('infall:first.crossing:evolving/radial.velocity:Hubble', data=infall_first1['radial.velocity:Hubble'])
                sh5.create_dataset('infall:first.crossing:evolving/tangential.velocity:Hubble', data=infall_first1['tangential.velocity:Hubble'])
                #sh5.create_dataset('infall:first.crossing:evolving/space.velocity',data=infall_first1['space.velocity'])
                #sh5.create_dataset('infall:first.crossing:evolving/infall.angle',data=infall_first1['infall.angle'])
                sh5.create_dataset('infall:first.crossing:evolving/norm:virial.radius', data=infall_first1['norm:virial.radius'])
                sh5.create_dataset('infall:first.crossing:evolving/norm:virial.velocity', data=infall_first1['norm:virial.velocity'])

                sh5.create_dataset('infall:last.crossing:evolving/redshift', data=infall_last1['redshift'])
                sh5.create_dataset('infall:last.crossing:evolving/lookback', data=infall_last1['lookback'])
                sh5.create_dataset('infall:last.crossing:evolving/distance', data=infall_last1['distance'])
                sh5.create_dataset('infall:last.crossing:evolving/total.velocity:noHubble', data=infall_last1['total.velocity:noHubble'])
                sh5.create_dataset('infall:last.crossing:evolving/radial.velocity:noHubble', data=infall_last1['radial.velocity:noHubble'])
                sh5.create_dataset('infall:last.crossing:evolving/tangential.velocity:noHubble', data=infall_last1['tangential.velocity:noHubble'])
                sh5.create_dataset('infall:last.crossing:evolving/total.velocity:Hubble', data=infall_last1['total.velocity:Hubble'])
                sh5.create_dataset('infall:last.crossing:evolving/radial.velocity:Hubble', data=infall_last1['radial.velocity:Hubble'])
                sh5.create_dataset('infall:last.crossing:evolving/tangential.velocity:Hubble', data=infall_last1['tangential.velocity:Hubble'])
                #sh5.create_dataset('infall:last.crossing:evolving/space.velocity', data=infall_last1['space.velocity'])
                #sh5.create_dataset('infall:last.crossing:evolving/infall.angle', data=infall_last1['infall.angle'])
                sh5.create_dataset('infall:last.crossing:evolving/norm:virial.radius', data=infall_last1['norm:virial.radius'])
                sh5.create_dataset('infall:last.crossing:evolving/norm:virial.velocity', data=infall_last1['norm:virial.velocity'])


        h5['sat.tree'].create_dataset('sat.id:list', data=np.array(ftree_id))

        h5.close()
        print(f'!!! finished simulation {hid} !!!')

if __name__ == "__main__":
    main()
