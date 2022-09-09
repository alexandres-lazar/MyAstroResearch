#!/usr/bin/env python3

# ---------------------------------------------------------------------------

def main() -> None:
    sample
    snapInfo

# ---------------------------------------------------------------------------

simPath = "/data18/brunello/aalazar/FIREbox/FB15N2048" 

snap103 = {
            'redshift': 0.0,
            'scale.factor': 1.0 / (1.0 + 0.0),
            'halo.catalog.path': f'{simPath}/halo/AHF/catalog/103/FB15N2048_DM.z0.000.AHF_halos',
            'halo.particle.path': f'{simPath}/halo/AHF/particles/snapdir_103/'
            }


snapInfo = dict()
snapInfo['103'] = snap103

if __name__ == "__main__":
    pass
