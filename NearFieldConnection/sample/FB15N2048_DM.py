#!/usr/bin/env

# ---------------------------------------------------------------------------

def main() -> None:
    sample
    snapInfo

# ---------------------------------------------------------------------------

simPath = "/data18/brunello/aalazar/FIREbox/FB15N2048_DM" 

snap103 = {
            'scale.factor': 1.0 / (1.0 + 0.0),
            'halo.catalog.path': f'{simPath}/halo/rockstar_dm/catalog/out_103.list',
            'halo.particle.path': f'{simPath}/halo/rockstar_dm/particles/snapdir_103'
            }

snapInfo = dict()
snapInfo['103'] = snap103

if __name__ == "__main__":
    pass
