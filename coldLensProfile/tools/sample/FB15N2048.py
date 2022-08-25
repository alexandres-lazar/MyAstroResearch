#!/usr/bin/env

# ---------------------------------------------------------------------------

def main() -> None:
    sample
    snapInfo

# ---------------------------------------------------------------------------

simPath = "/data18/brunello/aalazar/FIREbox/FB15N2048_DM" 

# define the snapshot information here
sample = [103, 352, 348, 344, 240]

snap103 = {
        'scale.factor': 1.0 / (1.0 + 0.0),
        'save.name': 'projection-z0.000_',
        'halo.catalog.path': f'{simPath}/halo/AHF/catalog/103/FB15N2048_DM.z0.000.AHF_halos',
        'halo.particle.path': f'{simPath}/halo/AHF/particles/snapdir_103/'
        }

snap352 = {
        'scale.factor': 1.0 / (1.0 + 0.501),
        'save.name': 'projection-z0.501_',
        'halo.catalog.path': f'{simPath}/halo/AHF/catalog/352/FB15N2048_DM.z0.501.AHF_halos',
        'halo.particle.path': f'{simPath}/halo/AHF/particles/snapdir_352/'
        }

snap348 = {
        'scale.factor': 1.0 / (1.0 + 1.250),
        'save.name': 'projection-z1.250_',
        'halo.catalog.path': f'{simPath}/halo/AHF/catalog/348/FB15N2048_DM.z1.250.AHF_halos',
        'halo.particle.path': f'{simPath}/halo/AHF/particles/snapdir_348/'
        }

snap344 = {
        'scale.factor': 1.0 / (1.0 + 2.000),
        'save.name': 'projection-z2.000_',
        'halo.catalog.path': f'{simPath}/halo/AHF/catalog/344/FB15N2048_DM.z2.000.AHF_halos',
        'halo.particle.path': f'{simPath}/halo/AHF/particles/snapdir_344/'
        }

snap240 = {
        'scale.factor': 1.0 / (1.0 + 3.000),
        'save.name': 'projection-z3.000_',
        'halo.catalog.path': f'{simPath}/halo/AHF/catalog/240/FB15N2048_DM.z3.000.AHF_halos',
        'halo.particle.path': f'{simPath}/halo/AHF/particles/snapdir_240/'
        }

snap176 = {
        'scale.factor': 1.0 / (1.0 + 4.000),
        'save.name': 'projection-z4.000_',
        'halo.catalog.path': f'{simPath}/halo/AHF/catalog/176/FB15N2048_DM.z4.000.AHF_halos',
        'halo.particle.path': f'{simPath}/halo/AHF/particles/snapdir_176/'
        }

snap132 = {
        'scale.factor': 1.0 / (1.0 + 5.058),
        'save.name': 'projection-z5.058_',
        'halo.catalog.path': f'{simPath}/halo/AHF/catalog/132/FB15N2048_DM.z5.058.AHF_halos',
        'halo.particle.path': f'{simPath}/halo/AHF/particles/snapdir_132/'
        }

snapInfo = dict()
snapInfo['103'] = snap103
snapInfo['352'] = snap352
snapInfo['348'] = snap348
snapInfo['344'] = snap344
snapInfo['240'] = snap240
snapInfo['176'] = snap176
snapInfo['132'] = snap132

if __name__ == "__main__":
    print("-" * 50)
    print(f"Number of snapshots to analyze: {len(sample)}")
    print("-" * 50)
    for sim in sample:
        print(f"snapshot {sim}")
    print("-" * 50)

