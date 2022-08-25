#!/usr/bin/env python3

simPath = '/data17/grenache/aalazar/FIRE/GVD/'

# ---------------------------------------------------------------------------

def main() -> str:
    sample

# ---------------------------------------------------------------------------

sample = list()
sample.append("m10f_res500")
sample.append("m10g_res500")
sample.append("m10h_res500")
sample.append("m10i_res500")
sample.append("m10j_res500")
sample.append("m10k_res500")
sample.append("m10l_res500")
sample.append("m10m_res500")
sample.append("m11d_res7100")
sample.append("m11i_res7100")
sample.append("m11e_res7100")
sample.append("m11h_res7100")

# analysis done at assigned snapshot 
i10 = [int(152), 1.0/(1.0 + 0.203), 0.203]
i11 = [int(486), 1.0/(1.0 + 0.201), 0.201] 

m10f_info = {
            "snapshot.number": i10[0],
            "scale.factor": i10[1],
            "redshift": i10[2],
            "ahf.catalog.path": f"{simPath}/m10f_res500/halo/AHF/catalog/{i10[0]}/ahf.snap_{i10[0]}.z{i10[2]}.AHF_halos",
            "snapshot.path": f"{simPath}/m10f_res500/output/hdf5/"
        }

m10g_info = {
            "snapshot.number": i10[0],
            "scale.factor": i10[1],
            "redshift": i10[2],
            "ahf.catalog.path": f"{simPath}/m10g_res500/halo/AHF/catalog/{i10[0]}/ahf.snap_{i10[0]}.z{i10[2]}.AHF_halos",
            "snapshot.path": f"{simPath}/m10g_res500/output/hdf5/"
        }

m10h_info = {
            "snapshot.number": i10[0],
            "scale.factor": i10[1],
            "redshift": i10[2],
            "ahf.catalog.path": f"{simPath}/m10h_res500/halo/AHF/catalog/{i10[0]}/ahf.snap_{i10[0]}.z{i10[2]}.AHF_halos",
            "snapshot.path": f"{simPath}/m10h_res500/output/hdf5/"
        }

m10i_info = {
            "snapshot.number": i10[0],
            "scale.factor": i10[1],
            "redshift": i10[2],
            "ahf.catalog.path": f"{simPath}/m10i_res500/halo/AHF/catalog/{i10[0]}/ahf.snap_{i10[0]}.z{i10[2]}.AHF_halos",
            "snapshot.path": f"{simPath}/m10i_res500/output/hdf5/"
        }

m10j_info = {
            "snapshot.number": i10[0],
            "scale.factor": i10[1],
            "redshift": i10[2],
            "ahf.catalog.path": f"{simPath}/m10j_res500/halo/AHF/catalog/{i10[0]}/ahf.snap_{i10[0]}.z{i10[2]}.AHF_halos",
            "snapshot.path": f"{simPath}/m10j_res500/output/hdf5/"
        }

m10k_info = {
            "snapshot.number": i10[0],
            "scale.factor": i10[1],
            "redshift": i10[2],
            "ahf.catalog.path": f"{simPath}/m10k_res500/halo/AHF/catalog/{i10[0]}/ahf.snap_{i10[0]}.z{i10[2]}.AHF_halos",
            "snapshot.path": f"{simPath}/m10k_res500/output/hdf5/"
        }

m10l_info = {
            "snapshot.number": i10[0],
            "scale.factor": i10[1],
            "redshift": i10[2],
            "ahf.catalog.path": f"{simPath}/m10l_res500/halo/AHF/catalog/{i10[0]}/ahf.snap_{i10[0]}.z{i10[2]}.AHF_halos",
            "snapshot.path": f"{simPath}/m10l_res500/output/hdf5/"
        }

m10m_info = {
            "snapshot.number": i10[0],
            "scale.factor": i10[1],
            "redshift": i10[2],
            "ahf.catalog.path": f"{simPath}/m10m_res500/halo/AHF/catalog/{i10[0]}/ahf.snap_{i10[0]}.z{i10[2]}.AHF_halos",
            "snapshot.path": f"{simPath}/m10m_res500/output/hdf5/"
        }

m11d_info = {
            "snapshot.number": i11[0],
            "scale.factor": i11[1],
            "redshift": i11[2],
            "ahf.catalog.path": f"{simPath}/m11d_res7100/halo/AHF/catalog/{i11[0]}/AHF.z{i11[2]}.AHF_halos",
            "snapshot.path": f"{simPath}/m11d_res7100/output/hdf5/"
        }

m11e_info = {
            "snapshot.number": i11[0],
            "scale.factor": i11[1],
            "redshift": i11[2],
            "ahf.catalog.path": f"{simPath}/m11e_res7100/halo/AHF/catalog/{i11[0]}/AHF.z{i11[2]}.AHF_halos",
            "snapshot.path": f"{simPath}/m11e_res7100/output/hdf5/"
        }

m11h_info = {
            "snapshot.number": i11[0],
            "scale.factor": i11[1],
            "redshift": i11[2],
            "ahf.catalog.path": f"{simPath}/m11h_res7100/halo/AHF/catalog/{i11[0]}/AHF.z{i11[2]}.AHF_halos",
            "snapshot.path": f"{simPath}/m11h_res7100/output/hdf5/"
        }

m11i_info = {
            "snapshot.number": i11[0],
            "scale.factor": i11[1],
            "redshift": i11[2],
            "ahf.catalog.path": f"{simPath}/m11i_res7100/halo/AHF/catalog/{i11[0]}/AHF.z{i11[2]}.AHF_halos",
            "snapshot.path": f"{simPath}/m11i_res7100/output/hdf5/"
        }

snapInfo = dict()
snapInfo["m10f_res500"] = m10f_info
snapInfo["m10g_res500"] = m10g_info
snapInfo["m10h_res500"] = m10h_info
snapInfo["m10i_res500"] = m10i_info
snapInfo["m10j_res500"] = m10j_info
snapInfo["m10k_res500"] = m10k_info
snapInfo["m10l_res500"] = m10l_info
snapInfo["m10m_res500"] = m10m_info
snapInfo["m11d_res7100"] = m11d_info
snapInfo["m11e_res7100"] = m11e_info
snapInfo["m11h_res7100"] = m11h_info
snapInfo["m11i_res7100"] = m11i_info

# assuming partile coordinates are in cMpc, convert them to ckpc
coord_conv = dict()
coord_conv['m10f_res500'] = 1e3
coord_conv['m10g_res500'] = 1e3
coord_conv['m10h_res500'] = 1e3
coord_conv['m10i_res500'] = 1e3
coord_conv['m10j_res500'] = 1e3
coord_conv['m10k_res500'] = 1e3
coord_conv['m10l_res500'] = 1.0
coord_conv['m10m_res500'] = 1.0
coord_conv['m11d_res7100'] = 1.0
coord_conv['m11i_res7100'] = 1.0
coord_conv['m11e_res7100'] = 1.0
coord_conv['m11h_res7100'] = 1.0

if __name__ == "__main__":
    print("-" * 50)
    print(f"Number of simulations in sample: {len(sample)}")
    print("-" * 50)
    for sim in sample:
        print(sim)
    print("-" * 50)
