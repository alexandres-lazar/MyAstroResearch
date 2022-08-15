#!/usr/bin/python3

import sys

if sys.version_info < (3, 6):
    sys.exit("!!! Please use Python 3.6+ to execute script!!!")

sample = {}
sample['m12_elvis_RomeoJuliet_res3500'] = 'm12_ELVIS_RomeoJuliet_tmp_pts_lr.txt'
sample['m12_elvis_ThelmaLouise_res4000'] = 'm12_ELVIS_ThelmaLouise_tmp_pts_lr.txt'
sample['m12_elvis_RomulusRemus_res4000'] = 'm12_ELVIS_ThelmaLouise_tmp_pts_lr.txt'
sample['m12b_res7100'] = 'ic_agora_m12b_ref12_rad5_points.txt'
sample['m12c_res7100'] = 'ic_agora_m12c_ref10_rad7_points.txt'
sample['m12f_res7100'] = 'ic_agora_m12f_ref12_rad5_points.txt'
sample['m12i_res7100'] = 'ic_agora_m12i_ref12_rad4_points.txt'
sample['m12m_res7100'] = 'ic_agora_m12m_ref12_rad5_points.txt'
sample['m12r_res7100'] = 'ic_agora_m12q_ref12_rad6_points.txt'
sample['m12q_res7100'] = 'ic_agora_m12q_ref12_rad6_points.txt'
sample['m12w_res7100'] = 'ic_agora_m12q_ref12_rad6_points.txt'
sample['m12z_res4200'] = 'm12z_lr_pts.txt'

boxLength = {}
boxLength['m12_elvis_RomeoJuliet_res3500'] = 92.48
boxLength['m12_elvis_ThelmaLouise_res4000'] = 50.
boxLength['m12_elvis_RomulusRemus_res4000'] = 50.
boxLength['m12b_res7100'] = 60.
boxLength['m12c_res7100'] = 60.
boxLength['m12f_res7100'] = 60.
boxLength['m12i_res7100'] = 60.
boxLength['m12m_res7100'] = 60.

# ---------------------------------------------------------------------------

def main() -> None:
    return sample

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Dictionary of Convex Hull point files")
