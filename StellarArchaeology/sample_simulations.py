#!/usr/bin/python3

import sys

if sys.version_info < (3, 6):
    sys.exit("!!! Please use Python 3.6+ to execute script!!!")

sample_list = list()
sample_list.append('m12b_res7100')
#sample_list.append('m12c_res7100')
sample_list.append('m12f_res7100')
sample_list.append('m12i_res7100')
#sample_list.append('m12m_res7100')
sample_list.append('m12r_res7100')
sample_list.append('m12w_res7100')
sample_list.append('m12z_res4200')
sample_list.append('m12_elvis_RomeoJuliet_res3500')
sample_list.append('m12_elvis_ThelmaLouise_res4000')
sample_list.append('m12_elvis_RomulusRemus_res4000')

# ---------------------------------------------------------------------------

def main() -> None:
    return sample_list

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("-" * 50)
    print(f"Number of simulations in sample: {len(sample_list)}")
    print("-" * 50)
    for sim in sample_list:
        print(sim)
    print("-" * 50)
