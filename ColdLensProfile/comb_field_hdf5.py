#!/usr/bin/env python3

# system ----
import os
import sys
import h5py
import numpy as np
 
from sample.FB15N2048 import snapInfo

def main() -> None:
    outPath = '/data17/grenache/aalazar/projects/researchproject_007/output'
    # do computation for these snapshot numbers
    snapList = [103, 352, 348, 344, 240]
    for snapNum in snapList:
        saveName = snapInfo[str(snapNum)]['save.name'] 
        # create new hdf5 file
        with h5py.File(f"{outPath}/{saveName}.hdf5", 'w') as nh5:
            # load in all hdf5 files
            outPath2 = f"{outPath}/field-massbins"
            h1 = h5py.File(f"{outPath2}/{saveName}10-11.hdf5", 'r')
            h2 = h5py.File(f"{outPath2}/{saveName}9-10.hdf5", 'r')
            h3 = h5py.File(f"{outPath2}/{saveName}8-9.hdf5", 'r')
            h4 = h5py.File(f"{outPath2}/{saveName}7-8.hdf5", 'r') 
            # loop through each mass partitioned file and combine all results
            for skey in h1.keys():
                cdata = np.concatenate((h1[skey][:], h2[skey], h3[skey]))
                nh5.create_dataset(skey, data=cdata)
            h1.close(); h2.close(); h3.close(); h4.close()
        print(f"...saved {snapNum}")


if __name__ == "__main__":
    main()
