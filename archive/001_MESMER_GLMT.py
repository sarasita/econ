
#%%

import sys
from pathlib import Path
sys.path.append('/Users/schoens/Documents/Projekte/Econ/Code/v3/')

import xarray as xr
import numpy as np

import config.settings as cset

#%%

if __name__ == '__main__':
    for scenario in cset.scenarios: 
        # load data 
        tas          = xr.load_dataset(cset.path_MESMER / f'{scenario}.nc')
        # spatial aggregation
        weights      = np.cos(np.deg2rad(tas.lat))
        weights.name = "weights"
        tas_fldmean  = tas.weighted(weights).mean(("lon", "lat"))
        # storing 
        #     - if output path does not exist yet, make path 
        cset.path_MESMER_fldmean.mkdir(parents=True, exist_ok=True)
        #     - store 
        tas_fldmean.to_netcdf(cset.path_MESMER_fldmean / f"{scenario}_fldmean.nc")
        
        
# %%
