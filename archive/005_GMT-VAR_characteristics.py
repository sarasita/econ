import sys
sys.path.append('/Users/schoens/Documents/Projekte/Econ/Code/v3/')

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import config.settings as cset

# combine all this into a single dataframe of key GLMT characteristics 

if __name__ == '__main__': 
    glmt_df = pd.read_csv(cset.path_MESMER_char / 'glmt_dataset.csv', index_col = 0)
    gmt     = 0.72591606*glmt_df.values.T+0.09085411
    columns = glmt_df.columns
    del glmt_df
    
    cset.path_MESMER_char.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gmt.T, index = np.linspace(cset.year_start,cset.year_stop,cset.n_years), columns = columns).to_csv(cset.path_MESMER_char / f'gmt-var_dataset.csv')

    # computing characteristics from gmt
    # ntwr, max, eoc, soc, cum, exc, uxc 
    gmt_ntwr = np.diff(gmt[:, 4:26], axis = 1).mean(axis=1) # mean warming rate between 2020 and 2060
    # gmt_max  = np.sort(gmt, axis = 1)[:,-5:].mean(axis = 1)
    gmt_max  = np.sort(gmt, axis = 1)[:, -5:].mean(axis = 1)
    gmt_eoc  = gmt[:,-5:].mean(axis = 1)
    gmt_soc  = gmt[:, :5].mean(axis = 1)
    gmt_cum  = gmt.sum(axis = 1)/86
    gmt_cum_adj = (gmt-np.mean(gmt[:, 0])).sum(axis = 1)/86

    gmt_tmp  = gmt-(np.ones_like(gmt).T*gmt_eoc).T
    gmt_tmp[gmt_tmp < 0] = 0
    gmt_od   = gmt_tmp.sum(axis = 1) 
    
    gmt_tmp  = gmt-(np.ones_like(gmt).T*gmt_eoc).T
    gmt_tmp[gmt_tmp > 0] = 0
    gmt_ud   = gmt_tmp.sum(axis = 1) 
        
    # exc & uxc require a threshold that we set them relative to; 
    # use present-day gmt (2020) as reference level: 
    # year_threshold = 2024
    # i_year         = year_threshold-cset.year_start
    # gmt_threshold = np.round(gmt[:, i_year].mean(),2)
    for gmt_threshold in cset.gmt_thresholds:
        # gmt threshold ~14.62 approx 1.3°C od warming 
        gmt_tmp       = (gmt-gmt_threshold).copy()
        gmt_tmp[gmt_tmp < 0] = 0
        gmt_exc       = gmt_tmp.sum(axis = 1)/86
        # gmt_tmp       = gmt.copy()
        # gmt_tmp[gmt_tmp < gmt_threshold] = 0
        # gmt_exc       = gmt_tmp.sum(axis = 1)/86
        gmt_tmp       = (gmt-gmt_threshold).copy()
        gmt_tmp[gmt_tmp > 0] = 0
        gmt_uxc       = -gmt_tmp.sum(axis = 1)/86
        # 
        gmt_tmp       = (gmt).copy()
        gmt_tmp[gmt_tmp < gmt_threshold] = 0
        gmt_exc_tot   = gmt_tmp.sum(axis = 1)/86
        # gmt_tmp       = gmt.copy()
        # gmt_tmp[gmt_tmp < gmt_threshold] = 0
        # gmt_exc       = gmt_tmp.sum(axis = 1)/86
        gmt_tmp       = (gmt).copy()
        gmt_tmp[gmt_tmp > gmt_threshold] = 0
        gmt_uxc_tot   = gmt_tmp.sum(axis = 1)/86
        # gmt_tmp       = gmt.copy()
        # gmt_tmp[gmt_tmp > gmt_threshold] = 0
        # gmt_uxc       = gmt_tmp.sum(axis = 1)/86
        binary = gmt.copy()
        binary[binary <= gmt_threshold] = 0
        binary[binary > gmt_threshold]  = 1
        gmt_frac_os   = binary.sum(axis = 1)/86

        # generating a pandas dataframe with all the key temperature charactersitics 
        gmt_char_df   = pd.DataFrame(data    = np.array([gmt_ntwr, gmt_max, gmt_eoc, gmt_soc, gmt_cum_adj, gmt_od, gmt_exc, gmt_uxc, gmt_frac_os, gmt_exc_tot, gmt_uxc_tot,  gmt_ud]).T, 
                                    index   = columns, 
                                    columns = ['gmt_ntwr', 'gmt_max', 'gmt_eoc', 'gmt_soc', 'gmt_cum', 'gmt_od', 'gmt_exc', 'gmt_uxc', 'gmt_frac_os', 'gmt_exc_tot', 'gmt_uxc_tot', 'gmt_ud'])
        
        cset.path_MESMER_char.mkdir(parents=True, exist_ok=True)
        gmt_char_df.to_csv(cset.path_MESMER_char / f'gmt-var_characteristics_thsld_{int(gmt_threshold*100)}.csv')
