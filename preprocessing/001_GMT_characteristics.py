import sys
import os
sys.path.append(os.getcwd())

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import config.settings as cset

# generating a single dataframe that contains the gmt characteristics for all scenarios and runs
# this is done by reading the FAIR runs and the MESMER ids, then extracting the gmt values from the FAIR runs
# and combining them with the MESMER ids.
# some metrics are threshold dependent, so we will compute a seperate datafrome for each threshold.

if __name__ == '__main__': 
    gmt       = np.zeros((cset.n_scenarios*100, cset.n_years))
    columns   = []
    model_ids = [] 
    mesmer_ids_all = []
    fair_ids_100  = pd.read_csv(cset.path_FAIR / 'ids_reconstructed.csv', sep = ',', index_col = 0).drop(columns = ['Ref']).rename(columns={'Ref_1p5': 'Ref'})
    for i_scen in range(cset.n_scenarios): 
        scenario = cset.scenarios[i_scen]
        all_fair_runs = pd.read_csv(cset.path_FAIR / f'scen_{scenario}.csv', index_col = 0).iloc[:, fair_ids_100.loc[:, scenario].values]
        all_fair_runs.columns = np.arange(100)
        mesmer_ids = xr.load_dataset(cset.path_MESMER_fldmean / f'{scenario}_fldmean.nc')['fair_esm_variability_realisation'].values
        fair_ids_mesmer = [int(f.split('_')[0]) for f in mesmer_ids]
        gmt[i_scen*100:(i_scen+1)*100,:] = all_fair_runs.loc[slice(2015,2100), fair_ids_mesmer].T
        columns.append([scenario + f'_{int(i_run+1)}' for i_run in range(100)])
        model_ids += [f.split('_')[1] for f in mesmer_ids]
        mesmer_ids_all.append(mesmer_ids)
    columns = np.array(columns).flatten()
    
    cset.path_MESMER_char.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gmt.T, index = np.linspace(cset.year_start,cset.year_stop,cset.n_years), columns = columns).to_csv(cset.path_MESMER_char / f'gmt_dataset.csv')
    pd.DataFrame(np.array(mesmer_ids_all).T, columns = cset.scenarios).to_csv(cset.path_MESMER_char / f'mesmer_ids.csv')

    # gmt characteristics 
    # computing characteristics from gmt
    # ntwr, max, eoc, soc, cum, exc, uxc 
    gmt_tmp  = gmt.copy() - (np.ones_like(gmt).T*gmt[:, 0]).T
    
    # # Warming per decade
    # warming_per_decade = slope * 10
    gmt_ntwr = np.polyfit(np.arange(15), gmt_tmp[:, :15].T, 1)[0]
    gmt_mtwr = np.sum(gmt_tmp[:, 25:50], axis = 1)/25 # mean warming between 2040 and 2065
    gmt_ltwr = np.polyfit(np.arange(86), gmt_tmp[:, :].T, 1)[0] # mean warming between 2015 and 2100
    
    # gmt_max = np.sort(gmt, axis = 1)[:, -5:].mean(axis = 1) # mean of the 5 highest values
    gmt_max = np.max(gmt, axis = 1) # maximum value
    gmt_eoc  = gmt[:,-1]
    gmt_soc  = gmt[:, 0]
    gmt_cum     = gmt.sum(axis = 1)/86
    gmt_cum_adj = (gmt-np.mean(gmt[:, 0])).sum(axis = 1)/86
    
    gmt_tmp     = gmt-(np.ones_like(gmt).T*gmt_eoc).T
    gmt_tmp[gmt_tmp < 0] = 0
    gmt_od   = gmt_tmp.sum(axis = 1) 
    gmt_tmp     = gmt-(np.ones_like(gmt).T*gmt_eoc).T
    gmt_tmp[gmt_tmp > 0] = 0
    gmt_ud   = -gmt_tmp.sum(axis = 1) 
    
    idx_max = np.argmax(gmt, axis = 1)
    gmt_tmp = gmt.copy()
    for i in range(gmt.shape[0]):
        gmt_tmp[i, idx_max[i]:] = gmt_max[i]
    gmt_max_exc = gmt_tmp.sum(axis = 1)/86
    
    gmt_tmp = gmt.copy()
    for i in range(gmt.shape[0]):
        gmt_tmp[i, gmt_tmp[i, :] > gmt_eoc[i]] = gmt_eoc[i]
    gmt_eoc_exc = gmt_tmp.sum(axis = 1)/86

    for i, gmt_threshold in enumerate(cset.gmt_thresholds):
        n_years_thsld = cset.n_years_thslds[i]

        gmt_tmp       = (gmt-gmt_threshold).copy()
        gmt_tmp[gmt_tmp < 0] = 0
        gmt_exc       = gmt_tmp.sum(axis = 1)/(86-n_years_thsld)

        gmt_tmp       = gmt.copy()-1.5
        gmt_tmp[gmt_tmp > 0] = 0
        gmt_uxc       = -gmt_tmp.sum(axis = 1)/(86-n_years_thsld)
        
        binary = gmt.copy()
        binary[binary <= gmt_threshold] = 0
        binary[binary > gmt_threshold]  = 1
        gmt_frac_os   = binary.sum(axis = 1)/(86-n_years_thsld)

        # generating a pandas dataframe with all the key temperature charactersitics 
        gmt_char_df   = pd.DataFrame(data    = np.array([gmt_ntwr, gmt_mtwr, gmt_ltwr, gmt_max, gmt_eoc, gmt_soc, gmt_cum, gmt_cum_adj, gmt_od, gmt_ud, gmt_exc, gmt_uxc, gmt_frac_os, gmt_max_exc, gmt_eoc_exc, model_ids]).T, 
                                    index   = columns, 
                                    columns = ['gmt_ntwr', 'gmt_mtwr', 'gmt_ltwr', 'gmt_max', 'gmt_eoc', 'gmt_soc', 'gmt_cum', 'gmt_cum_adj', 'gmt_od','gmt_ud', 'gmt_exc', 'gmt_uxc', 'gmt_frac_os', 'gmt_max_exc', 'gmt_eoc_exc', 'model_id'])
        
        cset.path_MESMER_char.mkdir(parents=True, exist_ok=True)
        gmt_char_df.to_csv(cset.path_MESMER_char / f'gmt_characteristics_thsld_{int(gmt_threshold*100)}.csv')
