

import sys
sys.path.append('/Users/schoens/Documents/Projekte/Econ/Code/v3/')

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import config.settings as cset

# combine all this into a single dataframe of key GLMT characteristics 

if __name__ == '__main__': 
    gmt       = np.zeros((cset.n_scenarios*100, cset.n_years))
    columns   = []
    for i_scen in range(cset.n_scenarios): 
        scenario      = cset.scenarios[i_scen]
        gmt[i_scen*100:(i_scen+1)*100,:] = pd.read_csv(cset.path_MESMER / 'lfr' / f'lfr_{scenario}.csv', index_col = 0).values[-86:, :].T
        columns.append([scenario + f'_{int(i_run+1)}' for i_run in range(100)])
    columns = np.array(columns).flatten()
    
    cset.path_MESMER_char.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gmt.T, index = np.linspace(cset.year_start,cset.year_stop,cset.n_years), columns = columns).to_csv(cset.path_MESMER_char / f'gmt-lfr_dataset.csv')

    gmt_og       = np.zeros((cset.n_scenarios*100, cset.n_years))
    fair_ids_100  = pd.read_csv(cset.path_FAIR / 'ids_reconstructed.csv', sep = ',', index_col = 0).drop(columns = ['Ref']).rename(columns={'Ref_1p5': 'Ref'})
    for i_scen in range(cset.n_scenarios): 
        scenario = cset.scenarios[i_scen]
        all_fair_runs = pd.read_csv(cset.path_FAIR / f'scen_{scenario}.csv', index_col = 0).iloc[:, fair_ids_100.loc[:, scenario].values]
        all_fair_runs.columns = np.arange(100)
        mesmer_ids = xr.load_dataset(cset.path_MESMER_fldmean / f'{scenario}_fldmean.nc')['fair_esm_variability_realisation'].values
        fair_ids_mesmer = [int(f.split('_')[0]) for f in mesmer_ids]
        gmt_og[i_scen*100:(i_scen+1)*100,:] = all_fair_runs.loc[slice(2015,2100), fair_ids_mesmer].T
        
    p = np.polyfit(gmt.flatten(), gmt_og.flatten(), deg = 1)
    
    gmt_rescaled = gmt*p[0]+p[1]

    del gmt


    # computing characteristics from gmt
    # ntwr, max, eoc, soc, cum, exc, uxc 
    gmt_ntwr = np.diff(gmt_rescaled[:, 4:26], axis = 1).mean(axis=1) # mean warming rate between 2020 and 2060
    # gmt_max  = np.sort(gmt, axis = 1)[:,-5:].mean(axis = 1)
    gmt_max  = np.max(gmt_rescaled, axis = 1)
    gmt_eoc  = gmt_rescaled[:,-1]
    gmt_soc  = gmt_rescaled[:, 0]
    gmt_cum  = gmt_rescaled.sum(axis = 1)/86
    gmt_tmp  = gmt_rescaled-(np.ones_like(gmt_rescaled).T*gmt_eoc).T
    gmt_tmp[gmt_tmp < 0] = 0
    gmt_od   = gmt_tmp.sum(axis = 1) 
    gmt_cum_adj = (gmt_rescaled-np.mean(gmt_rescaled[:, 0])).sum(axis = 1)/86
    # exc & uxc require a threshold that we set them relative to; 
    # use present-day gmt (2020) as reference level: 
    # year_threshold = 2024
    # i_year         = year_threshold-cset.year_start
    # gmt_threshold = np.round(gmt[:, i_year].mean(),2)
    for gmt_threshold in cset.gmt_thresholds:
        # gmt threshold ~14.62 approx 1.3°C od warming 
        gmt_tmp       = (gmt_rescaled-gmt_threshold).copy()
        gmt_tmp[gmt_tmp < 0] = 0
        gmt_exc       = gmt_tmp.sum(axis = 1)/86
        # gmt_tmp       = gmt.copy()
        # gmt_tmp[gmt_tmp < gmt_threshold] = 0
        # gmt_exc       = gmt_tmp.sum(axis = 1)/86
        gmt_tmp       = (gmt_rescaled-gmt_threshold).copy()
        gmt_tmp[gmt_tmp > 0] = 0
        gmt_uxc       = -gmt_tmp.sum(axis = 1)/86
        # gmt_tmp       = gmt.copy()
        # gmt_tmp[gmt_tmp > gmt_threshold] = 0
        # gmt_uxc       = gmt_tmp.sum(axis = 1)/86
        binary = gmt_rescaled.copy()
        binary[binary <= gmt_threshold] = 0
        binary[binary > gmt_threshold]  = 1
        gmt_frac_os   = binary.sum(axis = 1)/86

        # generating a pandas dataframe with all the key temperature charactersitics 
        gmt_char_df   = pd.DataFrame(data    = np.array([gmt_ntwr, gmt_max, gmt_eoc, gmt_soc, gmt_cum_adj, gmt_od, gmt_exc, gmt_uxc, gmt_frac_os]).T, 
                                    index   = columns, 
                                    columns = ['gmt_ntwr', 'gmt_max', 'gmt_eoc', 'gmt_soc', 'gmt_cum', 'gmt_od', 'gmt_exc', 'gmt_uxc', 'gmt_frac_os'])
        
        cset.path_MESMER_char.mkdir(parents=True, exist_ok=True)
        gmt_char_df.to_csv(cset.path_MESMER_char / f'gmt-lfr_characteristics_thsld_{int(gmt_threshold*100)}.csv')
