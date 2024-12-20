
#%%

import sys
sys.path.append('/Users/schoens/Documents/Projekte/Econ/Code/v3/')

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import config.settings as cset

#%%

# combine all this into a single dataframe of key GLMT characteristics 

if __name__ == '__main__': 
    glmt       = np.zeros((cset.n_scenarios*100, cset.n_years))
    columns    = []
    ref_values = []
    for i_scen in range(cset.n_scenarios): 
        scenario = cset.scenarios[i_scen]
        glmt_tmp = xr.load_dataset(cset.path_MESMER_fldmean / f"{scenario}_fldmean.nc")['tas'].values
        ref_values.append(glmt_tmp[:, :50].mean())
        glmt[i_scen*100:(i_scen+1)*100, :] = glmt_tmp[:, -86:]
        columns.append([scenario + f'_{int(i_run+1)}' for i_run in range(100)])
    glmt    = glmt - np.mean(ref_values)
    columns = np.array(columns).flatten()
    
    cset.path_MESMER_char.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(glmt.T, index = np.linspace(cset.year_start,cset.year_stop,cset.n_years), columns = columns).to_csv(cset.path_MESMER_char / f'glmt_dataset.csv')
    
    # computing characteristics from glmt
    # ntwr, max, eoc, soc, cum, exc, uxc 
    glmt_ntwr = np.diff(glmt[:, 4:26], axis = 1).mean(axis=1) # mean warming rate between 2020 and 2060
    # glmt_max  = glmt.max(axis = 1)
    glmt_max  = np.sort(glmt, axis = 1)[:, -5:].mean(axis = 1) # mean of top 5
    glmt_eoc  = glmt[:, -5:].mean(axis = 1)
    glmt_soc  = glmt[:, :5].mean(axis = 1)
    glmt_cum  = glmt.sum(axis = 1)
    glmt_tmp  = glmt-(np.ones_like(glmt).T*glmt_eoc).T
    glmt_tmp[glmt_tmp < 0] = 0
    glmt_od   = glmt_tmp.sum(axis = 1) 
        
    # exc & uxc require a threshold that we set them relative to; 
    # use present-day glmt (2020) as reference level: 
    # year_threshold = 2024
    # i_year         = year_threshold-cset.year_start
    # glmt_threshold = np.round(glmt[:, i_year].mean(),2)
    for glmt_threshold in cset.glmt_thresholds:
        # glmt threshold ~14.62 approx 1.3°C od warming 
        glmt_tmp       = (glmt-glmt_threshold).copy()
        glmt_tmp[glmt_tmp < 0] = 0
        glmt_exc       = glmt_tmp.sum(axis = 1)  
        glmt_tmp       = (glmt-glmt_threshold).copy()
        glmt_tmp[glmt_tmp > 0] = 0
        glmt_uxc       = -glmt_tmp.sum(axis = 1)  
        
        # generating a pandas dataframe with all the key temperature charactersitics 
        glmt_char_df   = pd.DataFrame(data    = np.array([glmt_ntwr, glmt_max, glmt_eoc, glmt_soc, glmt_cum, glmt_od, glmt_exc, glmt_uxc]).T, 
                                    index   = columns, 
                                    columns = ['glmt_ntwr', 'glmt_max', 'glmt_eoc', 'glmt_soc', 'glmt_cum', 'glmt_od', 'glmt_exc', 'glmt_uxc'])
        
        cset.path_MESMER_char.mkdir(parents=True, exist_ok=True)
        glmt_char_df.to_csv(cset.path_MESMER_char / f'glmt_characteristics_thsld_{int(glmt_threshold*100)}.csv')
        
# %%
