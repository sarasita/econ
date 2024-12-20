import sys
from pathlib import Path
import xarray as xr
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
import config.settings as cset

import matplotlib.pyplot as plt 

path_FAIR   = Path("/Users/schoens/Documents/Projekte/Econ/Data/FaIR_v3")


# find a,b such that gmt = a * glmt + b
result_df = pd.DataFrame(None, index = np.arange(100), columns = cset.scenarios)

# for scenario in cset.scenarios[2:-1]: 
for scenario in ['GS', 'ssp534-over', 'ModAct']:
    all_gmt = pd.read_csv(path_FAIR / f'scen_{scenario}.csv', index_col = 0).loc[slice(1850,2100)].iloc[:, :].values.T
    glmt    = np.load(f'/Users/schoens/Documents/Projekte/Econ/Data/tmp/{scenario}_glmt.npy')
    params   = []
    idx_keep = []
    for i in tqdm(range(100)): 
        rs = []
        for j in range(2237):
            ptmp = np.polyfit(glmt[i, :], all_gmt[j,:], 1)
            rs.append(np.sum((ptmp[0] * glmt[i, :] + ptmp[1] - all_gmt[j,:])**2))
        idx_keep.append(np.argmin(rs))
        params.append(np.polyfit(glmt[i, :], all_gmt[idx_keep[i],:], 1))
    result_df[scenario] = np.array(idx_keep)
    
result_df.to_csv('/Users/schoens/Documents/Projekte/Econ/Data/tmp/ids_reconstructed.csv')

plt.figure()
plt.plot(params[0][0]*glmt[:, :].T + params[0][1], color = 'C0', alpha = .2)
plt.plot(all_gmt[idx_keep[:], :].T, color = 'C1', alpha = .2)
plt.show()



#%%

scenario = 'CurPol'
fair_ids_100  = pd.read_csv(path_FAIR / 'ids_for_sarah.csv', sep = ',', index_col = 0).rename(columns = {'Ref_1p5': 'Ref'})
gmt   = pd.read_csv(path_FAIR / f'scen_{scenario}.csv', index_col = 0).loc[slice(1850,2100)].iloc[:, fair_ids_100.loc[:, scenario].values].values.T


plt.figure()
plt.plot(np.linspace(1850,2100,251), gmt[:5,:].T)
plt.show()

plt.figure()
plt.plot(np.linspace(1850,2100,251), gmt[:,:].T, color = 'C0', alpha = .2)
plt.show()


pd.DataFrame(np.array(idx_keep), index = np.arange(100), columns = ['CurPol']).to_csv('/Users/schoens/Documents/Projekte/Econ/Data/tmp/ids_reconstructed.csv')

plt.figure()
plt.plot(ptmp[0] * glmt[i, :] + ptmp[1], color = 'C0')
plt.plot(all_gmt[j, :], color = 'C1')
plt.show()

plt.figure()
plt.plot(params[0][0]*glmt[:, :].T + params[0][1], color = 'C0', alpha = .2)
plt.plot(all_gmt[idx_keep[:], :].T, color = 'C1', alpha = .2)
plt.show()

plt.figure()
plt.plot(glmt[:5, :].T)
plt.show() 

plt.figure()
plt.plot(all_gmt[idx_keep[:5], :].T) 
plt.show()  


