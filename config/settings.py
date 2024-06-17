
#%%

from pathlib import Path 


#%%

#SETTINGS
scenarios           = ['ssp119', 'SP', 'Neg', 'LD', 'Ren', 'Ref', 'GS', 'ssp534-over', 'ModAct', 'CurPol']
n_scenarios         = len(scenarios)

gdp_target_year     = 2100
gdp_ref_year        = 2015
gdp_reflevel_year   = 2020 # temp level to use as ref; 

year_start          = 2015
year_stop           = 2100  
n_years             = 86

# INPUT PATHS 
path_PROJECT        = Path("/Users/schoens/Documents/Projekte/Econ/")
path_MESMER         = Path.joinpath(path_PROJECT, "Data/MESMER/")
path_FAIR           = Path.joinpath(path_PROJECT, "Data/FaIR_v2/")
path_GDP            = Path.joinpath(path_PROJECT, "Data/BHM/")

# OUTPUT PATHS
path_MESMER_fldmean = Path.joinpath(path_PROJECT, "Data/MESMER/fldmean/")
path_MESMER_char    = Path.joinpath(path_PROJECT, "Data/MESMER/characteristics/")

# Analysis Results
path_CHAR_results   =  Path.joinpath(path_PROJECT, "Data/Results/Key_Characteristics/")

