
#%%

from pathlib import Path 
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

CODE_DIR = Path(__file__).parent
MAIN_DIR = Path(__file__).parent.parent

#%%

#SETTINGS
scenarios           = ['ssp119', 'SP', 'LD', 'Ref', 'Ren', 'Neg', 'GS', 'ssp534-over', 'ModAct', 'CurPol']
scenario_names       = ['SSP1-1.9', 'SP', 'LD', 'REF', 'REN', 'NEG', 'GS', 'SSP5-3.4-OS', 'ModAct', 'CurPol']
scenario_names_dict  = dict(zip(scenarios, scenario_names))
n_scenarios         = len(scenarios)
focus_scenarios     = ['SP', 'Ref', 'Neg',  'GS', 'ssp534-over', 'CurPol']
focus_scenarios_labels = ['SP', 'REF', 'NEG',  'GS', 'SSP5-3.4-OS', 'CurPol']

scenarios_colors = [
    '#001933',
    '#003366', 
    '#3366cc',
    '#4682B4',  # A more muted, relaxed blue
    '#b0a875',
    np.array([0.859, 0.569, 0.339, 1.0]),  # Slightly muted orange
    np.array([0.825, 0.282, 0.278, 1.0]),  # Slightly muted red
    np.array([0.529, 0.175, 0.359, 1.0]),  # Slightly muted dark pink
    'darkviolet',
    np.array([0.294, 0.137, 0.384, 1.0])    # Keeping the last color the same
]

focus_scenarios_colors = [
    '#003366', 
    '#4682B4',  # A more muted, relaxed blue
    np.array([0.859, 0.569, 0.339, 1.0]),  # Slightly muted orange
    np.array([0.825, 0.282, 0.278, 1.0]),  # Slightly muted red
    np.array([0.529, 0.175, 0.359, 1.0]),  # Slightly muted dark pink
    np.array([0.294, 0.137, 0.384, 1.0])    # Keeping the last color the same
]
focus_scenarios_color_dict = dict(zip(focus_scenarios, focus_scenarios_colors))
scenarios_color_dict = dict(zip(scenarios, scenarios_colors))
n_focus_scenarios   = len(focus_scenarios)

gdp_target_year     = 2100
gdp_ref_year        = 2015
gdp_reflevel_year   = 2020 # temp level to use as ref; 

year_start          = 2015
year_stop           = 2100  
n_years             = 86

# INPUT PATHS 
path_PROJECT        = Path("/Users/schoens/Documents/Projekte/Econ/")
path_DATA           = path_PROJECT / "Data_2410"
path_MESMER         = path_DATA / "MESMER"
path_FAIR           = path_DATA / "FaIR"
path_GDP            = path_DATA / "BHM"
file_burke          = path_GDP / '2410 Full Data BHM Main Result.csv'
file_tas            = path_GDP / '2410_countrylevel_total.csv'

# OUTPUT PATHS
path_MESMER_fldmean = path_MESMER / "fldmean"
path_MESMER_char    = path_MESMER / "characteristics"
path_GRAPHICS       = path_PROJECT / "Graphics" 

# Analysis Results
path_CHAR_results   = path_PROJECT / "Data_2410" / "Results" / "Key_Characteristics"
path_CLUSTER_results= path_PROJECT / "Data_2410" / "Results" / "Clustering"

glmt_thresholds     = [1.67, 1.79, 1.97, 2.00, 2.05, 2.10, 2.30, 2.50]
gmt_thresholds      = [1.2, 1.3, 1.35, 1.4, 1.5]
# n_years_thslds      = [3,7,9.5,11.5,16.5] 
n_years_thslds      = [0]*5 
sel_threshold       = 1.35


# found using model fitting
sel_predictors      = ['tas_soc', 'tas_soc tas_baseline', 'gmt_eoc', 'gmt_eoc tas_baseline', 'gmt_exc', 'gmt_exc tas_baseline'] 

# PLOTTING
labelsize_small  = 15
labelsize_medium = 16
labelsize_large  = 17

# diverging colormap
cmap_diverging = sns.diverging_palette(20, 220, as_cmap=True)
cmap_negref    =  LinearSegmentedColormap.from_list("negref", [focus_scenarios_color_dict['Neg'], 'whitesmoke', focus_scenarios_color_dict['Ref']])
 
# focus countries 
focus_countries        = ['DEU', 'NZL', 'USA', 'CHN', 'BRA', 'IND', 'NGA']
focus_countries_names  = ['Germany', 'New Zealand', 'USA', 'China', 'Brazil', 'India', 'Nigeria']

focus_countries_colors = [
    '#2A9D8F',  # Cool Teal
    '#3CB371',  # Medium Green
    '#8B4513',  # Saddle Brown
    '#E76F51',  # Coral Red
    '#B45B8B',  # Muted Pink
    '#9B59B6',  # Amethyst Purple
    '#6A0C9A'   # Dark Purple
]
# growth clusters
colors_clusters = colors = [np.array([0.09019608, 0.39294118, 0.67058824, 1.]), 
                                   np.array([0.99215686, 0.65647059, 0.3827451, 1.]), 
                                   np.array([0.38117647, 0.25176471, 0.60784314, 1.])
                                   ]


