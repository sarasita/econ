# %%
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations
import config.settings as cset
from matplotlib.lines import Line2D
import seaborn as sns


# goal: disentangle effects from regionalization and natural variability on GDP impacts

### LOAD DATA ### 
# === Load GMT characteristics 
thrshld_str = '135'
gmt_char_df = pd.read_csv(cset.path_MESMER_char / f'gmt_characteristics_thsld_{thrshld_str}.csv', index_col = 0)
gmt_char_df['gmt_ntwr'] *= 10 # convert to °C/decade
gmt_char_df['gmt_ltwr'] *= 10 # convert to °C/decade

# === Load GMT trajectories 
gmt_ds = pd.read_csv(cset.path_MESMER_char / 'gmt_dataset.csv', index_col=0)
gmt = gmt_ds.values.T

# === Load GSAT trajectories 
def load_gsat_as_pd(scenario):
    df = xr.load_dataarray(cset.path_MESMER_fldmean / f'{scenario}_fldmean.nc').sel(year = slice(1850,2100)).to_pandas().T
    df.columns = [f"{scenario}_{i+1}" for i in range(100)]
    df.index = np.arange(1850, 2101)
    return(df)
def load_gsat_ids(scenario):
    df = xr.load_dataarray(cset.path_MESMER_fldmean / f'{scenario}_fldmean.nc').sel(year = slice(1850,2100)).to_pandas().T
    return([scenario + '_' + f.split('_')[0] + '_' + f.split('_')[1] for f in df.columns])
gsat_ds = pd.concat([load_gsat_as_pd(scenario) for scenario in cset.scenarios], axis = 1)
gsat = gsat_ds.values.T
gsat_ids = [f for scenario in cset.scenarios for f in load_gsat_ids(scenario)]

# === Load GDP dataset 
gdp_dfs = [pd.read_csv(cset.path_OUT_PW / file, index_col=0) for file in cset.PW_dataset_names]
gdp_df = gdp_dfs[1]  # pick one for now

### PREPARE DATA ### 
gdp_long = gdp_df.median(axis = 0).reset_index().rename(columns = {0: 'gdp', 'index': 'run_id'})
gmt_ids = []
model_ids = []
run_ids = []
scenario_ids = []
for i_scen, scenario in enumerate(cset.scenarios):
    da = xr.load_dataarray(cset.path_MESMER_fldmean / f'{scenario}_fldmean.nc')
    df = da.to_pandas().T
    gmt_ids += [int(f.split('_')[0])+i_scen*100 for f in df.columns]
    model_ids += [f.split('_')[1] for f in df.columns]
    run_ids += [scenario + '_' + str(i+1) for i in range(100)]
    scenario_ids += [scenario] * 100
    
run_metadata = pd.DataFrame({
    'run_id': run_ids,         # e.g., ['ssp119_1', 'ssp119_2', ..., 'CurPol_100']
    'gmt_id': gmt_ids,         # extract leading number from identifier
    'esm_model': model_ids,      # extract model string from identifier
    'scenario': scenario_ids,  # e.g., ['ssp119', 'ssp119', ..., 'CurPol']
})

gdp_long = gdp_long.merge(run_metadata, on='run_id')
gdp_long['gmt_ntwr'] = gmt_char_df['gmt_ntwr'][gdp_long['run_id'].values].values

### PROCESSING (1): Identify  ### 
# === Find groups of virtually identical GMT trajectories ===
corr_matrix = np.corrcoef(gmt)
np.fill_diagonal(corr_matrix, 0)
corr_threshold = 0.999999
gmt_pairs = np.argwhere((corr_matrix > corr_threshold) & (np.triu(np.ones_like(corr_matrix), k=1) == 1))

# === Find groups of virtually identical GSAT trajectories ===
# Step 1: Group indices by exact full string
entry_to_indices = defaultdict(list)
for idx, entry in enumerate(gsat_ids):
    entry_to_indices[entry].append(idx)
# Step 2: For each group of duplicates, generate index pairs
gsat_pairs = []
for indices in entry_to_indices.values():
    if len(indices) > 1:  # Only if there are actual duplicates
        gsat_pairs.extend([list(pair) for pair in combinations(indices, 2)])

# === Optional: Visualize example trajectory group ===
plt.figure()
plt.plot(gmt_ds.iloc[:, gmt_pairs[0]])
plt.title("GMT: Group of Similar Trajectories")
plt.show()

plt.figure()
plt.plot(gsat_ds.iloc[:, gsat_pairs[0]])
plt.title("GSAT: Corresponding Trajectories (Variability)")
plt.show()

# === Analyze magnitude of regionalisation & variability effect
def generate_differences(gdp_dfs, pairs):
    effects_rel = []
    effects_abs = []
    for gdp_df in gdp_dfs:
        rel_columns = []
        abs_columns = []
        for i, j in pairs:
            gdp_i = gdp_df.iloc[:, i].values
            gdp_j = gdp_df.iloc[:, j].values
            diff_rel = np.abs(gdp_i - gdp_j) / np.abs(1-((gdp_i + gdp_j) / 2))*100 # relative difference
            diff_abs = np.abs(gdp_i - gdp_j)*100 # percentage point difference
            rel_columns.append(diff_rel)
            abs_columns.append(diff_abs)

        # Build DataFrames in one step
        df_rel_tmp = pd.DataFrame(np.column_stack(rel_columns), index=gdp_df.index)
        df_abs_tmp = pd.DataFrame(np.column_stack(abs_columns), index=gdp_df.index)

        effects_rel.append(df_rel_tmp)
        effects_abs.append(df_abs_tmp)

    return effects_abs, effects_rel

gmt_abs_effects, gmt_rel_effects = generate_differences(gdp_dfs, gmt_pairs)
gsat_abs_effects, gsat_rel_effects = generate_differences(gdp_dfs, gsat_pairs)

def prepare_plot_data(effect_lists, damage_types, category_label):
    plot_data = []
    for df, damage_type in zip(effect_lists, damage_types):
        medians = df.median(axis=1)  # Compute country-level medians
        for country, median_value in medians.items():
            plot_data.append({
                'Country': country,
                'Median Effect': median_value,
                'Damage Type': damage_type,
                'Category': category_label
            })
    return pd.DataFrame(plot_data)

abs_df = pd.concat([
    prepare_plot_data(gmt_abs_effects, cset.GDP_labels, 'regionalisation + variability'),
    prepare_plot_data(gsat_abs_effects, cset.GDP_labels, 'variability only')
])
rel_df = pd.concat([
    prepare_plot_data(gmt_rel_effects, cset.GDP_labels, 'regionalisation + variability'),
    prepare_plot_data(gsat_rel_effects, cset.GDP_labels, 'variability only')
])

abs_quantiles = abs_df.groupby(['Damage Type', 'Category'])['Median Effect'].quantile([0.05,0.5,0.95]).unstack().reset_index()
rel_quantiles = rel_df.groupby(['Damage Type', 'Category'])['Median Effect'].quantile([0.05,0.5,0.95]).unstack().reset_index()


palette = {'regionalisation + variability': '#1f77b4', 'variability only': '#ff7f0e'}  # Example: blue for GMT, orange for GSAT

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=rel_df,
    x='Damage Type',
    y='Median Effect',
    hue='Category',
    palette=palette
)
plt.title('Distribution of Country-Level Median Relative Effects')
plt.ylabel('Median Effect')
plt.xlabel('Damage Type')
plt.legend(title='Temperature Metric')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=abs_df,
    x='Damage Type',
    y='Median Effect',
    hue='Category',
    palette=palette
)
plt.title('Distribution of Country-Level Median Absolute Effects')
plt.ylabel('Median Effect')
plt.xlabel('Damage Type')
plt.legend(title='Temperature Metric')
plt.tight_layout()
plt.show()

