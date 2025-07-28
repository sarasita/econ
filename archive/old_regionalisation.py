

#%%

def prepare_plot_data(effect_lists, damage_types, category_label, n_bootstrap=1000):
    plot_data = []
    for df, damage_type in zip(effect_lists, damage_types):
        for country in df.index:
            samples = df.loc[country].dropna().values  # All samples for this country
            if len(samples) == 0:
                continue  # Skip if no data

            # Bootstrap: resample with replacement and compute medians
            boot_medians = [
                np.median(np.random.choice(samples, size=len(samples), replace=True))
                for _ in range(n_bootstrap)
            ]

            # Optional: compute central tendency and CIs
            median_of_medians = np.median(boot_medians)
            lower_ci = np.percentile(boot_medians, 2.5)
            upper_ci = np.percentile(boot_medians, 97.5)

            plot_data.append({
                'Country': country,
                'Median Effect': median_of_medians,
                'Lower CI': lower_ci,
                'Upper CI': upper_ci,
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


# Assume abs_df has 'Median Effect', 'Lower CI', 'Upper CI'
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=abs_df,
    x='Damage Type',
    y='Median Effect',
    hue='Category',
    palette=palette
)

# Overlay error bars for each point
for i, row in abs_df.iterrows():
    x_pos = list(cset.GDP_labels).index(row['Damage Type'])  # X position
    if row['Category'] == 'variability only':
        x_pos += 0.2  # Offset to match hue (tune based on actual plot)
    else:
        x_pos -= 0.2
    plt.errorbar(
        x=x_pos,
        y=row['Median Effect'],
        yerr=[[row['Median Effect'] - row['Lower CI']], [row['Upper CI'] - row['Median Effect']]],
        fmt='none',
        ecolor='black',
        alpha=0.5,
        capsize=3
    )

plt.title('Bootstrapped Median Effects with Uncertainty')
plt.tight_layout()
plt.show()


#%%


gdp_variability_effects = []
for scenario_idx, gsat_pairs in enumerate(gsat_pairs_by_scenario):
    scenario = cset.scenarios[scenario_idx]
    gdp_subset = gdp_df.copy()

    for i, j in gsat_pairs:
        traj_i = gdp_subset.iloc[:, i]
        traj_j = gdp_subset.iloc[:, j]
        diff   = np.abs(np.median(traj_i - traj_j))
        gdp_variability_effects.append(diff)

plt.figure()
plt.hist(np.array(gdp_variability_effects) * 100)
plt.title("GDP Impact Variability (same GMT + emulator)")
plt.xlabel("Percent GDP difference due to variability")
plt.show()

# === Analyze regionalization effect (same GMT, different emulator patterns) ===
trajectory_names = gmt_ds.columns.tolist()
regionalization_effects = []

for group in gmt_groups:
    if group[0] <= 100:  # sanity check on index range
        traj_names = [trajectory_names[i] for i in group]
        gdp_subset = gdp_df[traj_names]
        median_gdp = gdp_subset.median(axis=1)
        abs_deviation = (gdp_subset.subtract(median_gdp, axis=0)).abs()
        regionalization_effects.append(abs_deviation.values.mean())

plt.figure()
plt.hist(np.array(regionalization_effects) * 100)
plt.title("GDP Impact Deviation (regionalization effect)")
plt.xlabel("Mean absolute GDP deviation (%)")
plt.show()

# === Optional: plot single exact match comparison ===
example_gsat_pair = gsat_pairs_by_scenario[-1]
gdp_curpol = gdp_df.copy()
i, j = example_gsat_pair[2]

plt.figure()
plt.hist(gdp_curpol.iloc[:, i], alpha=0.7, label=f'Traj {i}', color='C0')
plt.axvline(gdp_curpol.iloc[:, i].median(), color='C0', linestyle='--')
plt.hist(gdp_curpol.iloc[:, j], alpha=0.5, label=f'Traj {j}', color='C1')
plt.axvline(gdp_curpol.iloc[:, j].median(), color='C1', linestyle='--')
plt.legend()
plt.title("Example GSAT Trajectory Pair (CurPol)")
plt.xlabel("GDP Impact")
plt.show()







def merge_overlapping_pairs(pairs):
    """Merge overlapping index pairs into groups."""
    merged = True
    grouped = [list(p) for p in pairs]
    while merged:
        merged = False
        new_grouped = []
        while grouped:
            first, *rest = grouped
            first = set(first)
            rest2 = []
            for group in rest:
                if first & set(group):
                    first |= set(group)
                    merged = True
                else:
                    rest2.append(group)
            new_grouped.append(list(first))
            grouped = rest2
        grouped = new_grouped
    return grouped

gmt_groups = merge_overlapping_pairs(gmt_pairs.tolist())

# === Load GSAT trajectories (with natural variability) for each scenario ===
gsat_dfs = []
gsat_pairs_by_scenario = []

for i_scen, scenario in enumerate(cset.scenarios):
    da = xr.load_dataarray(cset.path_MESMER_fldmean / f'{scenario}_fldmean.nc')
    df = da.to_pandas().T
    gsat_ids = [f.split('_')[0] + '_' + f.split('_')[1] for f in df.columns]

    # Identify GSAT trajectories from same GMT and same emulator (regionalization effect)
    id_to_indices = defaultdict(list)
    for idx, ident in enumerate(gsat_ids):
        id_to_indices[ident].append(idx + i_scen*100)

    scenario_gsat_pairs = []
    for indices in id_to_indices.values():
        if len(indices) > 1:
            scenario_gsat_pairs.extend(list(combinations(indices, 2)))

    gsat_pairs_by_scenario.append(scenario_gsat_pairs)

    # Rename GSAT columns for uniqueness
    df.columns = [f"{scenario}_{i+1}" for i in range(df.shape[1])]
    gsat_dfs.append(df.loc[2015:])

# Concatenate all GSAT data
gsat_df = pd.concat(gsat_dfs, axis=1).sort_index()

# === Optional: Visualize example trajectory group ===
plt.figure()
plt.plot(gmt_ds.iloc[:, gmt_groups[0]])
plt.title("GMT: Group of Similar Trajectories")
plt.show()

plt.figure()
plt.plot(gsat_df.iloc[:, gmt_groups[0]])
plt.title("GSAT: Corresponding Trajectories (Variability)")
plt.show()


# === Wilcoxon Test and Summary Plotting ===
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel

def difference_distribution(pair_lists, gdp_df):
    distribution_vals = []
    for scenario_idx, pair_list in enumerate(pair_lists):
        gdp_subset = 1-gdp_df.copy()
        for i, j in pair_list:
            traj_i = gdp_subset.iloc[:, i]
            traj_j = gdp_subset.iloc[:, j]
            diffs = (traj_i - traj_j)*100
            #/((traj_i+traj_j)/2)*100
            distribution_vals.append(np.abs(diffs))
    return np.array(distribution_vals).flatten()

gsat_distribution = difference_distribution(gsat_pairs_by_scenario, gdp_dfs[1])

gsat_distribution_grouped = gsat_distribution.reshape(-1, 157)[:, :]
# print(np.mean(gsat_distribution_grouped, axis = 0).min(), )
print(np.quantile(np.mean(gsat_distribution_grouped, axis = 0),  q = [0.05, 0.5, 0.9]))
# print(np.mean(gsat_distribution_grouped, axis = 0).max(), )

plt.figure()
plt.hist(np.mean(gsat_distribution_grouped, axis = 0))
plt.show()
plt.figure()
plt.hist(np.std(gsat_distribution_grouped, axis = 0))
plt.show()

gmt_distribution = difference_distribution([gmt_pairs], gdp_dfs[1])
gmt_distribution_grouped = gmt_distribution.reshape(-1, 157)[:, :]
print(np.quantile(np.mean(gmt_distribution_grouped, axis = 0),  q = [0.05, 0.5, 0.9]))
plt.figure()
plt.hist(np.mean(gmt_distribution_grouped, axis = 0))
plt.show()
plt.figure()
plt.hist(np.std(gmt_distribution_grouped, axis = 0))
plt.show()

plt.figure()
plt.hist(np.log(np.mean(gsat_distribution_grouped, axis = 0)))
plt.hist(np.log(np.mean(gmt_distribution_grouped, axis = 0)), alpha = .5)
plt.show()


def t_test_for_pairs(pair_lists, gdp_df, scenario_filter=None):
    results = []
    for scenario_idx, pair_list in enumerate(pair_lists):
        gdp_subset = gdp_df.copy()
        if scenario_filter is not None:
            gdp_subset = gdp_df.loc[:, gdp_df.columns.str.contains(scenario_filter)].copy()

        for i, j in pair_list:
            traj_i = gdp_subset.iloc[:, i]
            traj_j = gdp_subset.iloc[:, j]
            diffs = traj_i - traj_j
            stat, pval = ttest_rel(traj_i, traj_j, alternative='two-sided')
            median_diff = np.median(diffs)
            mean_magnitude = np.mean(np.abs([1 - traj_i.median(), 1 - traj_j.median()]))
            rel_effect_size = 100 * np.abs(median_diff) / mean_magnitude if mean_magnitude != 0 else np.nan
            results.append({
                'pair': (i, j),
                'statistic': stat,
                'p_value': pval,
                'median_relative_effect_pct': rel_effect_size,
                'scenario_idx': scenario_idx
            })
    return results

def wilcoxon_test_for_pairs(pair_lists, gdp_df, scenario_filter=None):
    results = []
    for scenario_idx, pair_list in enumerate(pair_lists):
        gdp_subset = gdp_df.copy()

        for i, j in pair_list:
            traj_i = gdp_subset.iloc[:, i]
            traj_j = gdp_subset.iloc[:, j]
            diffs = traj_i - traj_j
            stat, pval = wilcoxon(diffs, alternative='two-sided')
            median_diff = np.median(diffs)
            mean_magnitude = np.mean(np.abs([1 - traj_i.median(), 1 - traj_j.median()]))
            rel_effect_size = 100 * np.abs(median_diff) / mean_magnitude if mean_magnitude != 0 else np.nan
            results.append({
                'pair': (i, j),
                'statistic': stat,
                'p_value': pval,
                'median_relative_effect_pct': rel_effect_size,
                'scenario_idx': scenario_idx
            })
    return results

def summarize_and_plot(results, title_prefix):
    pvals = [r['p_value'] for r in results]
    effects = [r['median_relative_effect_pct'] for r in results]
    plt.figure()
    plt.hist(effects, bins=30, color='skyblue', edgecolor='black')
    plt.title(f"{title_prefix} - Distribution of Median Relative GDP Differences (%)")
    plt.xlabel("Median Relative GDP Difference (%)")
    plt.ylabel("Count")
    plt.show()
    sig_count = sum(p < 0.05 for p in pvals)
    print(f"{title_prefix}: {sig_count} out of {len(pvals)} pairs show significant differences (p < 0.05)")

# === Run Tests ===
variability_results = wilcoxon_test_for_pairs(gsat_pairs_by_scenario, gdp_dfs[1])
regionalisation_results = wilcoxon_test_for_pairs([gmt_pairs], gdp_dfs[1])  # note: wrapped in list for consistency

# variability_results = t_test_for_pairs(gsat_pairs_by_scenario, gdp_dfs[1])
# regionalisation_results = t_test_for_pairs([gmt_pairs], gdp_dfs[1])  # note: wrapped in list for consistency

# === Plot Summaries ===
summarize_and_plot(variability_results, "Variability Effect")
summarize_and_plot(regionalisation_results, "Regionalisation Effect")

# === Helper: Get GMT slopes for all trajectory indices ===
gmt_ntwr_all = gmt_char_df['gmt_ntwr'].values  # indexed same as GMT/GSAT/GDP trajs

# === PLOT 1: Regionalisation Scatter (gmt_pairs) ===
x_vals = []
y_vals = []
colors = []

for result in regionalisation_results:
    i, j = result['pair']
    diff = np.abs(gdp_df.iloc[:, i].median() - gdp_df.iloc[:, j].median()) * 100  # percentage point diff
    avg_gmt_slope = np.mean([gmt_ntwr_all[i], gmt_ntwr_all[j]])
    x_vals.append(avg_gmt_slope)
    y_vals.append(diff)
    colors.append('red' if result['p_value'] < 0.05 else 'gray')

plt.figure(figsize=(6, 4))
plt.scatter(x_vals, y_vals, c=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
plt.xlabel("GMT slope (°C per decade)")
plt.ylabel("Median GDP difference (%)")
plt.title("Regionalisation Effect: GDP Difference vs GMT Slope")
plt.grid(True)
plt.show()

# === PLOT 2: Variability Scatter (gsat_pairs) ===
x_vals = []
y_vals = []
colors = []

for result in variability_results:
    i, j = result['pair']
    diff = np.abs(gdp_df.iloc[:, i].median() - gdp_df.iloc[:, j].median()) * 100  # percentage point diff
    avg_gmt_slope = np.mean([gmt_ntwr_all[i], gmt_ntwr_all[j]])
    x_vals.append(avg_gmt_slope)
    y_vals.append(diff)
    colors.append('blue' if result['p_value'] < 0.05 else 'gray')

plt.figure(figsize=(6, 4))
plt.scatter(x_vals, y_vals, c=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
plt.xlabel("GMT slope (°C per decade)")
plt.ylabel("Median GDP difference (%)")
plt.title("Variability Effect: GDP Difference vs GMT Slope")
plt.grid(True)
plt.show()

# === Helper: GMT slopes and scenario names ===
gmt_ntwr_all = gmt_char_df['gmt_ntwr'].values
scenario_names_all = gdp_df.columns  # these are the same as gmt_char_df.index

# === Helper: Get scenario name from index ===
def get_scenario_name(index):
    return scenario_names_all[index].split('_')[0]

# === PLOT 1: Regionalisation Effect ===
plt.figure(figsize=(6, 4))

for result in regionalisation_results:
    i, j = result['pair']
    pval = result['p_value']
    diff = np.abs(gdp_df.iloc[:, i].median() - gdp_df.iloc[:, j].median()) * 100
    avg_gmt = np.mean([gmt_ntwr_all[i], gmt_ntwr_all[j]])

    scenario_name = get_scenario_name(i)
    color = cset.scenarios_color_dict[scenario_name]
    marker = 'o' if pval < 0.05 else 'X'

    plt.scatter(avg_gmt, diff, c=color, marker=marker, edgecolor='none', linewidth=0.5, s=60, alpha=0.8)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Significant difference (p < 0.05)',
           markerfacecolor='gray', markeredgecolor='black', markersize=8),
    Line2D([0], [0], marker='x', color='gray', label='Difference not significant (p ≥ 0.05)',
           markersize=8, linestyle='None')
]

plt.legend(handles=legend_elements, title='Test Result', loc='upper right', fontsize='small')
plt.xlabel("GMT near-term warming rate (°C per decade)")
plt.ylabel("Median GDP difference (%)")
plt.title("Effect of Regionalisation + Variabiltiy on GDP")
plt.grid(True)
plt.show()


# === PLOT 2: Variability Effect ===
plt.figure(figsize=(6, 4))

for result in variability_results:
    i, j = result['pair']
    pval = result['p_value']
    diff = np.abs(gdp_df.iloc[:, i].median() - gdp_df.iloc[:, j].median()) * 100
    avg_gmt = np.mean([gmt_ntwr_all[i], gmt_ntwr_all[j]])

    scenario_name = get_scenario_name(i)
    print(scenario_name)
    color = cset.scenarios_color_dict[scenario_name]
    marker = 'o' if pval < 0.05 else 'X'

    plt.scatter(avg_gmt, diff, c=color, marker=marker, edgecolor='none', linewidth=0.5, s=60, alpha=0.8)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Significant difference (p < 0.05)',
           markerfacecolor='gray', markeredgecolor='black', markersize=8),
    Line2D([0], [0], marker='x', color='gray', label='Difference not significant (p ≥ 0.05)',
           markersize=8, linestyle='None')
]

plt.legend(handles=legend_elements, title='Test Result', loc='upper right', fontsize='small')
plt.xlabel("GMT long-term warming rate (°C per decade)")
plt.ylabel("Median GDP difference (%)")
plt.title("Variability Effect Only")
plt.grid(True)
plt.show()
# %%


#%%

df = gdp_long.copy()
df['scenario'] = df['scenario'].astype('category')
df['gmt_id'] = df['gmt_id'].astype('category')
df['esm_model'] = df['esm_model'].astype('category')

import statsmodels.formula.api as smf

# model = smf.mixedlm(
#     "gdp ~ scenario + gmt_ntwr",                  # Fixed effects
#     df,
#     groups="gmt_id",                              # Random intercept per trajectory
#     re_formula="~esm_model"                       # Random slope for regionalisation within trajectories
# )
# result = model.fit()
# print(result.summary())

# # Group by scenario + gmt_id + esm_model
# grouped = df.groupby(['scenario', 'gmt_id', 'esm_model'])

# # Compute within-group variance (internal variability)
# within_var = grouped['gdp'].var().mean()
# print(f"Average internal variability: {within_var:.4f}")

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

model_anova = ols("gdp ~ C(scenario) + gmt_ntwr + C(gmt_id) + C(esm_model)", data=df).fit()
anova_table = anova_lm(model_anova)
print(anova_table)

import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(
    data=df,
    x='gmt_id',
    y='gdp',
    hue='esm_model'
)
plt.xticks(rotation=90)
plt.title("GDP across ESMs within each GMT trajectory")
plt.show()

model = ols("gdp ~ C(scenario) * gmt_ntwr + C(gmt_id) + C(esm_model)", data=df).fit()
anova_table = anova_lm(model)
print(anova_table)

from sklearn.metrics import r2_score

full_model = ols("gdp ~ C(scenario) + C(esm_model)", data=df).fit()
r2_full = full_model.rsquared

model_wo_esm = ols("gdp ~ C(scenario)", data=df).fit()
r2_wo_esm = model_wo_esm.rsquared

esm_contrib = r2_full - r2_wo_esm

model = ols("gdp ~ C(scenario) * gmt_ntwr", data=df).fit()
model.summary()

df['triplet'] = df['scenario'] + '_' + df['gmt_id'].astype(str) + '_' + df['esm_model']

model = ols("gdp ~ C(scenario) + C(gmt_id) + C(esm_model)", data=df).fit()
anova_lm(model)


df.groupby(['scenario', 'gmt_id', 'esm_model'])['gdp'].std().describe()
model = ols("gdp ~ C(scenario) + C(gmt_id) + C(esm_model)", data=df).fit()
df['residuals'] = model.resid
df.groupby(['scenario', 'gmt_id', 'esm_model'])['residuals'].std().describe()

#%%
