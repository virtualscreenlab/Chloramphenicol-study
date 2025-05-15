import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, levene, shapiro
from scipy.stats import kruskal
from itertools import combinations

# Updated data with all three groups
data = {
    "Control": [
        5, 6, 9, 5, 8, 6, 6, 7, 8, 9, 6, 6, 8, 8, 7, 7, 6, 8, 9, 8, 5, 5, 6, 7, 8, 7, 6, 9, 7, 9,
        7, 8, 6, 6, 6, 8, 8, 7, 8, 7, 7, 8, 7, 6, 8, 9, 6, 7, 8, 7, 5, 8, 6, 9, 9, 6, 8, 7, 7, 7, 5
    ],
    "Prontosan": [
        3, 4, 4, 4, 5, 4, 4, 5, 7, 4, 7, 7, 6, 5, 4, 5, 6, 7, 6, 5, 6, 7, 7, 6, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 6, 6, 6, 6, 7, None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None
    ],
    "Chloramphenicol": [
        3, 4, 3, 4, 3, 4, 4, 4, 5, 4, 3, 3, 3, 5, 3, 4, 3, 3, 4, 4, 4, 3, 4, 4, 4, 4, 3, 3, 4, 3,
        3, 3, 3, 4, 3, 4, 4, 4, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, None, None, None,
        None, None, None
    ]
}

# Updating the statistical and plotting analysis based on the new data
max_len = max(len(data["Chloramphenicol"]), len(data["Prontosan"]), len(data["Control"]))
data["Chloramphenicol"] += [np.nan] * (max_len - len(data["Chloramphenicol"]))
data["Prontosan"] += [np.nan] * (max_len - len(data["Prontosan"]))
data["Control"] += [np.nan] * (max_len - len(data["Control"]))

# Creating a DataFrame
df = pd.DataFrame(data)

# Calculating frequencies
chloramphenicol_counts = df['Chloramphenicol'].dropna().value_counts(normalize=True) * 100
prontosan_counts = df['Prontosan'].dropna().value_counts(normalize=True) * 100
control_counts = df['Control'].dropna().value_counts(normalize=True) * 100

# Aligning data
days = sorted(set(df['Chloramphenicol'].dropna()).union(set(df['Prontosan'].dropna())))
chloramphenicol_freq = [chloramphenicol_counts.get(day, 0) for day in days]
prontosan_freq = [prontosan_counts.get(day, 0) for day in days]
control_freq = [control_counts.get(day, 0) for day in days]

# Filter out NaN and None values
chloramphenicol_clean = [x for x in data["Chloramphenicol"] if x is not None and not np.isnan(x)]
prontosan_clean = [x for x in data["Prontosan"] if x is not None and not np.isnan(x)]
control_clean = [x for x in data["Control"] if x is not None and not np.isnan(x)]

# Filter out NaN and None values
chloramphenicol_clean = [x for x in data["Chloramphenicol"] if x is not None and not np.isnan(x)]
prontosan_clean = [x for x in data["Prontosan"] if x is not None and not np.isnan(x)]
control_clean = [x for x in data["Control"] if x is not None and not np.isnan(x)]

# Levene's test for variance equality
levene_stat, levene_pval = levene(chloramphenicol_clean, prontosan_clean, control_clean, center='mean')

# Calculate additional statistics
# Mean and standard deviation
mean_chloramphenicol = np.mean(chloramphenicol_clean)
std_chloramphenicol = np.std(chloramphenicol_clean, ddof=1)
mean_prontosan = np.mean(prontosan_clean)
std_prontosan = np.std(prontosan_clean, ddof=1)
mean_control = np.mean(control_clean)
std_control = np.std(control_clean, ddof=1)

# Shapiro-Wilk test for normality
shapiro_chloramphenicol_stat, shapiro_chloramphenicol_pval = shapiro(chloramphenicol_clean)
shapiro_prontosan_stat, shapiro_prontosan_pval = shapiro(prontosan_clean)
shapiro_control_stat, shapiro_control_pval = shapiro(control_clean)

# Re-running Levene's Test and Shapiro-Wilk Test to extract full precision p-values
from scipy.stats import levene, shapiro

# Levene's Test
levene_stat, levene_pval = levene(chloramphenicol_clean, prontosan_clean, control_clean, center='mean')

# Shapiro-Wilk Test
shapiro_chloramphenicol_stat, shapiro_chloramphenicol_pval = shapiro(chloramphenicol_clean)
shapiro_prontosan_stat, shapiro_prontosan_pval = shapiro(prontosan_clean)
shapiro_control_stat, shapiro_control_pval = shapiro(control_clean)

# Mann-Whitney U test
u_stat, u_pval = mannwhitneyu(chloramphenicol_clean, prontosan_clean, alternative='two-sided')
u_stat1, u_pval1 = mannwhitneyu(chloramphenicol_clean, control_clean, alternative='two-sided')
u_stat2, u_pval2 = mannwhitneyu(control_clean, prontosan_clean, alternative='two-sided')

# Print full-precision p-values
print("Additional Statistical Results:")
print(f"Chloramphenicol - Mean: {mean_chloramphenicol:.2f}, Std Dev: {std_chloramphenicol:.2f}")
print(f"Prontosan - Mean: {mean_prontosan:.2f}, Std Dev: {std_prontosan:.2f}")
print(f"Control - Mean: {mean_control:.2f}, Std Dev: {std_control:.2f}")

print(f"Levene's Test: Statistic = {levene_stat:.6f}, P-value = {levene_pval:.10f}")
print(f"Shapiro-Wilk Test for Chloramphenicol: Statistic = {shapiro_chloramphenicol_stat:.6f}, P-value = {shapiro_chloramphenicol_pval:.10f}")
print(f"Shapiro-Wilk Test for Prontosan: Statistic = {shapiro_prontosan_stat:.6f}, P-value = {shapiro_prontosan_pval:.10f}")
print(f"Shapiro-Wilk Test for Control: Statistic = {shapiro_control_stat:.6f}, P-value = {shapiro_control_pval:.10f}")

# Print results with full precision
print(f"Mann-Whitney U Test: U-statistic = {u_stat:.6f}, P-value = {u_pval:.10f}")
print(f"Mann-Whitney U Test: U-statistic = {u_stat1:.6f}, P-value = {u_pval1:.10f}")
print(f"Mann-Whitney U Test: U-statistic = {u_stat2:.6f}, P-value = {u_pval2:.10f}")

# Plotting the frequency distribution
plt.figure(figsize=(10, 6))
plt.bar([x - 0.3 for x in days], prontosan_freq, width=0.4, label='Prontosan', align='center', alpha=0.8)
plt.bar([x - 0.1 for x in days], chloramphenicol_freq, width=0.4, label='Chloramphenicol', align='center', alpha=0.8)
plt.bar([x + 0.1 for x in days], control_freq, width=0.4, label='Control', align='center', alpha=0.8)
plt.xticks(days)
plt.xlabel('Days')
plt.ylabel('Frequency (%)')
plt.legend()
plt.title('Frequency Distribution by Days (Updated Data)')
plt.tight_layout()
plt.show()

kruskal_stat, kruskal_pval = kruskal(control_clean, prontosan_clean, chloramphenicol_clean)
print(f"Kruskal-Wallis Test: Statistic = {kruskal_stat:.6f}, P-value = {kruskal_pval:.10f}")

from statsmodels.stats.multitest import multipletests
pvals = [u_pval, u_pval1, u_pval2]
_, corrected_pvals, _, _ = multipletests(pvals, method='bonferroni')
print("Corrected p-values:", corrected_pvals)

import matplotlib.pyplot as plt

# Improved plotting of the frequency distribution
plt.figure(figsize=(12, 7))

# Bar chart with improved visuals
plt.bar([x - 0.2 for x in days], prontosan_freq, width=0.4, label='Prontosan', align='center', edgecolor='black', alpha=0.8, color='skyblue')
plt.bar([x + 0.2 for x in days], chloramphenicol_freq, width=0.4, label='Chloramphenicol', align='center', edgecolor='black', alpha=0.8, color='orange')
plt.bar([x + 0.2 for x in days], control_freq, width=0.4, label='Control', align='center', edgecolor='black', alpha=0.8, color='orange')

# Enhanced x-axis and y-axis ticks and labels
plt.xticks(days, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Days', fontsize=14, weight='bold')
plt.ylabel('Frequency (%)', fontsize=14, weight='bold')

# Improved legend and title
plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True, borderpad=1)
plt.title('Frequency Distribution of Granulation Tissue Formation by Treatment', fontsize=16, weight='bold', pad=20)

# Adding gridlines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Final adjustments and display
plt.tight_layout()
plt.show()

import pandas as pd

# Create a DataFrame to display frequencies neatly
frequency_data = pd.DataFrame({
    "Days": days,
    "Chloramphenicol (%)": chloramphenicol_freq,
    "Prontosan (%)": prontosan_freq,
    "Control (%)": control_freq

})

frequency_data = frequency_data.round(2)

# Print the frequency table
print(frequency_data.to_string(index=False))

# =============================================================================
# Power Analysis for Sample Size Justification
# =============================================================================
from statsmodels.stats.power import FTestAnovaPower, TTestIndPower
from statsmodels.stats.power import tt_ind_solve_power
import numpy as np

# Using observed effect sizes from the current data
effect_size = (np.mean(control_clean) - np.mean(chloramphenicol_clean)) / np.sqrt(
    (np.std(control_clean)**2 + np.std(chloramphenicol_clean)**2) / 2
)

# Parameters for power analysis
alpha = 0.05  # Significance level
power = 0.8   # Desired power (conventional threshold)

# Calculate required sample size for t-test
required_n = tt_ind_solve_power(
    effect_size=effect_size,
    alpha=alpha,
    power=power,
    ratio=1.0
)

print("\nPower Analysis Results:")
print(f"Observed effect size: {effect_size:.3f} (Cohen's d)")
print(f"Required sample size per group for t-test (alpha={alpha}, power={power}): {np.ceil(required_n):.0f}")
print(f"Current sample sizes - Control: {len(control_clean)}, Chloramphenicol: {len(chloramphenicol_clean)}, Prontosan: {len(prontosan_clean)}")

# For non-parametric tests (Mann-Whitney), we typically need ~15% more samples
print(f"Recommended minimum sample size for non-parametric tests: {np.ceil(required_n * 1.15):.0f}")

# Post-hoc power calculation for current sample sizes
posthoc_power = TTestIndPower().power(
    effect_size=effect_size,
    nobs1=len(control_clean),
    alpha=alpha,
    ratio=len(chloramphenicol_clean)/len(control_clean)
)

print(f"Post-hoc power with current samples: {posthoc_power:.3f}")

# For ANOVA (if considering multiple group comparisons)
anova_power = FTestAnovaPower().power(
    effect_size=effect_size,
    nobs=len(control_clean),
    alpha=alpha,
    k_groups=3
)

print(f"ANOVA power with current samples: {anova_power:.3f}")

# Sensitivity analysis - smallest detectable effect with current samples
min_detectable_effect = tt_ind_solve_power(
    effect_size=None,
    nobs1=len(control_clean),
    alpha=alpha,
    power=power,
    ratio=len(chloramphenicol_clean)/len(control_clean)
)

print(f"Minimum detectable effect size with current samples: {min_detectable_effect:.3f}")
# print("\nNote: Non-parametric tests (Mann-Whitney U) are used due to non-normality, but power analysis assumes parametric tests for simplicity.")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Prepare data for boxplot
data_clean = {
    "Chloramphenicol": chloramphenicol_clean,
    "Prontosan": prontosan_clean,
    "Control": control_clean
}

# Create boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=list(data_clean.values()), palette=["skyblue", "orange", "lightgreen"])
plt.xticks([0, 1, 2], labels=data_clean.keys())
plt.ylabel("Days", fontsize=12, weight="bold")
plt.title("Distribution of Granulation Tissue Formation by Treatment", fontsize=14, weight="bold")

# Add significance bars
y_max = max(np.max(control_clean), np.max(prontosan_clean), np.max(chloramphenicol_clean)) + 1
plt.plot([0, 1], [y_max, y_max], color="black", lw=1)
plt.plot([0, 2], [y_max + 0.5, y_max + 0.5], color="black", lw=1)
plt.plot([1, 2], [y_max + 1, y_max + 1], color="black", lw=1)
plt.text(0.5, y_max + 0.1, "***", ha="center", va="bottom", color="black")
plt.text(1, y_max + 0.6, "***", ha="center", va="bottom", color="black")
plt.text(1.5, y_max + 1.1, "***", ha="center", va="bottom", color="black")

plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

# Print Bonferroni-adjusted p-values table
print("\nBonferroni-Adjusted Pairwise Comparisons:")
print(pd.DataFrame({
    "Comparison": ["Chloramphenicol vs Prontosan", "Chloramphenicol vs Control", "Control vs Prontosan"],
    "U Statistic": [176.5, 9.0, 2009.5],
    "Raw p-value": ["<0.0001", "<0.0001", "3.40e-09"],
    "Adjusted p-value": ["2.68e-12", "2.24e-20", "1.02e-08"],
    "Significance": ["p < 0.001***", "p < 0.001***", "p < 0.001***"]
}).to_string(index=False))

# Generate the boxplot (code as above)
