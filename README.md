# Statistical Analysis and Visualization of Granulation Tissue Formation

This repository contains Python code for analyzing and visualizing data on granulation tissue formation across three groups: **Control**, **Prontosan**, and **Chloramphenicol**. The analysis includes statistical tests (Levene's, Shapiro-Wilk, Mann-Whitney U, and Kruskal-Wallis) and power analysis, along with visualizations such as frequency distribution bar plots and boxplots.

## Project Overview

The code performs the following tasks:
1. **Data Preparation**: Handles missing values and aligns data for analysis.
2. **Statistical Analysis**:
   - Tests for variance equality (Levene's Test).
   - Tests for normality (Shapiro-Wilk Test).
   - Non-parametric comparisons (Mann-Whitney U and Kruskal-Wallis Tests).
   - Bonferroni correction for multiple comparisons.
3. **Power Analysis**: Calculates required sample sizes and post-hoc power for the study.
4. **Visualization**:
   - Bar plots for frequency distributions of days to granulation tissue formation.
   - Boxplots to compare distributions across groups with significance annotations.

## Requirements

To run the code, install the required Python packages:
```bash
pip install pandas numpy scipy matplotlib seaborn statsmodels
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Run the Python script:
   ```bash
   python analysis.py
   ```

3. The script will:
   - Process the dataset.
   - Perform statistical tests and print results.
   - Generate visualizations (bar plots and boxplots).
   - Output a frequency table and power analysis results.

## Dataset

The dataset consists of three groups:
- **Control**: 61 observations.
- **Prontosan**: 39 observations (with missing values padded as NaN).
- **Chloramphenicol**: 55 observations (with missing values padded as NaN).

Each value represents the number of days to granulation tissue formation.

## Key Results

- **Levene's Test**: Assesses equality of variances across groups.
- **Shapiro-Wilk Test**: Checks normality for each group.
- **Mann-Whitney U Test**: Pairwise comparisons between groups.
- **Kruskal-Wallis Test**: Non-parametric ANOVA for group differences.
- **Power Analysis**: Provides sample size requirements and post-hoc power calculations.
- **Visualizations**:
  - Bar plot showing frequency distributions of days for each group.
  - Boxplot with significance annotations for pairwise comparisons.

## Example Outputs

### Frequency Table
| Days | Chloramphenicol (%) | Prontosan (%) | Control (%) |
|------|---------------------|---------------|-------------|
| 3    | 43.64               | 5.13          | 0.00        |
| 4    | 38.18               | 23.08         | 0.00        |
| 5    | 9.09                | 41.03         | 11.48       |
| 6    | 0.00                | 17.95         | 26.23       |
| 7    | 1.82                | 12.82         | 31.15       |
| 8    | 0.00                | 0.00          | 21.31       |
| 9    | 0.00                | 0.00          | 9.84        |

### Visualizations
- **Bar Plot**: Displays frequency (%) of days for each group.
- **Boxplot**: Shows distribution of days with significance bars (***p < 0.001***).

### Statistical Tests
- **Levene's Test**: Tests variance homogeneity.
- **Shapiro-Wilk Test**: Indicates non-normality for some groups, justifying non-parametric tests.
- **Mann-Whitney U Test**: Significant differences between all pairs (Bonferroni-adjusted p-values < 0.001).
- **Kruskal-Wallis Test**: Confirms significant differences across groups.
- **Power Analysis**: Provides insights into sample size adequacy and detectable effect sizes.

## File Structure

- `analysis.py`: Main Python script containing the analysis and visualization code.
- `README.md`: This file, providing an overview and instructions.
- `output/`: Directory for generated plots (e.g., `frequency_distribution.png`, `boxplot.png`).

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for suggestions or bug reports.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please contact Prof. Sergey Shityakov [shityakoff@hotmail.com] or open an issue in the repository.

