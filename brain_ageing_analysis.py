import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn import datasets
from sklearn.linear_model import LinearRegression

# 1. Fetch the OASIS dataset directly from the web
print("Downloading/Fetching OASIS dataset (this may take a moment)...")
oasis_data = datasets.fetch_oasis_vbm(n_subjects=100)

# 2. Extract clinical data into a DataFrame
df = pd.DataFrame(oasis_data.ext_vars)

# 3. Data Formatting
# nWBV = Normalized Whole Brain Volume (a key biomarker of neurodegeneration)
df['age'] = df['age'].astype(float)
df['nwbv'] = df['nwbv'].astype(float)

# 4. Statistical Modeling
X = df[['age']]
y = df['nwbv']
model = LinearRegression().fit(X, y)
r_sq = model.score(X, y)

print(f"Analysis Complete. R-squared value: {r_sq:.2f}")

# 5. Professional Visualization
plt.figure(figsize=(10, 6))
sns.set_style("ticks")
plot = sns.regplot(x='age', y='nwbv', data=df, 
                  scatter_kws={'alpha':0.5, 'color':'#2c3e50'}, 
                  line_kws={'color':'#e74c3c', 'lw':3})

plt.title('Structural Brain Atrophy Across the Lifespan', fontsize=16, pad=20)
plt.xlabel('Age (Years)', fontsize=12)
plt.ylabel('Normalized Whole Brain Volume (nwbv)', fontsize=12)

# Save the plot 
plt.tight_layout()
plt.savefig('brain_aging_results.png', dpi=300)
print("Graph saved as 'brain_aging_results.png'.")
plt.show()
# Compare aging trajectories by Gender
plt.figure(figsize=(12, 8))
sns.lmplot(x='age', y='nwbv', hue='mf', data=df, markers=["o", "x"], palette="Set1")

plt.title('Brain Atrophy: Males vs. Females')
plt.xlabel('Age')
plt.ylabel('Normalized Whole Brain Volume')
plt.savefig('brain_aging_between_gender.png', dpi=300)
print("Graph saved as 'brain_aging_between_gender.png'.")
plt.show()

# Correlation analysis by gender
males = df[df['mf'] == "M" ]# Check your dataset if Male is 1 or 'M'
females = df[df['mf'] == "F"]

print(f"Male Age-Volume Correlation: {males['age'].corr(males['nwbv']):.2f}")
print(f"Female Age-Volume Correlation: {females['age'].corr(females['nwbv']):.2f}")

# 1. Define the list of potential columns we want to correlate
potential_cols = ['age', 'educ', 'ses', 'mmse', 'nwbv', 'etiv', 'gender']

# 2. Filter the list to only include columns that actually exist in the df
existing_cols = [col for col in potential_cols if col in df.columns]

print(f"Generating heatmap for available columns: {existing_cols}")

# 3. Create the correlation matrix
corr_matrix = df[existing_cols].corr()

# 4. Plot the Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap: Brain Metrics & Clinical Factors', fontsize=15)
plt.tight_layout()
plt.savefig('brain_correlation_heatmap.png', dpi=300) # Saves at high resolution
plt.show()


# 1. Focus on the elderly group (70+) to identify super agers
elderly_df = df[df['age'] >= 70]

# 2. Find the 75th percentile of volume within that elderly group
# This identifies the "healthiest" brains for that specific age
resilience_threshold = elderly_df['nwbv'].quantile(0.75)

# 3. Identify the Super-Agers
super_agers = elderly_df[elderly_df['nwbv'] >= resilience_threshold]

print(f"--- RELATIVE RESILIENCE ANALYSIS ---")
print(f"Number of Resilient Brains (Age 70+ in top 25% volume): {len(super_agers)}")

# 4. Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='nwbv', data=df, color='gray', alpha=0.3, label='General Population')
sns.scatterplot(x='age', y='nwbv', data=super_agers, color='gold', s=100, 
                label='Top 25% Resilience (Age 70+)', edgecolor='black')

# Add the regression line
sns.regplot(x='age', y='nwbv', data=df, scatter=False, color='red')

plt.title('Biological Resilience: Identifying Outliers in Brain Aging')
plt.savefig('resilient_brains_plot.png', dpi=300)
plt.show()

# 1. Calculate the average MMSE for Super-Agers vs. Other Elderly
super_ager_mmse = super_agers['mmse'].mean()
other_elderly_mmse = elderly_df[~elderly_df.index.isin(super_agers.index)]['mmse'].mean()
print(f"\n--- CLINICAL LINK ANALYSIS ---")
print(f"Average MMSE of Super-Agers: {super_ager_mmse:.2f}")
print(f"Average MMSE of other elderly (70+): {other_elderly_mmse:.2f}")

# 2. Visualize the difference with a Boxplot
plt.figure(figsize=(8, 6))
# Create a temporary column for labeling
df['Category'] = 'Normal Population'
df.loc[elderly_df.index, 'Category'] = 'Elderly (70+)'
df.loc[super_agers.index, 'Category'] = 'Super-Agers'
sns.boxplot(x='Category', y='mmse', data=df[df['Category'] != 'Normal Population'], palette="Set2")
plt.title('Cognitive Performance (MMSE): Super-Agers vs. Peers')
plt.ylabel('MMSE Score (0-30)')
plt.savefig('clinical_validation_mmse.png')
plt.show()
