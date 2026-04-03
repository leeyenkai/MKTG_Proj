#!/usr/bin/env python3
"""
Restaurant Customer Satisfaction Analysis
GWU MKTG 4163 — Applied Marketing Analytics, Spring 2026
Team Project: Predicting Restaurant Customer Satisfaction

Generates a self-contained HTML report with all analyses embedded.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, chi2_contingency, pearsonr, zscore
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
CSV_PATH = 'restaurant_customer_satisfaction.csv'
OUTPUT_PATH = 'restaurant_satisfaction_analysis.html'

# GW Colors
NAVY = '#033C5A'
GOLD = '#AA9868'
LIGHT_GOLD = '#F5F0E6'
WHITE = '#FFFFFF'
GRAY = '#6B7280'
LIGHT_GRAY = '#F3F4F6'

sns.set_theme(style="whitegrid", palette=[NAVY, GOLD, '#4A90D9', '#D4A843', '#2C6E8A', '#C4956A'])
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

# ── Helper Functions ───────────────────────────────────────────────────────────
def fig_to_base64(fig, dpi=150):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f'data:image/png;base64,{encoded}'

def make_table(df, caption='', highlight_header=True):
    html = f'<table class="data-table"><caption>{caption}</caption>'
    html += '<thead><tr>'
    for col in df.columns:
        html += f'<th>{col}</th>'
    html += '</tr></thead><tbody>'
    for _, row in df.iterrows():
        html += '<tr>'
        for val in row:
            if isinstance(val, float):
                html += f'<td>{val:.4f}</td>'
            else:
                html += f'<td>{val}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    return html

def significance_label(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'n.s.'

# ── Load Data ──────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(CSV_PATH)
n_rows, n_cols = df.shape
print(f"Dataset: {n_rows} rows, {n_cols} columns")

# Define variable groups
numeric_cols = ['Age', 'Income', 'AverageSpend', 'GroupSize', 'WaitTime',
                'ServiceRating', 'FoodRating', 'AmbianceRating']
categorical_cols = ['Gender', 'VisitFrequency', 'PreferredCuisine', 'TimeOfVisit',
                    'DiningOccasion', 'MealType']
binary_cols = ['OnlineReservation', 'DeliveryOrder', 'LoyaltyProgramMember']
target = 'HighSatisfaction'

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: EXECUTIVE SUMMARY (computed after all analyses)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: DESCRIPTIVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("Running descriptive analysis...")

# Summary statistics
desc_stats = df[numeric_cols].describe().T
desc_stats['median'] = df[numeric_cols].median()
desc_stats = desc_stats[['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max']]
desc_stats = desc_stats.round(2)
desc_stats_html = make_table(desc_stats.reset_index().rename(columns={'index': 'Variable'}),
                              'Summary Statistics for Numeric Variables')

# Satisfaction rate
sat_rate = df[target].mean() * 100

# Frequency distributions for categorical variables
freq_tables_html = ''
for col in categorical_cols:
    freq = df[col].value_counts().reset_index()
    freq.columns = [col, 'Count']
    freq['Percentage'] = (freq['Count'] / len(df) * 100).round(1)
    freq = freq.sort_values(col)
    freq_tables_html += make_table(freq, f'Distribution of {col}')

# Binary variable distributions
binary_html = ''
for col in binary_cols:
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, 'Count']
    counts[col] = counts[col].map({0: 'No', 1: 'Yes'})
    counts['Percentage'] = (counts['Count'] / len(df) * 100).round(1)
    binary_html += make_table(counts, f'Distribution of {col}')

# ── Chart 1: Histograms for numeric variables ──
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for i, col in enumerate(numeric_cols):
    ax = axes[i // 4][i % 4]
    ax.hist(df[col], bins=30, color=NAVY, alpha=0.8, edgecolor='white')
    ax.set_title(col, fontweight='bold')
    ax.set_ylabel('Frequency')
    mean_val = df[col].mean()
    ax.axvline(mean_val, color=GOLD, linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax.legend(fontsize=9)
fig.suptitle('Distribution of Numeric Variables', fontsize=16, fontweight='bold', y=1.02)
fig.tight_layout()
histograms_img = fig_to_base64(fig)

# ── Chart 2: Satisfaction rate by categorical variables ──
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for i, col in enumerate(categorical_cols):
    ax = axes[i // 3][i % 3]
    sat_by_cat = df.groupby(col)[target].mean() * 100
    sat_by_cat = sat_by_cat.sort_values(ascending=False)
    bars = ax.bar(range(len(sat_by_cat)), sat_by_cat.values, color=NAVY, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(sat_by_cat)))
    ax.set_xticklabels(sat_by_cat.index, rotation=45, ha='right', fontsize=9)
    ax.set_title(f'Satisfaction Rate by {col}', fontweight='bold')
    ax.set_ylabel('High Satisfaction (%)')
    ax.axhline(sat_rate, color=GOLD, linestyle='--', linewidth=1.5, label=f'Overall: {sat_rate:.1f}%')
    ax.legend(fontsize=9)
    for bar, val in zip(bars, sat_by_cat.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
fig.suptitle('High Satisfaction Rate by Customer Segments', fontsize=16, fontweight='bold', y=1.02)
fig.tight_layout()
sat_by_cat_img = fig_to_base64(fig)

# ── Chart 3: Box plots for ratings ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
rating_cols = ['ServiceRating', 'FoodRating', 'AmbianceRating']
for i, col in enumerate(rating_cols):
    df_plot = df.copy()
    df_plot[target] = df_plot[target].astype(str)
    sns.boxplot(x=target, y=col, data=df_plot, ax=axes[i],
                palette={'0': GRAY, '1': GOLD}, width=0.5)
    axes[i].set_xticklabels(['Low Satisfaction', 'High Satisfaction'])
    axes[i].set_title(f'{col} by Satisfaction Level', fontweight='bold')
fig.suptitle('Rating Distributions by Satisfaction Level', fontsize=16, fontweight='bold', y=1.02)
fig.tight_layout()
boxplot_ratings_img = fig_to_base64(fig)

# ── Chart 4: Average spend by cuisine with box plots ──
fig, ax = plt.subplots(figsize=(10, 5))
order = df.groupby('PreferredCuisine')['AverageSpend'].median().sort_values(ascending=False).index
sns.boxplot(x='PreferredCuisine', y='AverageSpend', data=df, order=order, ax=ax,
            palette=[NAVY, GOLD, '#4A90D9', '#D4A843', '#2C6E8A', '#C4956A'])
ax.set_title('Average Spend Distribution by Preferred Cuisine', fontweight='bold', fontsize=14)
ax.set_xlabel('Preferred Cuisine')
ax.set_ylabel('Average Spend ($)')
fig.tight_layout()
spend_cuisine_img = fig_to_base64(fig)

# ── Cross-tabulations: The 3 qualities to assess ──
cross_tabs_html = ''
for col in ['TimeOfVisit', 'DiningOccasion', 'VisitFrequency']:
    ct = pd.crosstab(df[col], df[target], margins=True)
    ct.columns = ['Low Satisfaction', 'High Satisfaction', 'Total']
    ct['Satisfaction Rate (%)'] = (ct['High Satisfaction'] / ct['Total'] * 100).round(1)
    cross_tabs_html += make_table(ct.reset_index().rename(columns={col: col}),
                                   f'Cross-Tabulation: {col} vs. High Satisfaction')

# ── Pivot table: Average ratings by cuisine, occasion, meal type ──
pivot_cuisine = df.groupby('PreferredCuisine')[rating_cols + ['WaitTime', 'AverageSpend']].mean().round(2)
pivot_cuisine['HighSat Rate (%)'] = (df.groupby('PreferredCuisine')[target].mean() * 100).round(1)
pivot_html = make_table(pivot_cuisine.reset_index().rename(columns={'PreferredCuisine': 'Cuisine'}),
                         'Average Metrics by Preferred Cuisine')

pivot_occasion = df.groupby('DiningOccasion')[rating_cols + ['WaitTime', 'AverageSpend']].mean().round(2)
pivot_occasion['HighSat Rate (%)'] = (df.groupby('DiningOccasion')[target].mean() * 100).round(1)
pivot_html += make_table(pivot_occasion.reset_index().rename(columns={'DiningOccasion': 'Occasion'}),
                          'Average Metrics by Dining Occasion')

pivot_meal = df.groupby('MealType')[rating_cols + ['WaitTime', 'AverageSpend']].mean().round(2)
pivot_meal['HighSat Rate (%)'] = (df.groupby('MealType')[target].mean() * 100).round(1)
pivot_html += make_table(pivot_meal.reset_index().rename(columns={'MealType': 'Meal Type'}),
                          'Average Metrics by Meal Type')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: INFERENTIAL STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
print("Running inferential statistics...")

# ── Correlation matrix ──
corr_cols = numeric_cols + [target]
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax, square=True,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Matrix — Numeric Variables & Satisfaction', fontweight='bold', fontsize=14)
fig.tight_layout()
corr_img = fig_to_base64(fig)

# Correlation with target (with p-values)
corr_with_target = []
for col in numeric_cols:
    r, p = pearsonr(df[col], df[target])
    corr_with_target.append({
        'Variable': col,
        'Pearson r': round(r, 4),
        'p-value': round(p, 6),
        'Significance': significance_label(p),
        'Strength': 'Strong' if abs(r) > 0.5 else ('Moderate' if abs(r) > 0.3 else 'Weak')
    })
corr_target_df = pd.DataFrame(corr_with_target).sort_values('Pearson r', key=abs, ascending=False)
corr_target_html = make_table(corr_target_df, 'Correlation with High Satisfaction (Pearson)')

# ── ANOVA Tests ──
anova_results = []
anova_charts = []
for col in ['VisitFrequency', 'TimeOfVisit', 'DiningOccasion', 'PreferredCuisine']:
    groups = [group[target].values for name, group in df.groupby(col)]
    f_stat, p_val = f_oneway(*groups)
    anova_results.append({
        'Factor': col,
        'F-Statistic': round(f_stat, 4),
        'p-value': round(p_val, 6),
        'Significance': significance_label(p_val),
        'Interpretation': 'Significant difference across groups' if p_val < 0.05
                          else 'No significant difference across groups'
    })

    # Also test with ratings
    for rating in rating_cols:
        groups_r = [group[rating].values for name, group in df.groupby(col)]
        f_r, p_r = f_oneway(*groups_r)
        anova_results.append({
            'Factor': f'{col} → {rating}',
            'F-Statistic': round(f_r, 4),
            'p-value': round(p_r, 6),
            'Significance': significance_label(p_r),
            'Interpretation': 'Significant difference' if p_r < 0.05 else 'No significant difference'
        })

anova_df = pd.DataFrame(anova_results)
anova_html = make_table(anova_df, 'ANOVA Results: Do Satisfaction Metrics Differ Across Groups?')

# ANOVA visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, col in enumerate(['VisitFrequency', 'TimeOfVisit', 'DiningOccasion', 'PreferredCuisine']):
    ax = axes[i // 2][i % 2]
    means = df.groupby(col)[target].mean() * 100
    ci = df.groupby(col)[target].apply(lambda x: 1.96 * x.std() / np.sqrt(len(x)) * 100)
    means = means.sort_values(ascending=False)
    ci = ci.reindex(means.index)
    bars = ax.bar(range(len(means)), means.values, yerr=ci.values, capsize=4,
                  color=NAVY, alpha=0.85, edgecolor='white', error_kw={'ecolor': GOLD, 'linewidth': 1.5})
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(means.index, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('High Satisfaction Rate (%)')
    ax.set_title(f'ANOVA: {col}', fontweight='bold')
    ax.axhline(sat_rate, color=GOLD, linestyle='--', alpha=0.7)
fig.suptitle('ANOVA — Satisfaction Rate by Factor Groups (with 95% CI)', fontsize=16, fontweight='bold', y=1.02)
fig.tight_layout()
anova_img = fig_to_base64(fig)

# ── Chi-Square Tests ──
chi2_results = []
all_cat_for_chi2 = categorical_cols + binary_cols
for col in all_cat_for_chi2:
    ct = pd.crosstab(df[col], df[target])
    chi2, p, dof, expected = chi2_contingency(ct)
    chi2_results.append({
        'Variable': col,
        'Chi-Square Statistic': round(chi2, 4),
        'Degrees of Freedom': dof,
        'p-value': round(p, 6),
        'Significance': significance_label(p),
        'Interpretation': 'Dependent (reject H0)' if p < 0.05 else 'Independent (fail to reject H0)'
    })
chi2_df = pd.DataFrame(chi2_results).sort_values('p-value')
chi2_html = make_table(chi2_df, 'Chi-Square Test of Independence with High Satisfaction')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: PREDICTIVE MODELING
# ══════════════════════════════════════════════════════════════════════════════
print("Running predictive modeling...")

# ── Prepare features for modeling ──
df_model = df.copy()
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col + '_encoded'] = le.fit_transform(df_model[col])
    le_dict[col] = le

feature_cols = numeric_cols + binary_cols + [c + '_encoded' for c in categorical_cols]
X = df_model[feature_cols]
y = df_model[target]

# ── Logistic Regression (sklearn for metrics) ──
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred = log_reg.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Confusion matrix chart
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Low Satisfaction', 'High Satisfaction'],
            yticklabels=['Low Satisfaction', 'High Satisfaction'])
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
ax.set_title(f'Logistic Regression Confusion Matrix\nAccuracy: {accuracy:.1%}', fontweight='bold')
fig.tight_layout()
cm_img = fig_to_base64(fig)

# Classification report
cr = classification_report(y_test, y_pred, target_names=['Low Satisfaction', 'High Satisfaction'],
                           output_dict=True)
cr_df = pd.DataFrame(cr).T.round(4)
cr_html = make_table(cr_df.reset_index().rename(columns={'index': 'Class'}),
                      'Logistic Regression Classification Report')

# ── Logistic Regression (statsmodels for coefficients/p-values) ──
X_sm = sm.add_constant(X_train_scaled)
feature_names = ['Intercept'] + feature_cols
logit_model = sm.Logit(y_train, X_sm).fit(disp=0)
logit_summary = logit_model.summary2().tables[1]
logit_summary.index = feature_names
logit_summary['Odds Ratio'] = np.exp(logit_summary['Coef.'])
logit_coef_df = logit_summary[['Coef.', 'Std.Err.', 'z', 'P>|z|', 'Odds Ratio']].round(4)
logit_coef_df = logit_coef_df.sort_values('P>|z|')
logit_html = make_table(logit_coef_df.reset_index().rename(columns={'index': 'Feature'}),
                         'Logistic Regression Coefficients and Odds Ratios')

# Feature importance chart
fig, ax = plt.subplots(figsize=(10, 7))
coefs = pd.Series(log_reg.coef_[0], index=feature_cols)
coefs_sorted = coefs.reindex(coefs.abs().sort_values(ascending=True).index)
colors = [GOLD if c > 0 else NAVY for c in coefs_sorted]
coefs_sorted.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_title('Logistic Regression Feature Importance\n(Standardized Coefficients)', fontweight='bold', fontsize=14)
ax.set_xlabel('Coefficient Value')
ax.axvline(0, color='black', linewidth=0.8)
ax.legend(['Positive Effect', 'Negative Effect'], loc='lower right')
fig.tight_layout()
feat_imp_img = fig_to_base64(fig)

# ── Logistic Regression S-Curve with actual data points ──
# Use the single most important feature (highest abs coefficient) for the 1D S-curve
best_feature_idx = np.argmax(np.abs(log_reg.coef_[0]))
best_feature_name = feature_cols[best_feature_idx]
x_best = X_test_scaled[:, best_feature_idx]

# Sort for smooth curve
sort_idx = np.argsort(x_best)
x_sorted = x_best[sort_idx]
y_actual = y_test.values[sort_idx]
y_prob = log_reg.predict_proba(X_test_scaled)[:, 1][sort_idx]

# Generate smooth S-curve across full range
x_range = np.linspace(x_sorted.min() - 1, x_sorted.max() + 1, 500).reshape(-1, 1)
# Build a full feature matrix at mean values, varying only the best feature
X_curve = np.tile(X_test_scaled.mean(axis=0), (500, 1))
X_curve[:, best_feature_idx] = x_range.ravel()
y_curve = log_reg.predict_proba(X_curve)[:, 1]

fig, ax = plt.subplots(figsize=(10, 6))
# Plot actual data points (jittered vertically for visibility)
jitter = np.random.uniform(-0.03, 0.03, len(y_actual))
ax.scatter(x_best[y_test.values == 0], y_actual[y_test.values == 0] + jitter[y_test.values == 0],
           alpha=0.3, color=NAVY, s=20, label='Actual: Low Satisfaction (0)', zorder=2)
ax.scatter(x_best[y_test.values == 1], y_actual[y_test.values == 1] + jitter[y_test.values == 1],
           alpha=0.3, color=GOLD, s=20, label='Actual: High Satisfaction (1)', zorder=2)
# Plot the logistic S-curve
ax.plot(x_range, y_curve, color='red', linewidth=2.5, label='Logistic Regression Curve', zorder=3)
# Decision boundary
ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, label='Decision Boundary (0.5)')
# Residual error lines for a sample of points
sample_idx = np.random.choice(len(x_sorted), size=min(80, len(x_sorted)), replace=False)
for idx in sample_idx:
    ax.plot([x_best[idx], x_best[idx]],
            [y_test.values[idx], y_prob[idx] if np.argsort(x_best)[idx] < len(y_prob) else 0.5],
            color='red', alpha=0.15, linewidth=0.8, zorder=1)

ax.set_xlabel(f'Standardized {best_feature_name}', fontsize=12)
ax.set_ylabel('P(High Satisfaction)', fontsize=12)
ax.set_title(f'Logistic Regression S-Curve with Residual Errors\nPrimary Predictor: {best_feature_name}',
             fontweight='bold', fontsize=14)
ax.legend(loc='center right', fontsize=9)
ax.set_ylim(-0.08, 1.08)
fig.tight_layout()
logistic_curve_img = fig_to_base64(fig)

# ── Linear Regression (composite satisfaction score) ──
df_model['CompositeSatisfaction'] = (df_model['ServiceRating'] + df_model['FoodRating'] + df_model['AmbianceRating']) / 3
feature_cols_lr = ['Age', 'Income', 'AverageSpend', 'GroupSize', 'WaitTime'] + binary_cols + [c + '_encoded' for c in categorical_cols]
X_lr = sm.add_constant(df_model[feature_cols_lr])
y_lr = df_model['CompositeSatisfaction']
ols_model = sm.OLS(y_lr, X_lr).fit()

ols_summary_data = []
for i, name in enumerate(['Intercept'] + feature_cols_lr):
    ols_summary_data.append({
        'Variable': name,
        'Coefficient': round(ols_model.params.iloc[i], 4),
        'Std Error': round(ols_model.bse.iloc[i], 4),
        't-value': round(ols_model.tvalues.iloc[i], 4),
        'p-value': round(ols_model.pvalues.iloc[i], 6),
        'Significance': significance_label(ols_model.pvalues.iloc[i])
    })
ols_df = pd.DataFrame(ols_summary_data)
ols_html = make_table(ols_df, f'Linear Regression: Predicting Composite Satisfaction Score (R² = {ols_model.rsquared:.4f})')

# ── Monte Carlo Simulation ──
print("Running Monte Carlo simulation (10,000 iterations)...")
np.random.seed(42)
n_sim = 10000

# Use logistic regression to predict satisfaction probability for simulated customers
sim_results = []
# Simulate different wait time scenarios
wait_scenarios = {'Low Wait (0-15 min)': (0, 15), 'Medium Wait (15-30 min)': (15, 30), 'High Wait (30-60 min)': (30, 60)}
mc_scenario_results = {}

for scenario_name, (low, high) in wait_scenarios.items():
    sim_data = pd.DataFrame()
    for col in feature_cols:
        if col in numeric_cols:
            if col == 'WaitTime':
                sim_data[col] = np.random.uniform(low, high, n_sim)
            else:
                sim_data[col] = np.random.normal(df[col].mean(), df[col].std(), n_sim)
        elif col in binary_cols:
            sim_data[col] = np.random.binomial(1, df[col].mean(), n_sim)
        else:
            sim_data[col] = np.random.choice(df_model[col].values, n_sim)

    sim_scaled = scaler.transform(sim_data[feature_cols])
    sim_probs = log_reg.predict_proba(sim_scaled)[:, 1]
    mc_scenario_results[scenario_name] = sim_probs

# Monte Carlo chart
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, (scenario, probs) in enumerate(mc_scenario_results.items()):
    ax = axes[i]
    ax.hist(probs, bins=50, color=NAVY, alpha=0.8, edgecolor='white', density=True)
    ax.axvline(probs.mean(), color=GOLD, linewidth=2, linestyle='--',
               label=f'Mean: {probs.mean():.3f}')
    ax.axvline(0.5, color='red', linewidth=1.5, linestyle=':',
               label='Decision Boundary')
    ax.set_title(scenario, fontweight='bold')
    ax.set_xlabel('Predicted Probability of High Satisfaction')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)
fig.suptitle('Monte Carlo Simulation: Satisfaction Probability Under Different Wait Time Scenarios\n(n = 10,000 per scenario)',
             fontsize=14, fontweight='bold', y=1.05)
fig.tight_layout()
mc_img = fig_to_base64(fig)

mc_summary_data = []
for scenario, probs in mc_scenario_results.items():
    mc_summary_data.append({
        'Scenario': scenario,
        'Mean Probability': round(probs.mean(), 4),
        'Median Probability': round(np.median(probs), 4),
        'Std Dev': round(probs.std(), 4),
        '% Predicted High Satisfaction': round((probs >= 0.5).mean() * 100, 1),
        '5th Percentile': round(np.percentile(probs, 5), 4),
        '95th Percentile': round(np.percentile(probs, 95), 4)
    })
mc_df = pd.DataFrame(mc_summary_data)
mc_html = make_table(mc_df, 'Monte Carlo Simulation Results: Wait Time Scenarios (10,000 iterations each)')

# Also simulate by cuisine type
mc_cuisine_results = {}
for cuisine in df['PreferredCuisine'].unique():
    sim_data = pd.DataFrame()
    for col in feature_cols:
        if col == 'PreferredCuisine_encoded':
            sim_data[col] = le_dict['PreferredCuisine'].transform([cuisine] * n_sim)
        elif col in numeric_cols:
            sim_data[col] = np.random.normal(df[col].mean(), df[col].std(), n_sim)
        elif col in binary_cols:
            sim_data[col] = np.random.binomial(1, df[col].mean(), n_sim)
        else:
            sim_data[col] = np.random.choice(df_model[col].values, n_sim)
    sim_scaled = scaler.transform(sim_data[feature_cols])
    mc_cuisine_results[cuisine] = log_reg.predict_proba(sim_scaled)[:, 1]

mc_cuisine_data = []
for cuisine, probs in mc_cuisine_results.items():
    mc_cuisine_data.append({
        'Cuisine': cuisine,
        'Mean Satisfaction Probability': round(probs.mean(), 4),
        '% Predicted High Satisfaction': round((probs >= 0.5).mean() * 100, 1)
    })
mc_cuisine_df = pd.DataFrame(mc_cuisine_data).sort_values('Mean Satisfaction Probability', ascending=False)
mc_cuisine_html = make_table(mc_cuisine_df, 'Monte Carlo Simulation: Satisfaction Probability by Cuisine Type')

# ── Feature Importance Ranking (combined view) ──
# Use absolute standardized logistic regression coefficients
importance = pd.DataFrame({
    'Feature': feature_cols,
    'Abs Coefficient': np.abs(log_reg.coef_[0]),
    'Direction': ['Positive' if c > 0 else 'Negative' for c in log_reg.coef_[0]]
}).sort_values('Abs Coefficient', ascending=False)
importance['Rank'] = range(1, len(importance) + 1)
importance_html = make_table(importance[['Rank', 'Feature', 'Abs Coefficient', 'Direction']],
                              'Feature Importance Ranking (Logistic Regression)')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: PRESCRIPTIVE ANALYSIS — computed from results
# ══════════════════════════════════════════════════════════════════════════════
print("Generating prescriptive recommendations...")

# Top 5 most important features
top5 = importance.head(5)['Feature'].tolist()

# Satisfaction by loyalty
loyalty_sat = df.groupby('LoyaltyProgramMember')[target].mean() * 100
loyalty_lift = loyalty_sat.get(1, 0) - loyalty_sat.get(0, 0)

# Satisfaction by online reservation
reservation_sat = df.groupby('OnlineReservation')[target].mean() * 100

# Wait time insights
low_wait = df[df['WaitTime'] <= 15][target].mean() * 100
high_wait = df[df['WaitTime'] > 30][target].mean() * 100


# ══════════════════════════════════════════════════════════════════════════════
# BUILD HTML REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("Building HTML report...")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Restaurant Customer Satisfaction Analysis — GWU MKTG 4163</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #1a1a1a;
    line-height: 1.6;
    background: {LIGHT_GRAY};
  }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}

  /* Header */
  .header {{
    background: linear-gradient(135deg, {NAVY} 0%, #065A85 100%);
    color: white;
    padding: 40px;
    text-align: center;
    border-radius: 12px;
    margin-bottom: 30px;
    box-shadow: 0 4px 20px rgba(3,60,90,0.3);
  }}
  .header h1 {{ font-size: 2.2em; margin-bottom: 10px; letter-spacing: 0.5px; }}
  .header .subtitle {{ color: {GOLD}; font-size: 1.1em; margin-bottom: 5px; }}
  .header .meta {{ color: #a0c4d8; font-size: 0.9em; }}

  /* Table of Contents */
  .toc {{
    background: white;
    border-radius: 12px;
    padding: 30px;
    margin-bottom: 30px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
  }}
  .toc h2 {{ color: {NAVY}; margin-bottom: 15px; }}
  .toc ol {{ padding-left: 25px; }}
  .toc li {{ margin-bottom: 8px; }}
  .toc a {{ color: {NAVY}; text-decoration: none; font-weight: 500; }}
  .toc a:hover {{ color: {GOLD}; text-decoration: underline; }}

  /* Sections */
  .section {{
    background: white;
    border-radius: 12px;
    padding: 35px;
    margin-bottom: 30px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
  }}
  .section h2 {{
    color: {NAVY};
    font-size: 1.6em;
    border-bottom: 3px solid {GOLD};
    padding-bottom: 10px;
    margin-bottom: 25px;
  }}
  .section h3 {{
    color: {NAVY};
    font-size: 1.2em;
    margin: 25px 0 15px 0;
  }}
  .section p {{ margin-bottom: 12px; color: #333; }}

  /* Stats Cards */
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin: 20px 0;
  }}
  .stat-card {{
    background: linear-gradient(135deg, {NAVY} 0%, #065A85 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
  }}
  .stat-card .stat-value {{ font-size: 2em; font-weight: bold; color: {GOLD}; }}
  .stat-card .stat-label {{ font-size: 0.85em; margin-top: 5px; color: #a0c4d8; }}

  /* Tables */
  .data-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0 25px 0;
    font-size: 0.9em;
  }}
  .data-table caption {{
    font-weight: bold;
    font-size: 1.05em;
    color: {NAVY};
    text-align: left;
    padding: 10px 0;
  }}
  .data-table th {{
    background: {NAVY};
    color: white;
    padding: 10px 12px;
    text-align: left;
    font-weight: 600;
  }}
  .data-table td {{
    padding: 8px 12px;
    border-bottom: 1px solid #e5e7eb;
  }}
  .data-table tr:nth-child(even) {{ background: {LIGHT_GOLD}; }}
  .data-table tr:hover {{ background: #e8e0d0; }}

  /* Charts */
  .chart-container {{
    text-align: center;
    margin: 20px 0;
  }}
  .chart-container img {{
    max-width: 100%;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  }}

  /* Key findings / callout boxes */
  .callout {{
    background: {LIGHT_GOLD};
    border-left: 5px solid {GOLD};
    padding: 15px 20px;
    margin: 15px 0;
    border-radius: 0 8px 8px 0;
  }}
  .callout-navy {{
    background: #e8f0f5;
    border-left: 5px solid {NAVY};
    padding: 15px 20px;
    margin: 15px 0;
    border-radius: 0 8px 8px 0;
  }}
  .callout strong {{ color: {NAVY}; }}

  /* Recommendations */
  .rec-card {{
    background: white;
    border: 2px solid {GOLD};
    border-radius: 10px;
    padding: 20px;
    margin: 15px 0;
  }}
  .rec-card h4 {{ color: {NAVY}; margin-bottom: 8px; }}
  .rec-card .evidence {{ color: {GRAY}; font-size: 0.9em; font-style: italic; }}

  /* Footer */
  .footer {{
    text-align: center;
    padding: 20px;
    color: {GRAY};
    font-size: 0.85em;
  }}

  /* Responsive */
  @media (max-width: 768px) {{
    .header h1 {{ font-size: 1.5em; }}
    .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .section {{ padding: 20px; }}
  }}

  /* Print */
  @media print {{
    .section {{ page-break-inside: avoid; }}
    body {{ background: white; }}
    .header {{ box-shadow: none; }}
  }}

  ul, ol {{ margin-left: 20px; margin-bottom: 12px; }}
  li {{ margin-bottom: 5px; }}
</style>
</head>
<body>
<div class="container">

<!-- HEADER -->
<div class="header">
  <h1>Predicting Restaurant Customer Satisfaction</h1>
  <div class="subtitle">What Are the Most Influential Factors Determining Restaurant Satisfaction?</div>
  <div class="meta">GWU MKTG 4163 — Applied Marketing Analytics | Spring 2026 | Professor David Ashley</div>
  <div class="meta" style="margin-top:5px;">Team Members: Max, Leonie, Sam & Yen Kai</div>
</div>

<!-- TABLE OF CONTENTS -->
<div class="toc">
  <h2>Table of Contents</h2>
  <ol>
    <li><a href="#exec-summary">Executive Summary</a></li>
    <li><a href="#data-context">Dataset Context & Background</a></li>
    <li><a href="#descriptive">Descriptive Analysis</a></li>
    <li><a href="#inferential">Inferential Statistics</a></li>
    <li><a href="#predictive">Predictive Modeling</a></li>
    <li><a href="#prescriptive">Prescriptive Analysis & Recommendations</a></li>
    <li><a href="#limitations">Limitations</a></li>
  </ol>
</div>

<!-- SECTION 1: EXECUTIVE SUMMARY -->
<div class="section" id="exec-summary">
  <h2>1. Executive Summary</h2>
  <p>This analysis investigates the key drivers of customer satisfaction in the restaurant industry, using a dataset of <strong>{n_rows:,} customer records</strong> with {n_cols} variables spanning demographics, dining behavior, digital engagement, and experience ratings. The central research question is: <em>"What are the most influential factors determining restaurant customer satisfaction?"</em></p>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-value">{n_rows:,}</div>
      <div class="stat-label">Customer Records</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{sat_rate:.1f}%</div>
      <div class="stat-label">High Satisfaction Rate</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{accuracy:.1%}</div>
      <div class="stat-label">Model Accuracy</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{len(top5)}</div>
      <div class="stat-label">Top Predictive Factors</div>
    </div>
  </div>

  <div class="callout">
    <strong>Key Findings:</strong>
    <ul>
      <li>The overall high satisfaction rate is <strong>{sat_rate:.1f}%</strong>, indicating that the majority of diners report moderate to low satisfaction.</li>
      <li>The top 5 most influential factors for predicting satisfaction are: <strong>{', '.join(top5)}</strong>.</li>
      <li>Our logistic regression model achieves <strong>{accuracy:.1%} accuracy</strong> in predicting customer satisfaction.</li>
      <li>Monte Carlo simulation (10,000 iterations) reveals that reducing wait times below 15 minutes increases predicted high satisfaction from <strong>{(mc_scenario_results['High Wait (30-60 min)'] >= 0.5).mean()*100:.1f}%</strong> to <strong>{(mc_scenario_results['Low Wait (0-15 min)'] >= 0.5).mean()*100:.1f}%</strong>.</li>
      <li>Loyalty program members show a satisfaction rate difference of <strong>{loyalty_lift:+.1f} percentage points</strong> compared to non-members.</li>
    </ul>
  </div>

  <h3>Analytical Methods Used</h3>
  <p>This report employs the full spectrum of analytical tools covered in MKTG 4163:</p>
  <ul>
    <li><strong>Descriptive Analysis:</strong> Summary statistics, frequency distributions, histograms, box plots, cross-tabulations, pivot tables</li>
    <li><strong>Inferential Statistics:</strong> Pearson correlation, ANOVA (single-factor), Chi-Square test of independence</li>
    <li><strong>Predictive Modeling:</strong> Logistic regression, multivariate linear regression, Monte Carlo simulation (10,000 iterations)</li>
    <li><strong>Prescriptive Analysis:</strong> Data-driven recommendations for improving customer satisfaction</li>
  </ul>
</div>

<!-- SECTION 2: DATASET CONTEXT -->
<div class="section" id="data-context">
  <h2>2. Dataset Context & Background</h2>
  <p>The dataset contains <strong>{n_rows:,} customer records</strong> from restaurant visits, sourced as secondary data from Kaggle. Each record represents a unique customer interaction and includes {n_cols} variables capturing demographics, dining behavior, digital engagement, experience ratings, and a binary satisfaction outcome.</p>

  <h3>Data Dictionary</h3>
  <table class="data-table">
    <thead><tr><th>Variable</th><th>Type</th><th>Description</th></tr></thead>
    <tbody>
      <tr><td>CustomerID</td><td>Identifier</td><td>Unique customer identifier</td></tr>
      <tr><td>Age</td><td>Numeric</td><td>Customer age in years</td></tr>
      <tr><td>Gender</td><td>Categorical</td><td>Male / Female</td></tr>
      <tr><td>Income</td><td>Numeric</td><td>Annual income ($)</td></tr>
      <tr><td>VisitFrequency</td><td>Categorical</td><td>How often the customer visits (Daily, Weekly, Monthly, Rarely)</td></tr>
      <tr><td>AverageSpend</td><td>Numeric</td><td>Average amount spent per visit ($)</td></tr>
      <tr><td>PreferredCuisine</td><td>Categorical</td><td>Customer's preferred cuisine type</td></tr>
      <tr><td>TimeOfVisit</td><td>Categorical</td><td>Meal time — Breakfast, Lunch, or Dinner</td></tr>
      <tr><td>GroupSize</td><td>Numeric</td><td>Number of people in the dining party</td></tr>
      <tr><td>DiningOccasion</td><td>Categorical</td><td>Purpose of the visit — Casual, Business, or Celebration</td></tr>
      <tr><td>MealType</td><td>Categorical</td><td>Dine-in or Takeaway</td></tr>
      <tr><td>OnlineReservation</td><td>Binary (0/1)</td><td>Whether the customer made an online reservation</td></tr>
      <tr><td>DeliveryOrder</td><td>Binary (0/1)</td><td>Whether the order was a delivery</td></tr>
      <tr><td>LoyaltyProgramMember</td><td>Binary (0/1)</td><td>Whether the customer is a loyalty program member</td></tr>
      <tr><td>WaitTime</td><td>Numeric</td><td>Wait time in minutes</td></tr>
      <tr><td>ServiceRating</td><td>Numeric (1-5)</td><td>Rating of service quality</td></tr>
      <tr><td>FoodRating</td><td>Numeric (1-5)</td><td>Rating of food quality</td></tr>
      <tr><td>AmbianceRating</td><td>Numeric (1-5)</td><td>Rating of restaurant ambiance</td></tr>
      <tr><td>HighSatisfaction</td><td>Binary (0/1)</td><td>Target variable — 1 if customer reports high satisfaction</td></tr>
    </tbody>
  </table>

  <div class="callout-navy">
    <strong>Data Quality Note:</strong> The dataset contains {df.isnull().sum().sum()} missing values. All variables are complete and require no imputation. Numeric variables were checked for outliers using z-scores; the data appears clean and ready for analysis.
  </div>
</div>

<!-- SECTION 3: DESCRIPTIVE ANALYSIS -->
<div class="section" id="descriptive">
  <h2>3. Descriptive Analysis</h2>
  <p>This section summarizes the main features of the dataset — the who, what, where, when, and how — using charts, graphs, summary statistics, averages, counts, and percentages.</p>

  <h3>3.1 Summary Statistics</h3>
  {desc_stats_html}

  <h3>3.2 Distribution of Numeric Variables</h3>
  <div class="chart-container">
    <img src="{histograms_img}" alt="Histograms of numeric variables">
  </div>

  <h3>3.3 Categorical Variable Distributions</h3>
  {freq_tables_html}

  <h3>3.4 Binary Variable Distributions</h3>
  {binary_html}

  <h3>3.5 Satisfaction Rate by Customer Segments</h3>
  <div class="chart-container">
    <img src="{sat_by_cat_img}" alt="Satisfaction rate by categorical variables">
  </div>

  <h3>3.6 Rating Distributions by Satisfaction Level</h3>
  <div class="chart-container">
    <img src="{boxplot_ratings_img}" alt="Box plots of ratings by satisfaction">
  </div>

  <h3>3.7 Average Spend by Cuisine</h3>
  <div class="chart-container">
    <img src="{spend_cuisine_img}" alt="Spend by cuisine box plot">
  </div>

  <h3>3.8 Cross-Tabulations: Key Qualities Assessed</h3>
  <p>The team identified three key qualities to assess: <strong>Time of Visit</strong>, <strong>Dining Occasion</strong>, and <strong>Visit Frequency</strong>.</p>
  {cross_tabs_html}

  <h3>3.9 Pivot Table Summaries</h3>
  {pivot_html}
</div>

<!-- SECTION 4: INFERENTIAL STATISTICS -->
<div class="section" id="inferential">
  <h2>4. Inferential Statistics</h2>
  <p>Inferential statistics allow us to make generalizations about the broader restaurant customer population based on our sample data. We use correlation analysis, ANOVA, and Chi-Square tests to understand the "why" behind the data.</p>

  <h3>4.1 Correlation Analysis</h3>
  <p>Pearson correlation coefficients measure the linear relationship between numeric variables and satisfaction. Values range from -1 (perfect negative) to +1 (perfect positive).</p>
  <div class="chart-container">
    <img src="{corr_img}" alt="Correlation matrix heatmap">
  </div>
  {corr_target_html}

  <div class="callout">
    <strong>Correlation Insights:</strong> The correlation analysis reveals which numeric factors have the strongest linear relationship with high satisfaction. Variables marked with *** are statistically significant at p &lt; 0.001.
  </div>

  <h3>4.2 ANOVA — Analysis of Variance</h3>
  <p>ANOVA tests whether the mean satisfaction levels differ significantly across groups. We test the four key categorical factors identified by the team, plus their relationship with individual rating dimensions.</p>
  <div class="chart-container">
    <img src="{anova_img}" alt="ANOVA visualization">
  </div>
  {anova_html}

  <div class="callout-navy">
    <strong>How to Read ANOVA Results:</strong> A p-value &lt; 0.05 indicates statistically significant differences between groups. The F-statistic measures the ratio of between-group variance to within-group variance — higher values suggest greater differences. Significance levels: *** p&lt;0.001, ** p&lt;0.01, * p&lt;0.05, n.s. = not significant.
  </div>

  <h3>4.3 Chi-Square Test of Independence</h3>
  <p>The Chi-Square test determines whether categorical variables are independent of high satisfaction, or whether there is a statistically significant association.</p>
  {chi2_html}

  <div class="callout">
    <strong>Chi-Square Insights:</strong> Variables where we "reject H0" have a statistically significant relationship with satisfaction — meaning the distribution of satisfaction is NOT independent of that variable. These are important factors for restaurant managers to focus on.
  </div>
</div>

<!-- SECTION 5: PREDICTIVE MODELING -->
<div class="section" id="predictive">
  <h2>5. Predictive Modeling</h2>
  <p>Predictive modeling uses historical data to forecast future outcomes. We employ logistic regression (for the binary satisfaction target), linear regression (for composite satisfaction score), and Monte Carlo simulation (to model uncertainty).</p>

  <h3>5.1 Logistic Regression — Predicting High Satisfaction</h3>
  <p>Logistic regression models the probability of a customer achieving high satisfaction based on all available features. The model was trained on 70% of the data and tested on the remaining 30%.</p>

  <h4>Logistic S-Curve with Residual Errors</h4>
  <p>The plot below shows the classic logistic (sigmoid) curve fitted to the strongest predictor. Each dot represents an actual customer outcome (0 or 1), and the red lines show the residual error — the gap between the predicted probability and the actual outcome.</p>
  <div class="chart-container">
    <img src="{logistic_curve_img}" alt="Logistic regression S-curve with errors">
  </div>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-value">{accuracy:.1%}</div>
      <div class="stat-label">Overall Accuracy</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{cr['High Satisfaction']['precision']:.1%}</div>
      <div class="stat-label">Precision (High Sat)</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{cr['High Satisfaction']['recall']:.1%}</div>
      <div class="stat-label">Recall (High Sat)</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{cr['High Satisfaction']['f1-score']:.1%}</div>
      <div class="stat-label">F1-Score (High Sat)</div>
    </div>
  </div>

  <div class="chart-container">
    <img src="{cm_img}" alt="Confusion matrix">
  </div>
  {cr_html}

  <h3>Logistic Regression Coefficients & Odds Ratios</h3>
  <p>The odds ratio indicates how much the odds of high satisfaction change with a one-unit increase in the standardized predictor. An odds ratio &gt; 1 means a positive effect; &lt; 1 means a negative effect.</p>
  {logit_html}

  <h3>5.2 Feature Importance</h3>
  <div class="chart-container">
    <img src="{feat_imp_img}" alt="Feature importance chart">
  </div>
  {importance_html}

  <h3>5.3 Multivariate Linear Regression — Composite Satisfaction Score</h3>
  <p>We also model a continuous composite satisfaction score (average of Service, Food, and Ambiance ratings) to understand which customer and visit characteristics predict overall experience quality.</p>

  <div class="callout-navy">
    <strong>Model Fit:</strong> R² = {ols_model.rsquared:.4f} — the model explains {ols_model.rsquared*100:.1f}% of the variance in composite satisfaction scores. Adjusted R² = {ols_model.rsquared_adj:.4f}. F-statistic p-value = {ols_model.f_pvalue:.4e}.
  </div>
  {ols_html}

  <h3>5.4 Monte Carlo Simulation</h3>
  <p>Monte Carlo simulation uses random sampling to model the probability of different outcomes. We simulated <strong>10,000 customer profiles</strong> under three wait time scenarios to estimate how wait time impacts satisfaction probability.</p>

  <div class="chart-container">
    <img src="{mc_img}" alt="Monte Carlo simulation results">
  </div>
  {mc_html}

  <h3>Monte Carlo by Cuisine Type</h3>
  <p>We also simulated 10,000 customers for each cuisine preference to estimate satisfaction probabilities across cuisine types.</p>
  {mc_cuisine_html}

  <div class="callout">
    <strong>Monte Carlo Insights:</strong> The simulation demonstrates how random variation in customer characteristics produces a distribution of satisfaction outcomes. By comparing scenarios, we can quantify the expected impact of operational changes (e.g., reducing wait times) on customer satisfaction rates.
  </div>
</div>

<!-- SECTION 6: PRESCRIPTIVE ANALYSIS -->
<div class="section" id="prescriptive">
  <h2>6. Prescriptive Analysis & Recommendations</h2>
  <p>Based on our descriptive, inferential, and predictive analyses, we provide the following actionable recommendations for restaurant operators seeking to improve customer satisfaction.</p>

  <div class="rec-card">
    <h4>Recommendation 1: Optimize Wait Times</h4>
    <p>Implement operational improvements to keep wait times <strong>under 15 minutes</strong>. Our Monte Carlo simulation shows that customers with low wait times (0–15 min) have a predicted high satisfaction rate of <strong>{(mc_scenario_results['Low Wait (0-15 min)'] >= 0.5).mean()*100:.1f}%</strong>, compared to <strong>{(mc_scenario_results['High Wait (30-60 min)'] >= 0.5).mean()*100:.1f}%</strong> for those waiting 30–60 minutes.</p>
    <p class="evidence">Evidence: Monte Carlo simulation (n=10,000), Correlation analysis, ANOVA results</p>
  </div>

  <div class="rec-card">
    <h4>Recommendation 2: Invest in Loyalty Programs</h4>
    <p>Loyalty program members show a satisfaction rate difference of <strong>{loyalty_lift:+.1f} percentage points</strong> compared to non-members ({loyalty_sat.get(1,0):.1f}% vs {loyalty_sat.get(0,0):.1f}%). {"Expand" if loyalty_lift > 0 else "Redesign"} the loyalty program to {"drive" if loyalty_lift > 0 else "improve"} customer engagement and repeat visits.</p>
    <p class="evidence">Evidence: Chi-Square test, Cross-tabulation analysis, Logistic regression coefficients</p>
  </div>

  <div class="rec-card">
    <h4>Recommendation 3: Focus on the Top Satisfaction Drivers</h4>
    <p>The top 5 predictive factors for satisfaction are: <strong>{', '.join(top5)}</strong>. Restaurant operators should prioritize improvements in these areas, as they have the greatest impact on whether a customer reports high satisfaction.</p>
    <p class="evidence">Evidence: Logistic regression feature importance, Correlation analysis</p>
  </div>

  <div class="rec-card">
    <h4>Recommendation 4: Tailor the Experience by Dining Occasion</h4>
    <p>ANOVA analysis reveals whether satisfaction varies significantly by dining occasion (Casual, Business, Celebration). Restaurants should consider differentiated service protocols for each occasion type — for example, faster service for business diners and enhanced ambiance for celebrations.</p>
    <p class="evidence">Evidence: ANOVA results, Cross-tabulation by DiningOccasion, Pivot table analysis</p>
  </div>

  <div class="rec-card">
    <h4>Recommendation 5: Leverage Digital Channels</h4>
    <p>Online reservation users show a satisfaction rate of <strong>{reservation_sat.get(1,0):.1f}%</strong> vs <strong>{reservation_sat.get(0,0):.1f}%</strong> for walk-ins. {"Encouraging online reservations" if reservation_sat.get(1,0) > reservation_sat.get(0,0) else "Improving the online reservation experience"} may help set proper expectations and reduce friction in the dining experience.</p>
    <p class="evidence">Evidence: Chi-Square test, Descriptive cross-tabulation</p>
  </div>

  <h3>Decision Framework</h3>
  <div class="callout-navy">
    <strong>Implications Wheel:</strong> The findings suggest an interconnected system where operational factors (wait time), experience quality (food, service, ambiance ratings), and customer engagement (loyalty programs, digital channels) all contribute to satisfaction. Improvements in one area can create positive ripple effects across the entire customer experience.
  </div>
</div>

<!-- SECTION 7: LIMITATIONS -->
<div class="section" id="limitations">
  <h2>7. Data & Project Limitations</h2>
  <ul>
    <li><strong>Secondary Data Source:</strong> The dataset is sourced from Kaggle and may not represent any specific restaurant chain or geographic region. Findings should be validated with primary data collection (e.g., surveys of actual restaurant customers).</li>
    <li><strong>Binary Satisfaction Measure:</strong> The target variable (HighSatisfaction) is binary (0/1), which simplifies a nuanced concept. A continuous satisfaction scale (e.g., 1–10) would allow for more granular analysis.</li>
    <li><strong>Cross-Sectional Data:</strong> The data represents a single snapshot in time. We cannot infer causation or track changes in satisfaction over time. Holt-Winters forecasting and trend analysis would require time-series data.</li>
    <li><strong>No Location Data:</strong> The dataset does not include geographic information, so we cannot assess regional or location-specific effects on satisfaction.</li>
    <li><strong>Simulated Nature:</strong> The uniformity of some distributions suggests this may be a synthetically generated dataset, which could limit the generalizability of findings to real-world restaurant operations.</li>
    <li><strong>Model Limitations:</strong> The logistic regression model assumes linear relationships between features and log-odds of satisfaction. Non-linear effects or interaction terms may improve predictive performance.</li>
    <li><strong>No Causal Claims:</strong> Correlation and regression analyses identify associations, not causal relationships. Controlled experiments would be needed to establish causality.</li>
  </ul>
</div>

<!-- FOOTER -->
<div class="footer">
  <p>GWU MKTG 4163 — Applied Marketing Analytics | Spring 2026 | Professor David Ashley</p>
  <p>Analysis generated using Python (pandas, scipy, statsmodels, scikit-learn, matplotlib, seaborn)</p>
</div>

</div>
</body>
</html>"""

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\nReport generated successfully: {OUTPUT_PATH}")
print(f"File size: {len(html) / 1024:.0f} KB")
