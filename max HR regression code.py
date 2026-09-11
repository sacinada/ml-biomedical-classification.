# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:54:48 2026

@author: sacin
"""

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from itertools import combinations
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor']   = 'white'
plt.rcParams['axes.grid']        = True
plt.rcParams['grid.alpha']       = 0.3


# Load Data

df = pd.read_csv(r'C:\Users\sacin\Downloads\Heart_Disease_Prediction.csv')

df = df.rename(columns={
    'Chest pain type'        : 'Chest_pain',
    'FBS over 120'           : 'FBS',
    'EKG results'            : 'EKG',
    'Max HR'                 : 'Max_HR',
    'Exercise angina'        : 'Ex_angina',
    'ST depression'          : 'ST_dep',
    'Slope of ST'            : 'Slope_ST',
    'Number of vessels fluro': 'Vessels',
    'Heart Disease'          : 'Heart_Disease'
})

print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("Missing values:", df.isnull().sum().sum())
print(df.head())


# Visualisation of Target Variable

fig, axes = plt.subplots(1, 3, figsize=(17, 4))

axes[0].hist(df['Max_HR'], bins=30, color='steelblue', edgecolor='black', alpha=0.8)
axes[0].axvline(df['Max_HR'].mean(),   color='red',    linestyle='--', linewidth=2, label=f"Mean={df['Max_HR'].mean():.1f}")
axes[0].axvline(df['Max_HR'].median(), color='orange', linestyle='--', linewidth=2, label=f"Median={df['Max_HR'].median():.1f}")
axes[0].set_title('Distribution of Max HR', fontweight='bold')
axes[0].set_xlabel('Max HR (bpm)'); axes[0].set_ylabel('Number of observations')
axes[0].legend()

stats.probplot(df['Max_HR'], dist='norm', plot=axes[1])
axes[1].set_title('Q-Q Plot of Max HR', fontweight='bold')

axes[2].boxplot(df['Max_HR'], patch_artist=True, boxprops=dict(facecolor='steelblue', alpha=0.6))
axes[2].set_title('Max HR outlier detection', fontweight='bold')
q1  = df['Max_HR'].quantile(0.25)
q3v = df['Max_HR'].quantile(0.75)
n_out = ((df['Max_HR'] < q1-1.5*(q3v-q1)) | (df['Max_HR'] > q3v+1.5*(q3v-q1))).sum()
axes[2].set_xlabel(f'IQR outliers: {n_out:,}', fontsize=9, color='gray')

plt.suptitle('Target Variable Max HR: Distribution Check', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('HD_00_maxhr_visualisation.png', dpi=150)
plt.show()


# Log Transform Check

print(f"Max HR skewness = {stats.skew(df['Max_HR']):.4f}")
print(f"Max HR mean     = {df['Max_HR'].mean():.2f}")
print(f"Max HR median   = {df['Max_HR'].median():.2f}")
print(f"Max HR zeros    = {(df['Max_HR']==0).sum()}")


# Train / Test Split

train_data, test_data = train_test_split(df, test_size=0.30, random_state=123)
train_data = train_data.reset_index(drop=True)
test_data  = test_data.reset_index(drop=True)

print(f"Training set : {len(train_data):,} patients")
print(f"Test set     : {len(test_data):,} patients")


# Correlation Check

data_cor_cols = ['Age', 'BP', 'Cholesterol', 'Max_HR', 'ST_dep',
                 'Ex_angina', 'Chest_pain', 'Slope_ST', 'Vessels', 'Thallium']

plt.figure(figsize=(10, 8))
sns.heatmap(df[data_cor_cols].corr(), annot=True, fmt='.2f',
            cmap='coolwarm', square=True, linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig('HD_01_corrplot.png', dpi=150)
plt.show()

candidates = ['Age', 'BP', 'Cholesterol', 'ST_dep',
              'Ex_angina', 'Chest_pain', 'Slope_ST', 'Vessels', 'Thallium']


# Exhaustive Best-Subset Selection

best_results = []

for k in range(1, len(candidates)+1):
    best_adjr2 = -np.inf
    best_sub   = None
    for subset in combinations(candidates, k):
        formula   = 'Max_HR ~ ' + ' + '.join(subset)
        model_sub = smf.ols(formula, data=train_data).fit()
        if model_sub.rsquared_adj > best_adjr2:
            best_adjr2 = model_sub.rsquared_adj
            best_sub   = list(subset)
    best_results.append({'n': k, 'vars': best_sub, 'adj_r2': round(best_adjr2, 5)})
    print(f"  {k} var(s): Adj R²={best_adjr2:.5f}  {best_sub}")

best_df  = pd.DataFrame(best_results)
best_row = best_df.loc[best_df['adj_r2'].idxmax()]
best_vars = best_row['vars']
print(f"Best subset: {best_vars}  Adj R²={best_row['adj_r2']:.5f}")

plt.figure(figsize=(9, 4))
plt.plot(best_df['n'], best_df['adj_r2'], marker='o', color='steelblue', linewidth=2, markersize=7)
for _, row in best_df.iterrows():
    plt.annotate(f"{row['adj_r2']:.4f}", (row['n'], row['adj_r2']),
                 textcoords='offset points', xytext=(4, 6), fontsize=8)
plt.xlabel('Number of variables'); plt.ylabel('Adjusted R²')
plt.title('Exhaustive Best-Subset Adj R² by Model Size', fontweight='bold')
plt.xticks(range(1, len(candidates)+1))
plt.tight_layout()
plt.savefig('HD_02_regsubsets_adjr2.png', dpi=150)
plt.show()


# Chosen Model and Scaled Model

formula_main = 'Max_HR ~ ' + ' + '.join(best_vars)
model        = smf.ols(formula_main, data=train_data).fit()
print(model.summary())

sc_vars  = best_vars + ['Max_HR']
sc_train = train_data[sc_vars].copy()
sc_train = (sc_train - sc_train.mean()) / sc_train.std()
model_scaled = smf.ols(formula_main, data=sc_train).fit()
print(model_scaled.summary())

imp = model_scaled.params.drop('Intercept').abs().sort_values(ascending=False)
for v, val in imp.items():
    p = model_scaled.pvalues[v]
    print(f"  {v:<12}: |β|={val:.4f}  p={p:.4e}  {'✓' if p<0.05 else '✗'}")

predictions = model.predict(test_data)
actual      = test_data['Max_HR']
mn = min(actual.min(), predictions.min()) - 2
mx = max(actual.max(), predictions.max()) + 2

plt.figure(figsize=(7, 5))
plt.scatter(actual, predictions, alpha=0.4, color='steelblue', s=20)
plt.plot([mn,mx],[mn,mx], color='red', linewidth=2, label='Perfect prediction (y=x)')
plt.xlabel('Actual Max HR'); plt.ylabel('Predicted Max HR')
plt.title('Predictions vs Actual Exhaustive Model', fontweight='bold')
plt.legend(); plt.tight_layout()
plt.savefig('HD_03_pred_vs_actual_exhaustive.png', dpi=150)
plt.show()


# Forward Selection

current_fwd = []
remaining   = candidates.copy()

for _ in range(len(candidates)):
    best_pval = np.inf
    best_var  = None
    for v in remaining:
        test_formula = 'Max_HR ~ ' + ' + '.join(current_fwd + [v])
        test_model   = smf.ols(test_formula, data=train_data).fit()
        p_val = test_model.pvalues[v]
        if p_val < best_pval:
            best_pval = p_val
            best_var  = v
    if best_var and best_pval < 0.05:
        current_fwd.append(best_var)
        remaining.remove(best_var)
        m_t = smf.ols('Max_HR ~ ' + ' + '.join(current_fwd), data=train_data).fit()
        print(f"  + {best_var:<15}  p={best_pval:.4e}  Adj R²={m_t.rsquared_adj:.5f}")
    else:
        print("  STOP no remaining variable has p < 0.05")
        break

formula_fwd   = 'Max_HR ~ ' + ' + '.join(current_fwd)
model_forward = smf.ols(formula_fwd, data=train_data).fit()
print(model_forward.summary())

predictions_forward = model_forward.predict(test_data)
plt.figure(figsize=(7, 5))
plt.scatter(actual, predictions_forward, alpha=0.4, color='seagreen', s=20)
plt.plot([mn,mx],[mn,mx], color='red', linewidth=2, label='y=x')
plt.xlabel('Actual Max HR'); plt.ylabel('Predicted Max HR')
plt.title('Forward Selection: Predictions vs Actual', fontweight='bold')
plt.legend(); plt.tight_layout()
plt.savefig('HD_04_pred_vs_actual_forward.png', dpi=150)
plt.show()


# Backward Selection

current_bwd = candidates.copy()

for _ in range(len(candidates)):
    m_t   = smf.ols('Max_HR ~ ' + ' + '.join(current_bwd), data=train_data).fit()
    pvals = m_t.pvalues[current_bwd]
    worst = pvals.idxmax()
    wp    = pvals.max()
    if wp > 0.05:
        current_bwd.remove(worst)
        m_n = smf.ols('Max_HR ~ ' + ' + '.join(current_bwd), data=train_data).fit()
        print(f"  - {worst:<15}  p={wp:.4f}  Adj R²={m_n.rsquared_adj:.5f}")
    else:
        print("  STOP all remaining variables have p < 0.05")
        break

formula_bwd    = 'Max_HR ~ ' + ' + '.join(current_bwd)
model_backward = smf.ols(formula_bwd, data=train_data).fit()
print(model_backward.summary())

predictions_backward = model_backward.predict(test_data)
plt.figure(figsize=(7, 5))
plt.scatter(actual, predictions_backward, alpha=0.4, color='coral', s=20)
plt.plot([mn,mx],[mn,mx], color='red', linewidth=2, label='y=x')
plt.xlabel('Actual Max HR'); plt.ylabel('Predicted Max HR')
plt.title('Backward Selection: Predictions vs Actual', fontweight='bold')
plt.legend(); plt.tight_layout()
plt.savefig('HD_05_pred_vs_actual_backward.png', dpi=150)
plt.show()


# Performance Table

def evaluate_model(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2   = r2_score(actual, predicted)
    return {'RMSE': round(rmse, 4), 'R2': round(r2, 4)}

pred_init     = model.predict(test_data)
pred_forward  = model_forward.predict(test_data)
pred_backward = model_backward.predict(test_data)

perf_init     = evaluate_model(actual, pred_init)
perf_forward  = evaluate_model(actual, pred_forward)
perf_backward = evaluate_model(actual, pred_backward)

perf_table = pd.DataFrame({
    'Model': ['Exhaustive', 'Forward', 'Backward'],
    'RMSE' : [perf_init['RMSE'], perf_forward['RMSE'], perf_backward['RMSE']],
    'R2'   : [perf_init['R2'],   perf_forward['R2'],   perf_backward['R2']],
})
print(perf_table.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, preds, title, col in zip(
        axes,
        [pred_init, pred_forward, pred_backward],
        ['Exhaustive', 'Forward', 'Backward'],
        ['steelblue', 'seagreen', 'coral']):
    ax.scatter(actual, preds, alpha=0.4, color=col, s=20)
    ax.plot([mn,mx],[mn,mx], color='red', linewidth=2)
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.set_title(title, fontweight='bold')
    perf = evaluate_model(actual, preds)
    ax.annotate(f"R²={perf['R2']:.4f}\nRMSE={perf['RMSE']}",
                xy=(0.05,0.83), xycoords='axes fraction', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.suptitle('Comparison Predictions vs Actual', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('HD_06_comparison_3panel.png', dpi=150)
plt.show()


# Residual Diagnostics

fitted_bwd = model_backward.fittedvalues
resid_bwd  = model_backward.resid
mean_resid = resid_bwd.mean()
med_resid  = resid_bwd.median()

print(f"Mean of residuals   : {mean_resid:.2e}")
print(f"Median of residuals : {med_resid:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

lowess_bwd = sm.nonparametric.lowess(resid_bwd, fitted_bwd, frac=0.2)
axes[0].scatter(fitted_bwd, resid_bwd, alpha=0.3, color='steelblue', s=10)
axes[0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0].plot(lowess_bwd[:,0], lowess_bwd[:,1], color='orange', linewidth=2.5, label='LOWESS trend')
axes[0].set_xlabel('Fitted Values'); axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Fitted Values', fontweight='bold')
axes[0].legend()

axes[1].hist(resid_bwd, bins=30, color='steelblue', edgecolor='black', alpha=0.8, density=True)
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
axes[1].axvline(mean_resid, color='orange', linestyle='--', linewidth=2, label=f'Mean={mean_resid:.2f}')
xn = np.linspace(resid_bwd.min(), resid_bwd.max(), 300)
axes[1].plot(xn, stats.norm.pdf(xn, mean_resid, resid_bwd.std()), color='black', linewidth=1.5, label='Normal curve')
axes[1].set_xlabel('Residuals'); axes[1].set_ylabel('Density')
axes[1].set_title('Histogram of Residuals', fontweight='bold')
axes[1].legend(fontsize=8)

plt.suptitle('Residual Diagnostics', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('HD_07_residual_diagnostics.png', dpi=150)
plt.show()


# Polynomial Regression

train_data['Age_sq'] = train_data['Age'] ** 2
test_data['Age_sq']  = test_data['Age']  ** 2

poly_vars    = best_vars + ['Age_sq']
formula_poly = 'Max_HR ~ ' + ' + '.join(poly_vars)
model_poly   = smf.ols(formula_poly, data=train_data).fit()
print(model_poly.summary())

b_age_sq = model_poly.params.get('Age_sq', np.nan)
p_age_sq = model_poly.pvalues.get('Age_sq', np.nan)

print(f"Age²  : β = {b_age_sq:.6e}  p = {p_age_sq:.4e}")
print(f"Backward Adj R²  : {model_backward.rsquared_adj:.5f}")
print(f"Polynomial Adj R²: {model_poly.rsquared_adj:.5f}")
print(f"Gain             : {(model_poly.rsquared_adj-model_backward.rsquared_adj)*100:+.3f} pp")

age_seq     = np.linspace(train_data['Age'].min(), train_data['Age'].max(), 300)
other_means = {v: train_data[v].mean() for v in best_vars if v != 'Age'}

lin_line = model_backward.params['Intercept'] + model_backward.params['Age'] * age_seq
for v, m in other_means.items():
    if v in model_backward.params:
        lin_line += model_backward.params[v] * m

poly_line = model_poly.params['Intercept'] + \
            model_poly.params['Age'] * age_seq + \
            model_poly.params['Age_sq'] * age_seq**2
for v, m in other_means.items():
    if v in model_poly.params:
        poly_line += model_poly.params[v] * m

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(train_data['Age'], train_data['Max_HR'], alpha=0.2, color='steelblue', s=15, label='Data')
axes[0].plot(age_seq, lin_line,  color='orange', linewidth=2, linestyle='--', label='Linear')
axes[0].plot(age_seq, poly_line, color='red',    linewidth=2.5, label='Polynomial (+Age²)')
axes[0].set_xlabel('Age'); axes[0].set_ylabel('Max HR')
axes[0].set_title('Linear vs Polynomial Fit for Age', fontweight='bold')
axes[0].legend(fontsize=8)

resid_poly  = model_poly.resid
fitted_poly = model_poly.fittedvalues
mask_poly   = (fitted_poly > fitted_poly.quantile(0.01)) & (fitted_poly < fitted_poly.quantile(0.99))
lowess_poly = sm.nonparametric.lowess(resid_poly[mask_poly], fitted_poly[mask_poly], frac=0.2)
axes[1].scatter(fitted_poly, resid_poly, alpha=0.3, color='steelblue', s=10)
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].plot(lowess_poly[:,0], lowess_poly[:,1], color='orange', linewidth=2.5, label='LOWESS trend')
axes[1].set_xlabel('Fitted Values'); axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals vs Fitted Polynomial Model', fontweight='bold')
axes[1].legend()

plt.suptitle('Polynomial Regression: Age + Age²', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('HD_08_polynomial_regression.png', dpi=150)
plt.show()


# Interaction Model

other_int      = [v for v in best_vars if v not in ['Age', 'ST_dep']]
formula_int    = ('Max_HR ~ Age * ST_dep' + ((' + ' + ' + '.join(other_int)) if other_int else ''))
model_interact = smf.ols(formula_int, data=train_data).fit()
print(model_interact.summary())

int_key = 'Age:ST_dep'
b_int   = model_interact.params.get(int_key, np.nan)
p_int   = model_interact.pvalues.get(int_key, np.nan)

print(f"Age x ST_dep: β = {b_int:.6e}  p = {p_int:.4e}")
print(f"Backward Adj R²   : {model_backward.rsquared_adj:.5f}")
print(f"Interaction Adj R²: {model_interact.rsquared_adj:.5f}")
print(f"Gain              : {(model_interact.rsquared_adj-model_backward.rsquared_adj)*100:+.3f} pp")

fitted_int = model_interact.fittedvalues
resid_int  = model_interact.resid

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

st_terciles = pd.qcut(train_data['ST_dep'], q=3, labels=['Low ST_dep', 'Mid ST_dep', 'High ST_dep'])
colors_t    = {'Low ST_dep': 'steelblue', 'Mid ST_dep': 'seagreen', 'High ST_dep': 'coral'}
for grp, col in colors_t.items():
    mask = st_terciles == grp
    axes[0].scatter(train_data.loc[mask, 'Age'], train_data.loc[mask, 'Max_HR'],
                    alpha=0.2, color=col, s=10, label=grp)
    sub = train_data[mask]
    slope, intercept = stats.linregress(sub['Age'], sub['Max_HR'])[:2]
    x_g = np.linspace(sub['Age'].min(), sub['Age'].max(), 100)
    axes[0].plot(x_g, intercept + slope*x_g, color=col, linewidth=2)
axes[0].set_xlabel('Age'); axes[0].set_ylabel('Max HR')
axes[0].set_title('Interaction: Age effect on Max HR by ST depression group', fontweight='bold')
axes[0].legend(fontsize=8)

mask_int   = (fitted_int > fitted_int.quantile(0.01)) & (fitted_int < fitted_int.quantile(0.99))
lowess_int = sm.nonparametric.lowess(resid_int[mask_int], fitted_int[mask_int], frac=0.2)
axes[1].scatter(fitted_int, resid_int, alpha=0.3, color='steelblue', s=10)
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].plot(lowess_int[:,0], lowess_int[:,1], color='orange', linewidth=2.5, label='LOWESS trend')
axes[1].set_xlabel('Fitted Values'); axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals vs Fitted Interaction Model', fontweight='bold')
axes[1].legend()

plt.suptitle('Interaction Model: Age x ST depression', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('HD_09_interaction_model.png', dpi=150)
plt.show()


# Outlier Detection and Clean Model

influence    = model_backward.get_influence()
std_resid    = influence.resid_studentized_internal
outlier_mask = np.abs(std_resid) > 2
n_outliers   = outlier_mask.sum()

print(f"Training patients  : {len(train_data):,}")
print(f"Flagged (|e*| > 2) : {n_outliers:,} ({100*n_outliers/len(train_data):.1f}%)")
print(f"Clean training set : {len(train_data)-n_outliers:,} patients")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

idx_arr = np.arange(len(std_resid))
axes[0].scatter(idx_arr[~outlier_mask], std_resid[~outlier_mask], alpha=0.3, color='steelblue', s=5, label='Normal')
axes[0].scatter(idx_arr[outlier_mask],  std_resid[outlier_mask],  color='red', s=30, zorder=5, label=f'Outliers (n={n_outliers:,})')
axes[0].axhline( 2, color='red', linestyle='--', linewidth=1.5, label='+-2 threshold')
axes[0].axhline(-2, color='red', linestyle='--', linewidth=1.5)
axes[0].set_xlabel('Observation Index'); axes[0].set_ylabel('Studentized Residual')
axes[0].set_title('Studentized Residuals', fontweight='bold')
axes[0].legend(fontsize=8)

train_clean  = train_data[~outlier_mask].copy().reset_index(drop=True)
model_clean  = smf.ols(formula_bwd, data=train_clean).fit()

clean_fitted = model_clean.fittedvalues
clean_resid  = model_clean.resid
q01 = clean_fitted.quantile(0.01)
q99 = clean_fitted.quantile(0.99)
mask_cl   = (clean_fitted > q01) & (clean_fitted < q99)
lowess_cl = sm.nonparametric.lowess(clean_resid[mask_cl], clean_fitted[mask_cl], frac=0.2)
axes[1].scatter(clean_fitted, clean_resid, alpha=0.3, color='seagreen', s=10)
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].plot(lowess_cl[:,0], lowess_cl[:,1], color='orange', linewidth=2.5, label='LOWESS trend')
axes[1].set_xlim(q01, q99)
axes[1].set_xlabel('Fitted Values (central 98%)'); axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals vs Fitted Clean Model', fontweight='bold')
axes[1].legend()

plt.suptitle('Outlier Detection and Clean Model', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('HD_10_outlier_detection.png', dpi=150)
plt.show()

pred_clean = model_clean.predict(test_data)
perf_clean = evaluate_model(test_data['Max_HR'], pred_clean)
perf_back  = evaluate_model(test_data['Max_HR'], model_backward.predict(test_data))

print(f"Backward model  Test R²: {perf_back['R2']}   RMSE: {perf_back['RMSE']}")
print(f"Clean model     Test R²: {perf_clean['R2']}   RMSE: {perf_clean['RMSE']}")
print(f"Train R² original : {model_backward.rsquared:.5f}")
print(f"Train R² clean    : {model_clean.rsquared:.5f}")


# Final Comparison

pred_poly = model_poly.predict(test_data)
pred_int  = model_interact.predict(test_data)

perf_poly = evaluate_model(test_data['Max_HR'], pred_poly)
perf_int  = evaluate_model(test_data['Max_HR'], pred_int)

all_models = pd.DataFrame({
    'Model': [
        'Exhaustive',
        'Forward',
        'Backward',
        'Polynomial + Age²',
        'Interaction Age x ST_dep',
        'Clean Backward',
    ],
    'Train Adj R²': [
        round(model.rsquared_adj,          5),
        round(model_forward.rsquared_adj,   5),
        round(model_backward.rsquared_adj,  5),
        round(model_poly.rsquared_adj,      5),
        round(model_interact.rsquared_adj,  5),
        round(model_clean.rsquared_adj,     5),
    ],
    'Test R²': [
        perf_init['R2'],
        perf_forward['R2'],
        perf_backward['R2'],
        perf_poly['R2'],
        perf_int['R2'],
        perf_clean['R2'],
    ],
    'Test RMSE': [
        perf_init['RMSE'],
        perf_forward['RMSE'],
        perf_backward['RMSE'],
        perf_poly['RMSE'],
        perf_int['RMSE'],
        perf_clean['RMSE'],
    ],
})
print(all_models.to_string(index=False))

best_final = all_models.loc[all_models['Test R²'].idxmax()]
print(f"Best model: {best_final['Model']}")
print(f"Test R²   = {best_final['Test R²']:.5f}")
print(f"Test RMSE = {best_final['Test RMSE']:.4f}")