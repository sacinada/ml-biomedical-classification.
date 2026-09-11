# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:58:06 2026

@author: sacin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r'C:\Users\sacin\Downloads\CASP.csv')

print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("Missing values:", df.isnull().sum().sum())
print(df.head())

print(f"F3 mean     : {df['F3'].mean():.4f}")
print(f"F3 median   : {df['F3'].median():.4f}")
print(f"F3 skewness : {stats.skew(df['F3']):.4f}")
print(f"F3 min      : {df['F3'].min():.4f}")
print(f"F3 max      : {df['F3'].max():.4f}")
print(f"F3 zeros    : {(df['F3']==0).sum()}")


# Visualisation of Target Variable

fig, axes = plt.subplots(1, 3, figsize=(17, 4))

axes[0].hist(df['F3'], bins=60, color='steelblue', edgecolor='black', alpha=0.8)
axes[0].axvline(df['F3'].mean(),   color='red',    linestyle='--', linewidth=2, label=f"Mean={df['F3'].mean():.4f}")
axes[0].axvline(df['F3'].median(), color='orange', linestyle='--', linewidth=2, label=f"Median={df['F3'].median():.4f}")
axes[0].set_title('Distribution of F3', fontweight='bold')
axes[0].set_xlabel('F3')
axes[0].set_ylabel('Number of observations')
axes[0].legend()

stats.probplot(df['F3'], dist='norm', plot=axes[1])
axes[1].set_title('Q-Q Plot of F3', fontweight='bold')

axes[2].boxplot(df['F3'], patch_artist=True, boxprops=dict(facecolor='steelblue', alpha=0.6))
axes[2].set_title('F3 outlier detection', fontweight='bold')
q1  = df['F3'].quantile(0.25)
q3v = df['F3'].quantile(0.75)
n_out = ((df['F3'] < q1-1.5*(q3v-q1)) | (df['F3'] > q3v+1.5*(q3v-q1))).sum()
axes[2].set_xlabel(f'IQR outliers: {n_out:,}')

plt.suptitle('Target Variable F3: Distribution Check', fontweight='bold')
plt.tight_layout()
plt.show()


# Log Transform Check

print(f"F3 skewness = {stats.skew(df['F3']):.4f}")


# Correlation Matrix

data_cor_cols = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','RMSD']

plt.figure(figsize=(10, 8))
sns.heatmap(df[data_cor_cols].corr(), annot=True, fmt='.2f',
            cmap='coolwarm', square=True, linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix CASP', fontweight='bold')
plt.tight_layout()
plt.show()


# Train / Test Split

train_data, test_data = train_test_split(df, test_size=0.30, random_state=123)
train_data = train_data.reset_index(drop=True)
test_data  = test_data.reset_index(drop=True)

print(f"Training set : {len(train_data):,} proteins (70%)")
print(f"Test set     : {len(test_data):,} proteins (30%)")


# Exhaustive Best-Subset Selection

candidates = ['F2', 'F4', 'F6', 'F7', 'F8', 'F9', 'RMSD']

best_results = []

for k in range(1, len(candidates)+1):
    best_adjr2 = -np.inf
    best_sub   = None
    for subset in combinations(candidates, k):
        formula   = 'F3 ~ ' + ' + '.join(subset)
        model_sub = smf.ols(formula, data=train_data).fit()
        if model_sub.rsquared_adj > best_adjr2:
            best_adjr2 = model_sub.rsquared_adj
            best_sub   = list(subset)
    best_results.append({'n': k, 'vars': best_sub, 'adj_r2': round(best_adjr2, 5)})
    print(f"  {k} var(s): Adj R²={best_adjr2:.5f}  {best_sub}")

best_df  = pd.DataFrame(best_results)
best_row = best_df.loc[best_df['adj_r2'].idxmax()]
best_vars = best_row['vars']
print(f"Best subset: {best_vars}")
print(f"Adj R²     : {best_row['adj_r2']:.5f}")


# Best-Subset Plot

plt.figure(figsize=(9, 4))
plt.plot(best_df['n'], best_df['adj_r2'], marker='o', color='steelblue', linewidth=2, markersize=7)
for _, row in best_df.iterrows():
    plt.annotate(f"{row['adj_r2']:.4f}", (row['n'], row['adj_r2']),
                 textcoords='offset points', xytext=(4, 6), fontsize=8)
plt.xlabel('Number of variables')
plt.ylabel('Adjusted R²')
plt.title('Exhaustive Best-Subset Adj R² by Model Size', fontweight='bold')
plt.xticks(range(1, len(candidates)+1))
plt.tight_layout()
plt.show()


# Chosen Model

formula_main = 'F3 ~ F2 + F4 + F6 + F7 + F8 + F9'
model        = smf.ols(formula_main, data=train_data).fit()
print(model.summary())


# Scaled Model

sc_vars  = ['F2', 'F4', 'F6', 'F7', 'F8', 'F9', 'F3']
sc_train = train_data[sc_vars].copy()
sc_train = (sc_train - sc_train.mean()) / sc_train.std()

model_scaled = smf.ols(formula_main, data=sc_train).fit()
print(model_scaled.summary())

predictions = model.predict(test_data)
actual      = test_data['F3']
mn = min(actual.min(), predictions.min()) - 0.005
mx = max(actual.max(), predictions.max()) + 0.005

plt.figure(figsize=(7, 5))
plt.scatter(actual, predictions, alpha=0.1, color='steelblue', s=5)
plt.plot([mn,mx],[mn,mx], color='red', linewidth=2, label='Perfect prediction (y=x)')
plt.xlabel('Actual F3')
plt.ylabel('Predicted F3')
plt.title('Predictions vs Actual Exhaustive Model', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()


# Forward Selection

current_fwd = []
remaining   = candidates.copy()
print("Forward selection steps:")

for _ in range(len(candidates)):
    best_pval = np.inf
    best_var  = None
    for v in remaining:
        test_formula = 'F3 ~ ' + ' + '.join(current_fwd + [v])
        test_model   = smf.ols(test_formula, data=train_data).fit()
        p_val = test_model.pvalues[v]
        if p_val < best_pval:
            best_pval = p_val
            best_var  = v
    if best_var and best_pval < 0.05:
        current_fwd.append(best_var)
        remaining.remove(best_var)
        m_t = smf.ols('F3 ~ ' + ' + '.join(current_fwd), data=train_data).fit()
        print(f"  + {best_var:<8}  p={best_pval:.4e}  Adj R²={m_t.rsquared_adj:.5f}")
    else:
        print("  STOP no remaining variable has p < 0.05")
        break

formula_fwd   = 'F3 ~ ' + ' + '.join(current_fwd)
model_forward = smf.ols(formula_fwd, data=train_data).fit()
print(f"Forward selected vars: {current_fwd}")

predictions_forward = model_forward.predict(test_data)

plt.figure(figsize=(7, 5))
plt.scatter(actual, predictions_forward, alpha=0.1, color='seagreen', s=5)
plt.plot([mn,mx],[mn,mx], color='red', linewidth=2, label='y=x')
plt.xlabel('Actual F3')
plt.ylabel('Predicted F3')
plt.title('Forward Selection: Predictions vs Actual', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()


# Backward Selection

current_bwd = candidates.copy()
print("Backward selection steps:")

for _ in range(len(candidates)):
    m_t   = smf.ols('F3 ~ ' + ' + '.join(current_bwd), data=train_data).fit()
    pvals = m_t.pvalues[current_bwd]
    worst = pvals.idxmax()
    wp    = pvals.max()
    if wp > 0.05:
        current_bwd.remove(worst)
        m_n = smf.ols('F3 ~ ' + ' + '.join(current_bwd), data=train_data).fit()
        print(f"  - {worst:<8}  p={wp:.4f}  Adj R²={m_n.rsquared_adj:.5f}")
    else:
        print("  STOP all remaining variables have p < 0.05")
        break

print(f"Backward selected vars: {current_bwd}")

formula_bwd    = 'F3 ~ ' + ' + '.join(current_bwd)
model_backward = smf.ols(formula_bwd, data=train_data).fit()
predictions_backward = model_backward.predict(test_data)

plt.figure(figsize=(7, 5))
plt.scatter(actual, predictions_backward, alpha=0.1, color='coral', s=5)
plt.plot([mn,mx],[mn,mx], color='red', linewidth=2, label='y=x')
plt.xlabel('Actual F3')
plt.ylabel('Predicted F3')
plt.title('Backward Selection: Predictions vs Actual', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()


# Performance Table

def evaluate_model(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2   = r2_score(actual, predicted)
    return {'RMSE': round(rmse, 6), 'R2': round(r2, 6)}

pred_init     = model.predict(test_data)
pred_forward  = model_forward.predict(test_data)

formula_bwd    = 'F3 ~ ' + ' + '.join(current_bwd)
model_backward = smf.ols(formula_bwd, data=train_data).fit()
pred_backward  = model_backward.predict(test_data)

actual = test_data['F3']

perf_init     = evaluate_model(actual, pred_init)
perf_forward  = evaluate_model(actual, pred_forward)
perf_backward = evaluate_model(actual, pred_backward)

perf_table = pd.DataFrame({
    'Model': ['Exhaustive', 'Forward', 'Backward'],
    'RMSE' : [perf_init['RMSE'], perf_forward['RMSE'], perf_backward['RMSE']],
    'R2'   : [perf_init['R2'],   perf_forward['R2'],   perf_backward['R2']],
})
print(perf_table.to_string(index=False))


# Residual Diagnostics

fitted_bwd  = model_backward.fittedvalues
resid_bwd   = model_backward.resid
mean_resid  = resid_bwd.mean()
med_resid   = resid_bwd.median()

print(f"Mean of residuals   : {mean_resid:.2e}")
print(f"Median of residuals : {med_resid:.6f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

lowess_bwd = sm.nonparametric.lowess(resid_bwd, fitted_bwd, frac=0.05)
axes[0].scatter(fitted_bwd, resid_bwd, alpha=0.05, color='steelblue', s=3)
axes[0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0].plot(lowess_bwd[:,0], lowess_bwd[:,1], color='orange', linewidth=2.5, label='LOWESS trend')
axes[0].set_xlim(fitted_bwd.quantile(0.01), fitted_bwd.quantile(0.99))
axes[0].set_xlabel('Fitted Values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Fitted Values', fontweight='bold')
axes[0].legend()

axes[1].hist(resid_bwd, bins=80, color='steelblue', edgecolor='black', alpha=0.8, density=True)
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
axes[1].axvline(mean_resid, color='orange', linestyle='--', linewidth=2, label=f'Mean={mean_resid:.4f}')
xn = np.linspace(resid_bwd.min(), resid_bwd.max(), 300)
axes[1].plot(xn, stats.norm.pdf(xn, mean_resid, resid_bwd.std()), color='black', linewidth=1.5, label='Normal curve')
axes[1].set_xlabel('Residuals')
axes[1].set_ylabel('Density')
axes[1].set_title('Histogram of Residuals', fontweight='bold')
axes[1].legend(fontsize=8)

plt.suptitle('Residual Diagnostics', fontweight='bold')
plt.tight_layout()
plt.show()


# Polynomial Regression

train_data['F2_sq'] = train_data['F2'] ** 2
test_data['F2_sq']  = test_data['F2']  ** 2

formula_poly = 'F3 ~ F2 + F4 + F6 + F7 + F8 + F9 + F2_sq'
model_poly   = smf.ols(formula_poly, data=train_data).fit()
print(model_poly.summary())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

f2_seq      = np.linspace(train_data['F2'].min(), train_data['F2'].max(), 300)
other_means = {v: train_data[v].mean() for v in ['F4','F6','F7','F8','F9']}

lin_line  = model_backward.params['Intercept'] + model_backward.params['F2'] * f2_seq
for v, m in other_means.items():
    lin_line += model_backward.params[v] * m

poly_line = model_poly.params['Intercept'] + model_poly.params['F2'] * f2_seq + model_poly.params['F2_sq'] * f2_seq**2
for v, m in other_means.items():
    poly_line += model_poly.params[v] * m

axes[0].scatter(train_data['F2'], train_data['F3'], alpha=0.04, color='steelblue', s=3, label='Data')
axes[0].plot(f2_seq, lin_line,  color='orange', linewidth=2, linestyle='--', label='Linear')
axes[0].plot(f2_seq, poly_line, color='red',    linewidth=2.5, label='Polynomial (+F2²)')
axes[0].set_xlabel('F2'); axes[0].set_ylabel('F3')
axes[0].set_title('Linear vs Polynomial Fit for F2', fontweight='bold')
axes[0].legend(fontsize=8)

resid_poly  = model_poly.resid
fitted_poly = model_poly.fittedvalues
mask_poly   = (fitted_poly > 0.05) & (fitted_poly < 0.60)
lowess_poly = sm.nonparametric.lowess(resid_poly[mask_poly], fitted_poly[mask_poly], frac=0.1)
axes[1].scatter(fitted_poly, resid_poly, alpha=0.05, color='steelblue', s=3)
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].plot(lowess_poly[:,0], lowess_poly[:,1], color='orange', linewidth=2.5, label='LOWESS trend')
axes[1].set_xlim(0.05, 0.60)
axes[1].set_xlabel('Fitted Values'); axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals vs Fitted Polynomial Model', fontweight='bold')
axes[1].legend()

plt.suptitle('Polynomial Regression: F2 + F2²', fontweight='bold')
plt.tight_layout()
plt.show()


# Interaction Model

formula_int    = 'F3 ~ RMSD * F2 + F4 + F6 + F7 + F8 + F9'
model_interact = smf.ols(formula_int, data=train_data).fit()
print(model_interact.summary())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

f2_terciles = pd.qcut(train_data['F2'], q=3, labels=['Low F2', 'Mid F2', 'High F2'])
colors_t    = {'Low F2': 'steelblue', 'Mid F2': 'seagreen', 'High F2': 'coral'}

for grp, col in colors_t.items():
    mask = f2_terciles == grp
    axes[0].scatter(train_data.loc[mask, 'RMSD'], train_data.loc[mask, 'F3'],
                    alpha=0.05, color=col, s=3, label=grp)
    sub = train_data[mask]
    slope, intercept = stats.linregress(sub['RMSD'], sub['F3'])[:2]
    x_g = np.linspace(sub['RMSD'].min(), sub['RMSD'].max(), 100)
    axes[0].plot(x_g, intercept + slope*x_g, color=col, linewidth=2)

axes[0].set_xlabel('RMSD')
axes[0].set_ylabel('F3')
axes[0].set_title('Interaction: RMSD effect on F3 by F2 group', fontweight='bold')
axes[0].legend(fontsize=8)

fitted_int  = model_interact.fittedvalues
resid_int   = model_interact.resid
mask_int    = (fitted_int > 0.05) & (fitted_int < 0.58)
lowess_int  = sm.nonparametric.lowess(resid_int[mask_int], fitted_int[mask_int], frac=0.1)
axes[1].scatter(fitted_int, resid_int, alpha=0.05, color='steelblue', s=3)
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].plot(lowess_int[:,0], lowess_int[:,1], color='orange', linewidth=2.5, label='LOWESS trend (clipped)')
axes[1].set_xlim(0.05, 0.58)
axes[1].set_xlabel('Fitted Values')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals vs Fitted Interaction Model', fontweight='bold')
axes[1].legend()

plt.suptitle('Interaction Model: RMSD x F2', fontweight='bold')
plt.tight_layout()
plt.show()


# Outlier Detection and Clean Model

influence    = model_backward.get_influence()
std_resid    = influence.resid_studentized_internal
outlier_mask = np.abs(std_resid) > 2
n_outliers   = outlier_mask.sum()

print(f"Training proteins  : {len(train_data):,}")
print(f"Flagged (|e*| > 2) : {n_outliers:,} ({100*n_outliers/len(train_data):.1f}%)")
print(f"Clean training set : {len(train_data)-n_outliers:,} proteins")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

idx_arr = np.arange(len(std_resid))
axes[0].scatter(idx_arr[~outlier_mask], std_resid[~outlier_mask], alpha=0.15, color='steelblue', s=3, label='Normal')
axes[0].scatter(idx_arr[outlier_mask],  std_resid[outlier_mask],  color='red', s=8, zorder=5, label=f'Outliers (n={n_outliers:,})')
axes[0].axhline( 2, color='red', linestyle='--', linewidth=1.5, label='+-2 threshold')
axes[0].axhline(-2, color='red', linestyle='--', linewidth=1.5)
axes[0].set_xlabel('Observation Index')
axes[0].set_ylabel('Studentized Residual')
axes[0].set_title('Studentized Residuals', fontweight='bold')
axes[0].legend(fontsize=8)

train_clean  = train_data[~outlier_mask].copy().reset_index(drop=True)
model_clean  = smf.ols(formula_bwd, data=train_clean).fit()

clean_fitted = model_clean.fittedvalues
clean_resid  = model_clean.resid
q01 = clean_fitted.quantile(0.01)
q99 = clean_fitted.quantile(0.99)
mask_cl   = (clean_fitted > q01) & (clean_fitted < q99)
lowess_cl = sm.nonparametric.lowess(clean_resid[mask_cl], clean_fitted[mask_cl], frac=0.1)
axes[1].scatter(clean_fitted, clean_resid, alpha=0.05, color='seagreen', s=3)
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].plot(lowess_cl[:,0], lowess_cl[:,1], color='orange', linewidth=2.5, label='LOWESS trend')
axes[1].set_xlim(q01, q99)
axes[1].set_xlabel('Fitted Values (central 98%)')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals vs Fitted Clean Model', fontweight='bold')
axes[1].legend()

plt.suptitle('Outlier Detection and Clean Model', fontweight='bold')
plt.tight_layout()
plt.show()


# Final Comparison

pred_clean = model_clean.predict(test_data)
perf_clean = evaluate_model(test_data['F3'], pred_clean)
perf_back  = evaluate_model(test_data['F3'], model_backward.predict(test_data))

print(f"Backward model  Test R²: {perf_back['R2']}   RMSE: {perf_back['RMSE']}")
print(f"Clean model     Test R²: {perf_clean['R2']}   RMSE: {perf_clean['RMSE']}")
print(f"Train R² original : {model_backward.rsquared:.5f}")
print(f"Train R² clean    : {model_clean.rsquared:.5f}")

pred_poly = model_poly.predict(test_data)
pred_int  = model_interact.predict(test_data)

perf_poly = evaluate_model(test_data['F3'], pred_poly)
perf_int  = evaluate_model(test_data['F3'], pred_int)

all_models = pd.DataFrame({
    'Model': [
        'Exhaustive',
        'Forward',
        'Backward',
        'Polynomial + F2²',
        'Interaction RMSD x F2',
        'Clean Backward',
    ],
    'Train Adj R²': [
        round(model.rsquared_adj,         5),
        round(model_forward.rsquared_adj,  5),
        round(model_backward.rsquared_adj, 5),
        round(model_poly.rsquared_adj,     5),
        round(model_interact.rsquared_adj, 5),
        round(model_clean.rsquared_adj,    5),
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