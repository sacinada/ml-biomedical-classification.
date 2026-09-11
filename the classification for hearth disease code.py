# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:50:57 2026

@author: sacin
"""

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, roc_auc_score, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')


# Load Data

df = pd.read_csv(r'C:\Users\sacin\Downloads\Heart_Disease_Prediction.csv')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())


# Data Understanding

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

print("\nTarget Distribution:")
print(df['Heart Disease'].value_counts())
print(df['Heart Disease'].value_counts(normalize=True).round(3))


# EDA

continuous_features  = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
categorical_features = ['Sex', 'Chest pain type', 'FBS over 120',
                         'EKG results', 'Exercise angina',
                         'Slope of ST', 'Number of vessels fluro', 'Thallium']

plt.figure(figsize=(6, 4))
df['Heart Disease'].value_counts().plot(kind='bar',
                                         color=['steelblue', 'salmon'],
                                         edgecolor='black')
plt.title('Class Distribution Heart Disease')
plt.xlabel('Heart Disease')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('01_class_distribution.png', dpi=150)
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, col in enumerate(continuous_features):
    for label, color in zip(['Presence', 'Absence'], ['salmon', 'steelblue']):
        axes[i].hist(df[df['Heart Disease'] == label][col],
                     alpha=0.6, label=label, color=color,
                     bins=15, edgecolor='black')
    axes[i].set_title(f'{col} by Heart Disease Status')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].legend()
axes[-1].set_visible(False)
plt.suptitle('Histograms of Continuous Features by Heart Disease Status', fontsize=13)
plt.tight_layout()
plt.savefig('02_histograms_continuous.png', dpi=150)
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, col in enumerate(continuous_features):
    presence = df[df['Heart Disease'] == 'Presence'][col]
    absence  = df[df['Heart Disease'] == 'Absence'][col]
    axes[i].boxplot([absence, presence],
                    labels=['Absence', 'Presence'],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue'),
                    medianprops=dict(color='red', linewidth=2))
    axes[i].set_title(f'{col} by Heart Disease Status')
    axes[i].set_ylabel(col)
axes[-1].set_visible(False)
plt.suptitle('Boxplots of Continuous Features by Heart Disease Status', fontsize=13)
plt.tight_layout()
plt.savefig('03_boxplots.png', dpi=150)
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(categorical_features):
    ct = pd.crosstab(df[col], df['Heart Disease'], normalize='index')
    ct.plot(kind='bar', ax=axes[i],
            color=['steelblue', 'salmon'],
            edgecolor='black', legend=(i == 0))
    axes[i].set_title(f'{col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Proportion')
    axes[i].tick_params(axis='x', rotation=0)
plt.suptitle('Categorical Features vs Heart Disease (Proportion)', fontsize=13)
plt.tight_layout()
plt.savefig('04_categorical_features.png', dpi=150)
plt.show()

plt.figure(figsize=(12, 8))
df_corr = df.copy()
df_corr['Heart Disease'] = (df_corr['Heart Disease'] == 'Presence').astype(int)
corr_matrix = df_corr.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=0.5)
plt.title('Correlation Heatmap All Features')
plt.tight_layout()
plt.savefig('05_correlation_heatmap.png', dpi=150)
plt.show()


# Outlier Detection

print("\nOutlier Detection (IQR Method):")
for col in continuous_features:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower    = Q1 - 1.5 * IQR
    upper    = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers  "
          f"(lower bound={lower:.1f}, upper bound={upper:.1f})")


# Preprocessing

df['Target'] = (df['Heart Disease'] == 'Presence').astype(int)

cat_cols  = ['Chest pain type', 'EKG results', 'Slope of ST',
             'Number of vessels fluro', 'Thallium']
cont_cols = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']

df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

drop_cols    = ['Heart Disease', 'Target']
feature_cols = [c for c in df_encoded.columns if c not in drop_cols]
X = df_encoded[feature_cols]
y = df_encoded['Target']

print("\nFinal feature matrix shape:", X.shape)
print("Features:", feature_cols)


# Baseline Logistic Regression (Full Dataset)

def make_formula(target, features):
    terms = []
    for f in features:
        if ' ' in f or f[0].isdigit():
            terms.append(f'Q("{f}")')
        else:
            terms.append(f)
    return target + ' ~ ' + ' + '.join(terms)

formula = make_formula('Target', feature_cols)
logit_full = smf.logit(formula, data=df_encoded).fit()
print(logit_full.summary())

print("\nSignificant Predictors (p < 0.05):")
pvals = logit_full.pvalues.drop('Intercept')
sig   = pvals[pvals < 0.05].sort_values()
print(sig.round(4))

print("\nCoefficient Interpretation (Log-Odds):")
coefs = logit_full.params.drop('Intercept')
for feat, coef in coefs[sig.index].items():
    direction = "increases" if coef > 0 else "decreases"
    print(f"  {feat}: A 1-unit increase {direction} the log-odds "
          f"of heart disease by {abs(coef):.4f}  (OR = {np.exp(coef):.4f})")


# Train / Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=123,
    stratify=y
)

print(f"Training set : {X_train.shape[0]} samples ({100*X_train.shape[0]/len(X):.0f}%)")
print(f"Test set     : {X_test.shape[0]} samples ({100*X_test.shape[0]/len(X):.0f}%)")
print(f"\nClass balance in train: {y_train.value_counts().to_dict()}")
print(f"Class balance in test : {y_test.value_counts().to_dict()}")

scaler         = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()

X_train_scaled[cont_cols] = scaler.fit_transform(X_train[cont_cols])
X_test_scaled[cont_cols]  = scaler.transform(X_test[cont_cols])


# Evaluate Model

def evaluate_model(name, y_test, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")

    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()

    print(f"\nConfusion Matrix:")
    print(f"                  Predicted Absence  Predicted Presence")
    print(f"  Actual Absence       {TN:>4}                {FP:>4}")
    print(f"  Actual Presence      {FN:>4}                {TP:>4}")

    accuracy    = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    auc         = roc_auc_score(y_test, y_prob)

    print(f"\n  Accuracy    : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  Sensitivity : {sensitivity:.4f}  ({sensitivity*100:.1f}%)")
    print(f"  Specificity : {specificity:.4f}  ({specificity*100:.1f}%)")
    print(f"  AUC         : {auc:.4f}")

    return {
        'Model'      : name,
        'Accuracy'   : round(accuracy, 4),
        'Sensitivity': round(sensitivity, 4),
        'Specificity': round(specificity, 4),
        'AUC'        : round(auc, 4),
        'y_pred'     : y_pred,
        'y_prob'     : y_prob
    }

results = []


# Logistic Regression

log_reg = LogisticRegression(random_state=123, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
log_prob = log_reg.predict_proba(X_test_scaled)[:, 1]
res_log = evaluate_model('Logistic Regression', y_test, log_prob)
results.append(res_log)

print("\n  Logistic Regression Coefficients (sorted by importance):")
coef_df = pd.DataFrame({
    'Feature'    : feature_cols,
    'Coefficient': log_reg.coef_[0],
    'Odds Ratio' : np.exp(log_reg.coef_[0])
}).sort_values('Coefficient', key=abs, ascending=False)
print(coef_df.to_string(index=False))


# LDA

lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train)
lda_prob = lda.predict_proba(X_test_scaled)[:, 1]
res_lda = evaluate_model('LDA', y_test, lda_prob)
results.append(res_lda)


# QDA

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_scaled, y_train)
qda_prob = qda.predict_proba(X_test_scaled)[:, 1]
res_qda = evaluate_model('QDA', y_test, qda_prob)
results.append(res_qda)


# KNN

print("\nKNN: 10-Fold Cross-Validation to Select Optimal K (1-100)...")
k_values  = list(range(1, 101))
cv_scores = []

for k in k_values:
    knn    = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k        = k_values[np.argmax(cv_scores)]
best_cv_score = max(cv_scores)
print(f"Best K : {best_k}  (10-fold CV Accuracy = {best_cv_score:.4f})")

plt.figure(figsize=(10, 5))
plt.plot(k_values, cv_scores, color='steelblue', linewidth=1.5)
plt.axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Best K={best_k}')
plt.scatter([best_k], [best_cv_score], color='red', zorder=5, s=80)
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('10-Fold Cross-Validated Accuracy')
plt.title('KNN Cross-Validated Accuracy vs K')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('06_knn_cv_accuracy.png', dpi=150)
plt.show()

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)
knn_prob = knn_best.predict_proba(X_test_scaled)[:, 1]
res_knn = evaluate_model(f'KNN (K={best_k})', y_test, knn_prob)
results.append(res_knn)


# Naive Bayes

nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
nb_prob = nb.predict_proba(X_test_scaled)[:, 1]
res_nb = evaluate_model('Naive Bayes', y_test, nb_prob)
results.append(res_nb)


# Model Comparison

summary_df = pd.DataFrame([{
    'Model'      : r['Model'],
    'Accuracy'   : r['Accuracy'],
    'Sensitivity': r['Sensitivity'],
    'Specificity': r['Specificity'],
    'AUC'        : r['AUC']
} for r in results])

print("\nFinal Model Comparison:")
print(summary_df.to_string(index=False))

all_names = [r['Model'] for r in results]

fig, axes = plt.subplots(1, 5, figsize=(24, 4))
for ax, r in zip(axes, results):
    cm   = confusion_matrix(y_test, r['y_pred'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['Absence', 'Presence'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(r['Model'], fontsize=9)
plt.suptitle('Confusion Matrices All Models (Test Set)', fontsize=13)
plt.tight_layout()
plt.savefig('07_confusion_matrices.png', dpi=150)
plt.show()

plt.figure(figsize=(8, 6))
colors = ['blue', 'green', 'orange', 'red', 'purple']
for r, color in zip(results, colors):
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    plt.plot(fpr, tpr, label=f"{r['Model']} (AUC={r['AUC']:.3f})", color=color, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC=0.5)')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curves All Models')
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('08_roc_curves.png', dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
auc_values = [r['AUC'] for r in results]
bars = plt.bar(all_names, auc_values, color=colors, edgecolor='black', alpha=0.8)
plt.ylabel('AUC Score')
plt.title('AUC Comparison All Models')
plt.ylim(0.5, 1.0)
plt.xticks(rotation=15, ha='right')
for bar, val in zip(bars, auc_values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('09_auc_comparison.png', dpi=150)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(all_names, [r['Sensitivity'] for r in results], color=colors, edgecolor='black', alpha=0.8)
axes[0].set_title('Sensitivity by Model')
axes[0].set_ylabel('Sensitivity')
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis='x', rotation=15)
for i, r in enumerate(results):
    axes[0].text(i, r['Sensitivity'] + 0.01, f"{r['Sensitivity']:.3f}", ha='center', fontsize=9)

axes[1].bar(all_names, [r['Specificity'] for r in results], color=colors, edgecolor='black', alpha=0.8)
axes[1].set_title('Specificity by Model')
axes[1].set_ylabel('Specificity')
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis='x', rotation=15)
for i, r in enumerate(results):
    axes[1].text(i, r['Specificity'] + 0.01, f"{r['Specificity']:.3f}", ha='center', fontsize=9)

plt.suptitle('Sensitivity and Specificity All Models', fontsize=13)
plt.tight_layout()
plt.savefig('10_sensitivity_specificity.png', dpi=150)
plt.show()

print("\nDone. Final summary:")
print(summary_df.to_string(index=False))