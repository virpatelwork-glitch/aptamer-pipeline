#!/usr/bin/env python3
"""
L4 Aptamer Aβ42 Binding Score Regression Analysis
=================================================
This script builds a Random Forest Regressor specifically for L4 aptamers
to predict Aβ42 binding scores and identifies impactful sequence features.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)
sns.set_style("whitegrid")

print("=" * 80)
print("L4 Aptamer Aβ42 Binding Score Regression")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING AND FILTERING
# ============================================================================
print("\n[1] Loading and Filtering Data...")
df = pd.read_csv('/home/ubuntu/upload/aptamer_raw_data.csv')

# Filter for L4 aptamers only
l4_df = df[df['Design_Level'] == 'L4'].copy()
print(f"Total L4 samples found: {len(l4_df)}")

if len(l4_df) == 0:
    print("Error: No L4 samples found in the dataset!")
    exit()

# ============================================================================
# 2. SEQUENCE FEATURE ENGINEERING
# ============================================================================
print("\n[2] Engineering Sequence Features...")

def extract_sequence_features(df):
    # Base composition
    df['A_count'] = df['Sequence'].str.count('A')
    df['C_count'] = df['Sequence'].str.count('C')
    df['G_count'] = df['Sequence'].str.count('G')
    df['U_count'] = df['Sequence'].str.count('U')
    
    # Base percentages
    seq_len = df['Sequence'].str.len()
    df['A_pct'] = df['A_count'] / seq_len
    df['C_pct'] = df['C_count'] / seq_len
    df['G_pct'] = df['G_count'] / seq_len
    df['U_pct'] = df['U_count'] / seq_len
    
    # Dinucleotide frequencies (common ones)
    dinucleotides = ['GG', 'CC', 'AA', 'UU', 'GC', 'CG', 'AU', 'UA']
    for di in dinucleotides:
        df[f'{di}_count'] = df['Sequence'].str.count(di)
        
    return df

l4_df = extract_sequence_features(l4_df)

# Features to use: Sequence features + Structural features
features = [
    'A_pct', 'C_pct', 'G_pct', 'U_pct',
    'GG_count', 'CC_count', 'AA_count', 'UU_count', 
    'GC_count', 'CG_count', 'AU_count', 'UA_count',
    'Length_nt', 'GC_Content_%', 'Stem_Length_bp', 'Loop_Size_nt',
    'Folding_DeltaG_kcal_mol'
]

# Target variable: Vina_Aβ42_Score (Binding Score)
# Note: Lower scores (more negative) usually mean better binding.
target = 'Vina_Aβ42_Score'

X = l4_df[features]
y = l4_df[target]

print(f"Features used: {features}")
print(f"Target variable: {target}")

# ============================================================================
# 3. MODEL TRAINING
# ============================================================================
print("\n[3] Training Random Forest Regressor...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid Search for best parameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2', None]
}

rf_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf_reg, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# ============================================================================
# 4. EVALUATION
# ============================================================================
print("\n[4] Evaluating Model...")

y_pred = best_rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# ============================================================================
# 5. FEATURE IMPORTANCE
# ============================================================================
print("\n[5] Analyzing Feature Importance...")

importances = best_rf.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

print("\nTop 10 Most Impactful Features for Binding Score:")
print(feature_imp_df.head(10))

# ============================================================================
# 6. VISUALIZATION
# ============================================================================
print("\n[6] Generating Visualizations...")

plt.figure(figsize=(20, 12))

# Subplot 1: Feature Importance
plt.subplot(2, 2, 1)
sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(15), palette='magma')
plt.title('Top 15 Impactful Features (L4 Aβ42 Binding)', fontsize=14)

# Subplot 2: Actual vs Predicted
plt.subplot(2, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.7, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Vina Aβ42 Score')
plt.ylabel('Predicted Vina Aβ42 Score')
plt.title(f'Actual vs Predicted (R² = {r2:.3f})', fontsize=14)

# Subplot 3: Residuals
plt.subplot(2, 2, 3)
residuals = y_test - y_pred
sns.histplot(residuals, kde=True, color='purple')
plt.title('Residuals Distribution', fontsize=14)
plt.xlabel('Residual (Actual - Predicted)')

# Subplot 4: Top Feature Correlation
top_feat = feature_imp_df.iloc[0]['Feature']
plt.subplot(2, 2, 4)
sns.regplot(x=X[top_feat], y=y, color='coral', scatter_kws={'alpha':0.5})
plt.title(f'Impact of {top_feat} on Binding Score', fontsize=14)
plt.xlabel(top_feat)
plt.ylabel('Vina Aβ42 Score')

plt.tight_layout()
plt.savefig('/home/ubuntu/l4_regression_results.png', dpi=300)
print("Visualization saved to 'l4_regression_results.png'")

# Save data
feature_imp_df.to_csv('/home/ubuntu/l4_feature_impact.csv', index=False)
l4_df.to_csv('/home/ubuntu/l4_processed_data.csv', index=False)

print("\nAnalysis Complete!")
