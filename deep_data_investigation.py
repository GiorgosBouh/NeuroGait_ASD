#!/usr/bin/env python3
"""
Βαθύτερη διερεύνηση για την πηγή του data leakage
ΣΤΟΧΟΣ: Εντοπισμός της ακριβούς αιτίας των υψηλών AUC scores
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def deep_data_investigation():
    """Βαθύτερη διερεύνηση δεδομένων"""
    
    print("🕵️ ΒΑΘΥΤΕΡΗ ΔΙΕΡΕΥΝΗΣΗ ΔΕΔΟΜΕΝΩΝ")
    print("="*80)
    
    # Load data
    try:
        df = pd.read_csv('Final dataset.csv', sep=';', decimal=',', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv('Final dataset.csv', sep=';', decimal=',', encoding='latin-1')
    
    # Convert to numeric
    numeric_cols = [col for col in df.columns if col != 'class']
    for col in numeric_cols:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    # Create participant mapping
    df['participant_id'] = df.index // 8
    df['diagnosis'] = df['class'].map({'A': 1, 'T': 0})
    
    # Select features
    features = [
        'mean HESHL', 'mean SPELR', 'mean SHWRL', 'mean SHWRR',
        'mean ELHAL', 'mean THHAR', 'mean SPKNL', 'mean SPKNR',
        'mean HIANR', 'GaCT', 'StaT', 'SwiT',
        'mean-x-Midspain', 'mean-y-Midspain', 'mean-z-Midspain',
        'mean-x-SpineBase', 'mean-y-SpineBase', 'mean-z-SpineBase',
        'Velocity'
    ]
    
    available_features = [f for f in features if f in df.columns]
    df_clean = df[available_features + ['participant_id', 'diagnosis', 'class']].dropna()
    
    print(f"📊 Dataset: {len(df_clean)} samples, {len(available_features)} features")
    print(f"📊 Participants: {df_clean['participant_id'].nunique()}")
    
    # INVESTIGATION 1: Detailed duplicate analysis
    print("\n🔍 INVESTIGATION 1: Λεπτομερής ανάλυση duplicates")
    
    # Find exact duplicates
    duplicate_mask = df_clean.duplicated(subset=available_features, keep=False)
    duplicates = df_clean[duplicate_mask].sort_values(['participant_id'] + available_features)
    
    print(f"   📊 Total duplicate samples: {duplicate_mask.sum()}")
    
    if len(duplicates) > 0:
        print("   🔍 Analyzing duplicate patterns...")
        
        # Group duplicates
        duplicate_groups = duplicates.groupby(available_features).agg({
            'participant_id': list,
            'diagnosis': list,
            'class': list
        }).reset_index()
        
        print(f"   📊 Unique duplicate groups: {len(duplicate_groups)}")
        
        for i, row in duplicate_groups.head(5).iterrows():
            participants = row['participant_id']
            diagnoses = row['diagnosis']
            classes = row['class']
            
            print(f"   Group {i+1}: Participants {participants}")
            print(f"           Diagnoses: {diagnoses}")
            print(f"           Classes: {classes}")
            
            # Check if duplicates cross diagnosis boundaries
            if len(set(diagnoses)) > 1:
                print(f"           🚨 CRITICAL: Duplicate crosses diagnosis boundaries!")
    
    # INVESTIGATION 2: Statistical separation analysis
    print("\n🔍 INVESTIGATION 2: Στατιστική ανάλυση διαχωρισμού")
    
    asd_data = df_clean[df_clean['diagnosis'] == 1][available_features]
    typical_data = df_clean[df_clean['diagnosis'] == 0][available_features]
    
    print("   📊 Statistical tests for each feature:")
    significant_features = []
    
    for feature in available_features:
        asd_values = asd_data[feature].dropna()
        typical_values = typical_data[feature].dropna()
        
        # Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(asd_values, typical_values, equal_var=False)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(asd_values) - 1) * asd_values.var() + 
                             (len(typical_values) - 1) * typical_values.var()) / 
                            (len(asd_values) + len(typical_values) - 2))
        
        if pooled_std > 0:
            cohens_d = (asd_values.mean() - typical_values.mean()) / pooled_std
        else:
            cohens_d = 0
        
        # Check overlap
        asd_min, asd_max = asd_values.min(), asd_values.max()
        typ_min, typ_max = typical_values.min(), typical_values.max()
        
        overlap = max(0, min(asd_max, typ_max) - max(asd_min, typ_min))
        total_range = max(asd_max, typ_max) - min(asd_min, typ_min)
        overlap_pct = (overlap / total_range * 100) if total_range > 0 else 0
        
        if p_value < 0.001 and abs(cohens_d) > 2.0:
            significant_features.append(feature)
            print(f"   🚨 {feature}:")
            print(f"      p-value: {p_value:.2e}, Cohen's d: {cohens_d:.3f}")
            print(f"      ASD: {asd_values.mean():.3f}±{asd_values.std():.3f}")
            print(f"      Typical: {typical_values.mean():.3f}±{typical_values.std():.3f}")
            print(f"      Overlap: {overlap_pct:.1f}%")
    
    print(f"\n   📊 Features with very strong separation: {len(significant_features)}")
    if len(significant_features) > 5:
        print("   🚨 TOO MANY strongly separating features - suspicious!")
    
    # INVESTIGATION 3: Correlation analysis between features and diagnosis
    print("\n🔍 INVESTIGATION 3: Ανάλυση συσχέτισης με διάγνωση")
    
    correlations = []
    for feature in available_features:
        corr = df_clean[feature].corr(df_clean['diagnosis'])
        correlations.append((feature, abs(corr)))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print("   📊 Top correlations with diagnosis:")
    for i, (feature, corr) in enumerate(correlations[:5]):
        print(f"   {i+1}. {feature}: {corr:.3f}")
        if corr > 0.8:
            print(f"      🚨 VERY HIGH CORRELATION!")
    
    # INVESTIGATION 4: Realistic baseline with noise
    print("\n🔍 INVESTIGATION 4: Realistic baseline με noise")
    
    # Create participant-level split
    participant_info = df_clean.groupby('participant_id')['diagnosis'].first().reset_index()
    train_pids, test_pids = train_test_split(
        participant_info['participant_id'].values,
        test_size=0.2,
        stratify=participant_info['diagnosis'].values,
        random_state=42
    )
    
    train_mask = df_clean['participant_id'].isin(train_pids)
    test_mask = df_clean['participant_id'].isin(test_pids)
    
    X_train = df_clean.loc[train_mask, available_features]
    X_test = df_clean.loc[test_mask, available_features]
    y_train = df_clean.loc[train_mask, 'diagnosis']
    y_test = df_clean.loc[test_mask, 'diagnosis']
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different amounts of noise
    print("   📊 Testing with different noise levels:")
    
    noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0]
    
    for noise_level in noise_levels:
        # Add noise to features
        if noise_level > 0:
            noise_train = np.random.normal(0, noise_level, X_train_scaled.shape)
            noise_test = np.random.normal(0, noise_level, X_test_scaled.shape)
            X_train_noisy = X_train_scaled + noise_train
            X_test_noisy = X_test_scaled + noise_test
        else:
            X_train_noisy = X_train_scaled
            X_test_noisy = X_test_scaled
        
        # Train model
        rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=3)
        rf.fit(X_train_noisy, y_train)
        pred_proba = rf.predict_proba(X_test_noisy)[:, 1]
        auc = roc_auc_score(y_test, pred_proba)
        
        print(f"   Noise level {noise_level:.1f}: AUC = {auc:.3f}")
        
        if auc < 0.85:
            print(f"      ✅ Realistic AUC achieved with noise level {noise_level}")
            break
    
    # INVESTIGATION 5: Feature subset analysis
    print("\n🔍 INVESTIGATION 5: Ανάλυση υποσυνόλων features")
    
    # Test with only clinical features (non-movement)
    clinical_features = ['GaCT', 'StaT', 'SwiT']
    if all(f in available_features for f in clinical_features):
        X_train_clinical = X_train[clinical_features]
        X_test_clinical = X_test[clinical_features]
        
        scaler_clinical = StandardScaler()
        X_train_clinical_scaled = scaler_clinical.fit_transform(X_train_clinical)
        X_test_clinical_scaled = scaler_clinical.transform(X_test_clinical)
        
        rf_clinical = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_clinical.fit(X_train_clinical_scaled, y_train)
        pred_clinical = rf_clinical.predict_proba(X_test_clinical_scaled)[:, 1]
        auc_clinical = roc_auc_score(y_test, pred_clinical)
        
        print(f"   📊 Clinical features only AUC: {auc_clinical:.3f}")
    
    # Test with only movement features
    movement_features = [f for f in available_features if f not in ['GaCT', 'StaT', 'SwiT', 'Velocity']]
    if len(movement_features) > 0:
        X_train_movement = X_train[movement_features]
        X_test_movement = X_test[movement_features]
        
        scaler_movement = StandardScaler()
        X_train_movement_scaled = scaler_movement.fit_transform(X_train_movement)
        X_test_movement_scaled = scaler_movement.transform(X_test_movement)
        
        rf_movement = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_movement.fit(X_train_movement_scaled, y_train)
        pred_movement = rf_movement.predict_proba(X_test_movement_scaled)[:, 1]
        auc_movement = roc_auc_score(y_test, pred_movement)
        
        print(f"   📊 Movement features only AUC: {auc_movement:.3f}")
    
    # INVESTIGATION 6: Data source analysis
    print("\n🔍 INVESTIGATION 6: Ανάλυση πηγής δεδομένων")
    
    print("   🔍 Checking for systematic patterns...")
    
    # Check if participant IDs correlate with diagnosis
    participant_diag = df_clean.groupby('participant_id')['diagnosis'].first()
    participant_ids = participant_diag.index.values
    diagnoses = participant_diag.values
    
    # Check if low IDs are mostly one class
    first_half = participant_ids < participant_ids.mean()
    first_half_asd_ratio = diagnoses[first_half].mean()
    second_half_asd_ratio = diagnoses[~first_half].mean()
    
    print(f"   📊 First half participants ASD ratio: {first_half_asd_ratio:.3f}")
    print(f"   📊 Second half participants ASD ratio: {second_half_asd_ratio:.3f}")
    
    if abs(first_half_asd_ratio - second_half_asd_ratio) > 0.2:
        print("   🚨 SYSTEMATIC BIAS: Participant IDs correlate with diagnosis!")
    
    # Final assessment
    print("\n💡 ΣΥΜΠΕΡΑΣΜΑΤΑ:")
    
    if len(significant_features) > 10:
        print("   🚨 ΚΡΙΣΙΜΟ: Πάρα πολλά features με εξαιρετικό διαχωρισμό")
        print("   📋 Πιθανά αίτια:")
        print("      • Τεχνητά δεδομένα ή προσομοίωση")
        print("      • Systematic bias στη συλλογή")
        print("      • Pre-processing που δημιουργεί artifacts")
    
    max_correlation = max([corr[1] for corr in correlations])
    if max_correlation > 0.9:
        print(f"   🚨 ΚΡΙΣΙΜΟ: Εξαιρετικά υψηλή συσχέτιση ({max_correlation:.3f})")
        print("   📋 Αυτό δεν είναι realistic για medical data")
    
    if duplicate_mask.sum() > len(df_clean) * 0.1:
        print(f"   ⚠️ ΠΡΟΣΟΧΗ: Πολλά duplicates ({duplicate_mask.sum()}/{len(df_clean)})")
        print("   📋 Μπορεί να οδηγεί σε overfitting")
    
    print("\n🔬 ΠΡΟΤΑΣΕΙΣ ΛΥΣΗΣ:")
    print("   1. Χρησιμοποίησε μόνο ένα subset των πιο realistic features")
    print("   2. Προσθέστε τεχνητό noise για realistic performance")
    print("   3. Χρησιμοποίησε πιο conservative ML models")
    print("   4. Εφάρμοσε strict regularization")
    print("   5. Reduce το dataset size για harder classification")
    
    # Return key metrics for further analysis
    return {
        'max_correlation': max_correlation,
        'significant_features': len(significant_features),
        'duplicates': duplicate_mask.sum(),
        'asd_ratio_bias': abs(first_half_asd_ratio - second_half_asd_ratio)
    }

if __name__ == "__main__":
    results = deep_data_investigation()