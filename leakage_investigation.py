#!/usr/bin/env python3
"""
Data Leakage Investigation Script
CRITICAL: These AUC scores (>0.99) indicate serious data leakage
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def investigate_data_leakage():
    """Comprehensive data leakage investigation"""
    
    print("🔍 COMPREHENSIVE DATA LEAKAGE INVESTIGATION")
    print("="*60)
    
    # Load data
    try:
        df = pd.read_csv('Final dataset.csv', sep=';', decimal=',', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv('Final dataset.csv', sep=';', decimal=',', encoding='latin-1')
    
    print(f"📊 Dataset loaded: {len(df)} samples")
    
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
    print(f"📋 Using {len(available_features)} features")
    
    # Remove missing data
    df_clean = df[available_features + ['participant_id', 'diagnosis']].dropna()
    print(f"📊 Clean dataset: {len(df_clean)} samples")
    
    # INVESTIGATION 1: Perfect separation check
    print("\n🔍 INVESTIGATION 1: Perfect separation analysis")
    X = df_clean[available_features]
    y = df_clean['diagnosis']
    
    # Check if any single feature perfectly separates classes
    perfect_features = []
    for feature in available_features:
        if feature in df_clean.columns:
            asd_values = df_clean[df_clean['diagnosis'] == 1][feature]
            typical_values = df_clean[df_clean['diagnosis'] == 0][feature]
            
            # Check for perfect separation
            asd_max = asd_values.max()
            asd_min = asd_values.min()
            typ_max = typical_values.max()
            typ_min = typical_values.min()
            
            if asd_max < typ_min or typ_max < asd_min:
                perfect_features.append(feature)
                print(f"   🚨 PERFECT SEPARATOR: {feature}")
                print(f"      ASD range: [{asd_min:.3f}, {asd_max:.3f}]")
                print(f"      Typical range: [{typ_min:.3f}, {typ_max:.3f}]")
    
    if len(perfect_features) == 0:
        print("   ✅ No single features with perfect separation")
    else:
        print(f"   ❌ Found {len(perfect_features)} features with perfect separation!")
    
    # INVESTIGATION 2: Participant-level vs Sample-level split
    print("\n🔍 INVESTIGATION 2: Split level comparison")
    
    # Sample-level split (WRONG - causes leakage)
    X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Participant-level split (CORRECT)
    participant_info = df_clean.groupby('participant_id')['diagnosis'].first().reset_index()
    train_pids, test_pids = train_test_split(
        participant_info['participant_id'].values,
        test_size=0.2,
        stratify=participant_info['diagnosis'].values,
        random_state=42
    )
    
    train_mask = df_clean['participant_id'].isin(train_pids)
    test_mask = df_clean['participant_id'].isin(test_pids)
    
    X_train_participant = df_clean.loc[train_mask, available_features]
    X_test_participant = df_clean.loc[test_mask, available_features]
    y_train_participant = df_clean.loc[train_mask, 'diagnosis']
    y_test_participant = df_clean.loc[test_mask, 'diagnosis']
    
    # Train models on both splits
    scaler_sample = StandardScaler()
    X_train_sample_scaled = scaler_sample.fit_transform(X_train_sample)
    X_test_sample_scaled = scaler_sample.transform(X_test_sample)
    
    scaler_participant = StandardScaler()
    X_train_participant_scaled = scaler_participant.fit_transform(X_train_participant)
    X_test_participant_scaled = scaler_participant.transform(X_test_participant)
    
    # Random Forest on both
    rf_sample = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_sample.fit(X_train_sample_scaled, y_train_sample)
    sample_pred = rf_sample.predict_proba(X_test_sample_scaled)[:, 1]
    sample_auc = roc_auc_score(y_test_sample, sample_pred)
    
    rf_participant = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_participant.fit(X_train_participant_scaled, y_train_participant)
    participant_pred = rf_participant.predict_proba(X_test_participant_scaled)[:, 1]
    participant_auc = roc_auc_score(y_test_participant, participant_pred)
    
    print(f"   Sample-level split AUC: {sample_auc:.3f}")
    print(f"   Participant-level split AUC: {participant_auc:.3f}")
    
    if sample_auc > 0.95:
        print("   🚨 Sample-level split shows leakage!")
    if participant_auc > 0.95:
        print("   🚨 Even participant-level split shows potential issues!")
    
    # INVESTIGATION 3: Data distribution analysis
    print("\n🔍 INVESTIGATION 3: Data distribution analysis")
    
    # Check for artificial patterns
    print("   Checking for artificial patterns...")
    
    # Look for identical samples
    duplicates = df_clean.duplicated(subset=available_features).sum()
    print(f"   Duplicate samples: {duplicates}")
    
    # Check class distribution per participant
    participant_classes = df_clean.groupby('participant_id')['diagnosis'].nunique()
    mixed_participants = (participant_classes > 1).sum()
    print(f"   Participants with mixed classes: {mixed_participants}")
    
    if mixed_participants > 0:
        print("   🚨 CRITICAL: Some participants have both ASD and Typical samples!")
        print("   This causes severe leakage!")
    
    # INVESTIGATION 4: Feature importance analysis
    print("\n🔍 INVESTIGATION 4: Feature importance analysis")
    
    # Get feature importance from the leaky model
    importances = rf_sample.feature_importances_
    feature_importance = list(zip(available_features, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("   Top 5 most important features:")
    for i, (feature, importance) in enumerate(feature_importance[:5]):
        print(f"   {i+1}. {feature}: {importance:.3f}")
    
    # Check if top features are suspiciously important
    top_importance = feature_importance[0][1]
    if top_importance > 0.5:
        print(f"   🚨 Top feature has {top_importance:.3f} importance - suspiciously high!")
    
    # INVESTIGATION 5: Temporal patterns
    print("\n🔍 INVESTIGATION 5: Temporal patterns check")
    
    # Check if there are systematic patterns by sample index
    df_clean['sample_within_participant'] = df_clean.index % 8
    
    # Check if diagnosis correlates with sample position
    correlation_results = []
    for pos in range(8):
        pos_data = df_clean[df_clean['sample_within_participant'] == pos]
        if len(pos_data) > 10:  # Enough samples
            asd_ratio = pos_data['diagnosis'].mean()
            correlation_results.append((pos, asd_ratio, len(pos_data)))
    
    print("   ASD ratio by sample position:")
    for pos, ratio, count in correlation_results:
        print(f"   Position {pos}: {ratio:.3f} ASD ratio ({count} samples)")
    
    # Check for systematic bias
    ratios = [r[1] for r in correlation_results]
    if max(ratios) - min(ratios) > 0.2:
        print("   🚨 Large variation in ASD ratio by position - potential bias!")
    
    # RECOMMENDATIONS
    print("\n💡 RECOMMENDATIONS:")
    
    if len(perfect_features) > 0:
        print("   🚨 CRITICAL: Remove perfect separator features!")
        print(f"      Features to investigate: {perfect_features}")
    
    if mixed_participants > 0:
        print("   🚨 CRITICAL: Fix participant labeling - each participant should have one diagnosis!")
    
    if sample_auc > 0.95 and participant_auc < 0.85:
        print("   ✅ Use participant-level splitting only")
    elif participant_auc > 0.95:
        print("   🚨 CRITICAL: Even participant-level split shows leakage!")
        print("   📋 Investigate data collection methodology")
        print("   📋 Check for systematic biases in data")
    
    print(f"\n📊 Expected realistic AUC range: 0.70-0.85")
    print(f"📊 Your results: {participant_auc:.3f}")
    
    if participant_auc > 0.90:
        print("🚨 CONCLUSION: Data likely contains artificial patterns or collection bias")
    else:
        print("✅ CONCLUSION: Results look more realistic with proper participant-level split")

if __name__ == "__main__":
    investigate_data_leakage()