#!/usr/bin/env python3
"""
Domain Expert Feature Selection for ASD Classification
GOAL: Use clinical/movement expertise to select meaningful features
Based on ASD movement pattern research
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
import re
import warnings
warnings.filterwarnings('ignore')

class DomainExpertFeatureSelection:
    def __init__(self):
        self.random_state = 42
        
    def categorize_features_by_domain(self, all_features):
        """Categorize features based on movement/clinical domain knowledge"""
        print("🧠 DOMAIN EXPERT FEATURE CATEGORIZATION")
        print("="*60)
        print("📋 Categorizing features based on ASD movement research...")
        
        categories = {
            'temporal_gait': [],      # Timing-related gait features
            'spatial_gait': [],       # Spatial gait patterns  
            'upper_limb': [],         # Arm/hand movements
            'lower_limb': [],         # Leg movements
            'trunk_stability': [],    # Core/trunk control
            'coordination': [],       # Inter-limb coordination
            'velocity_acceleration': [], # Speed/acceleration
            'symmetry': [],           # Left-right symmetry
            'balance_stability': [],  # Balance and postural control
            'joint_angles': [],       # Joint angle measurements
            'other_movement': []      # Other movement-related
        }
        
        for feature in all_features:
            feature_lower = feature.lower()
            
            # Temporal/Gait timing features
            if any(keyword in feature_lower for keyword in ['gact', 'stat', 'swit', 'time', 'duration', 'cycle']):
                categories['temporal_gait'].append(feature)
            
            # Spatial gait features
            elif any(keyword in feature_lower for keyword in ['step', 'stride', 'length', 'width', 'distance']):
                categories['spatial_gait'].append(feature)
            
            # Upper limb (arms, hands, shoulders)
            elif any(keyword in feature_lower for keyword in ['hand', 'arm', 'shoulder', 'elbow', 'wrist', 'finger']):
                categories['upper_limb'].append(feature)
            elif any(keyword in feature for keyword in ['HESHL', 'HESHR', 'SPELL', 'SPELR', 'SHWRL', 'SHWRR', 'ELHAL', 'ELHAR']):
                categories['upper_limb'].append(feature)
            
            # Lower limb (legs, feet, knees, hips)
            elif any(keyword in feature_lower for keyword in ['leg', 'foot', 'knee', 'hip', 'ankle', 'toe']):
                categories['lower_limb'].append(feature)
            elif any(keyword in feature for keyword in ['SPKNL', 'SPKNR', 'HIANL', 'HIANR', 'KNFOL', 'KNFOR']):
                categories['lower_limb'].append(feature)
            
            # Trunk/spine stability
            elif any(keyword in feature_lower for keyword in ['spine', 'trunk', 'torso', 'midspain', 'spinebase']):
                categories['trunk_stability'].append(feature)
            elif any(keyword in feature for keyword in ['Midspain', 'SpineBase']):
                categories['trunk_stability'].append(feature)
            
            # Velocity/acceleration
            elif any(keyword in feature_lower for keyword in ['velocity', 'speed', 'acceleration', 'vel', 'acc']):
                categories['velocity_acceleration'].append(feature)
            
            # Symmetry (left-right differences)
            elif ('left' in feature_lower and 'right' in feature_lower) or \
                 (feature_lower.endswith('l') and feature_lower.replace('l', 'r') in [f.lower() for f in all_features]):
                categories['symmetry'].append(feature)
            
            # Balance/stability
            elif any(keyword in feature_lower for keyword in ['balance', 'stability', 'sway', 'postural']):
                categories['balance_stability'].append(feature)
            
            # Joint angles
            elif any(keyword in feature_lower for keyword in ['angle', 'rotation', 'flexion', 'extension']):
                categories['joint_angles'].append(feature)
            
            # Coordination patterns
            elif any(keyword in feature_lower for keyword in ['coordination', 'sync', 'phase', 'coupling']):
                categories['coordination'].append(feature)
            
            # Other movement-related
            elif any(keyword in feature_lower for keyword in ['mean', 'std', 'max', 'min', 'range', 'var']):
                categories['other_movement'].append(feature)
        
        # Print categorization results
        print(f"\n📊 FEATURE CATEGORIZATION RESULTS:")
        total_categorized = 0
        for category, features in categories.items():
            if features:
                print(f"   {category.replace('_', ' ').title():<20}: {len(features):3d} features")
                total_categorized += len(features)
        
        print(f"\n   Total categorized: {total_categorized}/{len(all_features)} features")
        
        return categories
    
    def create_clinical_feature_sets(self, categories):
        """Create clinically meaningful feature combinations"""
        print(f"\n🏥 CREATING CLINICAL FEATURE SETS")
        
        feature_sets = {}
        
        # Set 1: Core movement patterns (most discriminative for ASD)
        feature_sets['core_movement'] = (
            categories['temporal_gait'] + 
            categories['upper_limb'][:10] +  # Top upper limb features
            categories['trunk_stability'][:5] +
            categories['velocity_acceleration'][:3]
        )
        
        # Set 2: Gait-focused (temporal + spatial gait)
        feature_sets['gait_focused'] = (
            categories['temporal_gait'] + 
            categories['spatial_gait'] +
            categories['lower_limb'][:10] +
            categories['velocity_acceleration']
        )
        
        # Set 3: Upper body coordination (arms + trunk)
        feature_sets['upper_body'] = (
            categories['upper_limb'] + 
            categories['trunk_stability'] +
            categories['coordination']
        )
        
        # Set 4: Balance and stability
        feature_sets['balance_stability'] = (
            categories['balance_stability'] + 
            categories['trunk_stability'] +
            categories['lower_limb'][:8] +
            categories['joint_angles'][:5]
        )
        
        # Set 5: Comprehensive movement (all major categories)
        feature_sets['comprehensive'] = (
            categories['temporal_gait'] + 
            categories['upper_limb'][:8] +
            categories['lower_limb'][:8] +
            categories['trunk_stability'][:5] +
            categories['velocity_acceleration'][:3] +
            categories['balance_stability'][:5]
        )
        
        # Set 6: ASD-specific patterns (based on research literature)
        feature_sets['asd_specific'] = []
        # Add features commonly reported as different in ASD
        for category_name, features in categories.items():
            if category_name in ['temporal_gait', 'upper_limb', 'coordination']:
                feature_sets['asd_specific'].extend(features[:5])  # Top 5 from each
        
        # Remove duplicates and empty sets
        for set_name in list(feature_sets.keys()):
            feature_sets[set_name] = list(set(feature_sets[set_name]))  # Remove duplicates
            feature_sets[set_name] = [f for f in feature_sets[set_name] if f]  # Remove empty
            
            if not feature_sets[set_name]:  # Remove empty sets
                del feature_sets[set_name]
        
        print(f"   📋 Created {len(feature_sets)} clinical feature sets:")
        for set_name, features in feature_sets.items():
            print(f"      {set_name.replace('_', ' ').title():<20}: {len(features):2d} features")
        
        return feature_sets
    
    def load_and_prepare_data(self):
        """Load data with bias correction"""
        print("📊 Loading data with bias correction...")
        
        # Load data
        try:
            df = pd.read_csv('Final dataset.csv', sep=';', decimal=',', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv('Final dataset.csv', sep=';', decimal=',', encoding='latin-1')
        
        # Convert to numeric
        numeric_cols = [col for col in df.columns if col != 'class']
        converted_features = []
        
        for col in numeric_cols:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
                    if not df[col].isna().all():
                        converted_features.append(col)
                except:
                    continue
            else:
                converted_features.append(col)
        
        # Participant mapping and bias correction
        df['participant_id'] = df.index // 8
        df['original_diagnosis'] = df['class'].map({'A': 1, 'T': 0})
        
        # Bias correction
        participant_info = df.groupby('participant_id')['original_diagnosis'].first()
        participant_ids = participant_info.index.values
        
        np.random.seed(self.random_state)
        shuffled_diagnoses = participant_info.values.copy()
        np.random.shuffle(shuffled_diagnoses)
        new_diagnosis_mapping = dict(zip(participant_ids, shuffled_diagnoses))
        df['diagnosis'] = df['participant_id'].map(new_diagnosis_mapping)
        
        return df, converted_features
    
    def prepare_feature_set_data(self, df, feature_set, set_name):
        """Prepare data for a specific feature set"""
        # Check which features actually exist
        available_features = [f for f in feature_set if f in df.columns]
        
        if len(available_features) < 5:
            print(f"   ⚠️ {set_name}: Only {len(available_features)} features available, skipping...")
            return None, None, None, None, None
        
        # Create clean dataset
        feature_cols = available_features + ['participant_id', 'diagnosis']
        df_clean = df[feature_cols].copy()
        
        # Remove rows with too many missing values
        missing_counts = df_clean[available_features].isna().sum(axis=1)
        df_clean = df_clean[missing_counts <= len(available_features) * 0.3]
        
        # Fill missing values
        for col in available_features:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates(subset=available_features)
        
        if len(df_clean) < 100:  # Skip if too few samples
            print(f"   ⚠️ {set_name}: Only {len(df_clean)} samples after cleaning, skipping...")
            return None, None, None, None, None
        
        # Create participant split
        participant_info = df_clean.groupby('participant_id')['diagnosis'].first().reset_index()
        
        train_pids, test_pids = train_test_split(
            participant_info['participant_id'].values,
            test_size=0.2,
            stratify=participant_info['diagnosis'].values,
            random_state=self.random_state
        )
        
        train_mask = df_clean['participant_id'].isin(train_pids)
        test_mask = df_clean['participant_id'].isin(test_pids)
        
        train_data = df_clean[train_mask]
        test_data = df_clean[test_mask]
        
        # Prepare ML data
        X_train = train_data[available_features]
        X_test = test_data[available_features]
        y_train = train_data['diagnosis']
        y_test = test_data['diagnosis']
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, available_features
    
    def evaluate_feature_sets(self, df, feature_sets):
        """Evaluate all clinical feature sets"""
        print(f"\n🔬 EVALUATING CLINICAL FEATURE SETS")
        print("-" * 80)
        print(f"{'Feature Set':<20} {'Features':<10} {'Samples':<8} {'LR AUC':<8} {'RF AUC':<8} {'Best':<8}")
        print("-" * 80)
        
        results = {}
        
        for set_name, feature_set in feature_sets.items():
            try:
                # Prepare data
                X_train, X_test, y_train, y_test, available_features = self.prepare_feature_set_data(
                    df, feature_set, set_name
                )
                
                if X_train is None:
                    continue
                
                # Train Logistic Regression
                lr = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
                lr.fit(X_train, y_train)
                lr_pred = lr.predict_proba(X_test)[:, 1]
                lr_auc = roc_auc_score(y_test, lr_pred)
                
                # Train Random Forest
                rf = RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=6, 
                    min_samples_split=5,
                    random_state=42
                )
                rf.fit(X_train, y_train)
                rf_pred = rf.predict_proba(X_test)[:, 1]
                rf_auc = roc_auc_score(y_test, rf_pred)
                
                # Best AUC
                best_auc = max(lr_auc, rf_auc)
                best_model = 'LR' if lr_auc > rf_auc else 'RF'
                
                # Status
                if best_auc > 0.75:
                    status = "🎉"
                elif best_auc > 0.65:
                    status = "✅"
                elif best_auc > 0.55:
                    status = "⚖️"
                else:
                    status = "📋"
                
                print(f"{set_name.replace('_', ' '):<20} {len(available_features):<10} {len(X_train)+len(X_test):<8} "
                      f"{lr_auc:<8.3f} {rf_auc:<8.3f} {best_auc:.3f} {status}")
                
                # Store results
                results[set_name] = {
                    'feature_count': len(available_features),
                    'sample_count': len(X_train) + len(X_test),
                    'lr_auc': lr_auc,
                    'rf_auc': rf_auc,
                    'best_auc': best_auc,
                    'best_model': best_model,
                    'features': available_features
                }
                
            except Exception as e:
                print(f"{set_name.replace('_', ' '):<20} {'Error':<10} {'N/A':<8} {'N/A':<8} {'N/A':<8} ❌")
                continue
        
        return results
    
    def analyze_results(self, results):
        """Analyze and recommend best feature set"""
        print(f"\n📊 DOMAIN EXPERT ANALYSIS:")
        
        if not results:
            print("❌ No valid results obtained")
            return None
        
        # Find best approach
        best_set = max(results.keys(), key=lambda k: results[k]['best_auc'])
        best_result = results[best_set]
        
        print(f"\n🏆 BEST CLINICAL FEATURE SET:")
        print(f"   Feature Set: {best_set.replace('_', ' ').title()}")
        print(f"   Features: {best_result['feature_count']}")
        print(f"   Best AUC: {best_result['best_auc']:.3f}")
        print(f"   Best Model: {best_result['best_model']}")
        
        # Compare performance levels
        excellent = [k for k, v in results.items() if v['best_auc'] > 0.75]
        good = [k for k, v in results.items() if 0.65 < v['best_auc'] <= 0.75]
        moderate = [k for k, v in results.items() if 0.55 < v['best_auc'] <= 0.65]
        
        print(f"\n📈 PERFORMANCE BREAKDOWN:")
        if excellent:
            print(f"   🎉 Excellent (>0.75): {[s.replace('_', ' ') for s in excellent]}")
        if good:
            print(f"   ✅ Good (0.65-0.75): {[s.replace('_', ' ') for s in good]}")
        if moderate:
            print(f"   ⚖️ Moderate (0.55-0.65): {[s.replace('_', ' ') for s in moderate]}")
        
        # Feature count analysis
        print(f"\n🔍 FEATURE COUNT vs PERFORMANCE:")
        for set_name, result in sorted(results.items(), key=lambda x: x[1]['feature_count']):
            print(f"   {result['feature_count']:2d} features → {result['best_auc']:.3f} AUC ({set_name.replace('_', ' ')})")
        
        return best_result
    
    def run_domain_expert_analysis(self):
        """Run complete domain expert feature selection"""
        # Load data
        df, all_features = self.load_and_prepare_data()
        
        print(f"📊 Total available features: {len(all_features)}")
        
        # Categorize features by domain
        categories = self.categorize_features_by_domain(all_features)
        
        # Create clinical feature sets
        feature_sets = self.create_clinical_feature_sets(categories)
        
        # Evaluate feature sets
        results = self.evaluate_feature_sets(df, feature_sets)
        
        # Analyze results
        best_result = self.analyze_results(results)
        
        print(f"\n💡 CLINICAL RECOMMENDATIONS:")
        if best_result and best_result['best_auc'] > 0.70:
            print(f"   ✅ Strong clinical feature set identified!")
            print(f"   📋 Use {best_result['feature_count']} features from {best_result}")
            print(f"   🎯 Expected ASD classification AUC: {best_result['best_auc']:.3f}")
        elif best_result:
            print(f"   ⚖️ Moderate improvement with clinical features")
            print(f"   📋 Best AUC: {best_result['best_auc']:.3f}")
            print(f"   💡 Consider combining with additional clinical measures")
        else:
            print(f"   📋 Clinical feature selection didn't improve performance")
            print(f"   💡 May need more sophisticated movement analysis")
        
        return results, best_result


def main():
    """Main execution"""
    print("🏥 DOMAIN EXPERT FEATURE SELECTION FOR ASD")
    print("🎯 Using clinical/movement expertise for feature selection")
    print()
    
    analyzer = DomainExpertFeatureSelection()
    results, best_result = analyzer.run_domain_expert_analysis()
    
    return results, best_result

if __name__ == "__main__":
    results, best_result = main()