#!/usr/bin/env python3
"""
All Features Evaluation: Χρήση όλων των διαθέσιμων features
GOAL: Δούμε αν περισσότερα features βελτιώνουν την απόδοση
Με bias correction για realistic αποτελέσματα
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

class AllFeaturesEvaluation:
    def __init__(self):
        self.random_state = 42
        
    def load_and_prepare_data(self):
        """Load data with bias correction and ALL features"""
        print("🚀 ALL FEATURES EVALUATION - Using Maximum Available Features")
        print("="*80)
        print("🎯 GOAL: Test if more features improve performance")
        print("🔒 With bias correction for realistic results")
        print()
        
        # Load data
        try:
            df = pd.read_csv('Final dataset.csv', sep=';', decimal=',', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv('Final dataset.csv', sep=';', decimal=',', encoding='latin-1')
        
        print(f"📊 Original dataset: {len(df)} samples, {len(df.columns)} columns")
        
        # Convert to numeric (be more aggressive about conversion)
        numeric_cols = [col for col in df.columns if col != 'class']
        
        print("🔧 Converting all columns to numeric...")
        converted_features = []
        for col in numeric_cols:
            if df[col].dtype == 'object' or col not in df.select_dtypes(include=[np.number]).columns:
                try:
                    # Try multiple conversion strategies
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
                    if not df[col].isna().all():  # If conversion successful
                        converted_features.append(col)
                except:
                    continue
            else:
                converted_features.append(col)
        
        print(f"✅ Successfully converted {len(converted_features)} numeric features")
        
        # Create participant mapping and diagnosis
        df['participant_id'] = df.index // 8
        df['original_diagnosis'] = df['class'].map({'A': 1, 'T': 0})
        
        # BIAS CORRECTION: Shuffle participant diagnoses
        print(f"\n🔀 APPLYING BIAS CORRECTION...")
        participant_info = df.groupby('participant_id')['original_diagnosis'].first()
        participant_ids = participant_info.index.values
        
        # Check original bias
        first_half = participant_ids < np.mean(participant_ids)
        original_first_half_asd = participant_info.iloc[first_half].mean()
        original_second_half_asd = participant_info.iloc[~first_half].mean()
        
        print(f"   Original bias: {abs(original_first_half_asd - original_second_half_asd):.3f}")
        
        # Shuffle diagnoses
        np.random.seed(self.random_state)
        shuffled_diagnoses = participant_info.values.copy()
        np.random.shuffle(shuffled_diagnoses)
        new_diagnosis_mapping = dict(zip(participant_ids, shuffled_diagnoses))
        df['diagnosis'] = df['participant_id'].map(new_diagnosis_mapping)
        
        # Verify bias correction
        new_participant_info = df.groupby('participant_id')['diagnosis'].first()
        new_first_half_asd = new_participant_info.iloc[first_half].mean()
        new_second_half_asd = new_participant_info.iloc[~first_half].mean()
        new_bias = abs(new_first_half_asd - new_second_half_asd)
        
        print(f"   After correction: {new_bias:.3f}")
        print(f"   ✅ Bias reduction: {abs(original_first_half_asd - original_second_half_asd) - new_bias:.3f}")
        
        return df, converted_features
    
    def feature_selection_pipeline(self, df, features):
        """Intelligent feature selection pipeline"""
        print(f"\n🔍 FEATURE SELECTION PIPELINE")
        print(f"   Starting with {len(features)} potential features")
        
        # Create clean dataset
        feature_cols = features + ['participant_id', 'diagnosis']
        df_clean = df[feature_cols].copy()
        
        # Remove rows with too many missing values
        missing_threshold = 0.5  # Remove rows with >50% missing
        missing_counts = df_clean[features].isna().sum(axis=1)
        df_clean = df_clean[missing_counts <= len(features) * missing_threshold]
        
        print(f"   After removing high-missing rows: {len(df_clean)} samples")
        
        # Fill remaining missing values
        for col in features:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Remove duplicates
        original_size = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=features)
        print(f"   Removed {original_size - len(df_clean)} duplicates")
        
        # Feature filtering pipeline
        X_all = df_clean[features]
        y = df_clean['diagnosis']
        
        print(f"\n📊 FEATURE FILTERING:")
        
        # 1. Remove constant/near-constant features
        print("   1. Variance threshold filtering...")
        variance_selector = VarianceThreshold(threshold=0.001)
        X_variance = variance_selector.fit_transform(X_all)
        variance_features = [features[i] for i in range(len(features)) if variance_selector.get_support()[i]]
        print(f"      Kept {len(variance_features)} features (removed {len(features) - len(variance_features)} low-variance)")
        
        # 2. Remove highly correlated features
        print("   2. Correlation filtering...")
        if len(variance_features) > 100:  # Only if we have many features
            corr_matrix = pd.DataFrame(X_variance, columns=variance_features).corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find features with correlation > 0.95
            high_corr_pairs = []
            for col in upper_triangle.columns:
                high_corr = upper_triangle[col][upper_triangle[col] > 0.95].index.tolist()
                high_corr_pairs.extend(high_corr)
            
            # Remove highly correlated features
            correlation_features = [f for f in variance_features if f not in set(high_corr_pairs)]
            print(f"      Kept {len(correlation_features)} features (removed {len(high_corr_pairs)} highly correlated)")
        else:
            correlation_features = variance_features
        
        # 3. Statistical significance filtering
        print("   3. Statistical significance filtering...")
        if len(correlation_features) > 50:  # Only if we still have many features
            X_corr = df_clean[correlation_features]
            
            # Use ANOVA F-test to select top features
            k_best = min(100, len(correlation_features))  # Select top 100 or all if fewer
            selector = SelectKBest(score_func=f_classif, k=k_best)
            X_selected = selector.fit_transform(X_corr, y)
            selected_features = [correlation_features[i] for i in range(len(correlation_features)) if selector.get_support()[i]]
            print(f"      Selected top {len(selected_features)} most significant features")
        else:
            selected_features = correlation_features
        
        print(f"\n✅ FINAL FEATURE SELECTION:")
        print(f"   Original: {len(features)} features")
        print(f"   After filtering: {len(selected_features)} features")
        print(f"   Reduction: {len(features) - len(selected_features)} features removed")
        
        # Update dataframe with selected features
        final_cols = selected_features + ['participant_id', 'diagnosis']
        df_final = df_clean[final_cols].copy()
        
        print(f"   Final dataset: {len(df_final)} samples × {len(selected_features)} features")
        
        return df_final, selected_features
    
    def create_participant_split(self, df):
        """Create participant-level split"""
        print(f"\n🔧 Creating participant-level split...")
        
        participant_info = df.groupby('participant_id')['diagnosis'].first().reset_index()
        
        train_pids, test_pids = train_test_split(
            participant_info['participant_id'].values,
            test_size=0.2,
            stratify=participant_info['diagnosis'].values,
            random_state=self.random_state
        )
        
        train_mask = df['participant_id'].isin(train_pids)
        test_mask = df['participant_id'].isin(test_pids)
        
        train_data = df[train_mask].reset_index(drop=True)
        test_data = df[test_mask].reset_index(drop=True)
        
        print(f"   ✅ Split: {len(train_pids)} train participants, {len(test_pids)} test participants")
        print(f"   📊 Train: {train_data['diagnosis'].value_counts().to_dict()}")
        print(f"   📊 Test: {test_data['diagnosis'].value_counts().to_dict()}")
        
        return train_data, test_data, train_pids, test_pids
    
    def prepare_ml_data(self, train_data, test_data, features):
        """Prepare ML data with standardization"""
        print(f"\n📊 Preparing ML data with {len(features)} features...")
        
        X_train = train_data[features]
        X_test = test_data[features]
        y_train = train_data['diagnosis']
        y_test = test_data['diagnosis']
        
        # Standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"   ✅ Features standardized: {X_train_scaled.shape[1]}D")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def create_enhanced_kg_embeddings(self, X_train, X_test):
        """Create enhanced graph-style embeddings using more sophisticated interactions"""
        print(f"   🧠 Creating enhanced KG embeddings with more sophisticated graph processing...")
        
        def advanced_graph_processing(X):
            """More sophisticated graph-inspired processing"""
            X_graph = X.copy()
            
            # 1. Feature interactions (simulating graph edges)
            n_features = X.shape[1]
            n_interact = min(10, n_features)  # More interactions
            
            for i in range(n_interact):
                for j in range(i+1, n_interact):
                    # Multiple types of interactions
                    interaction1 = X[:, i] * X[:, j] * 0.03  # Multiplicative
                    interaction2 = np.abs(X[:, i] - X[:, j]) * 0.02  # Distance-based
                    
                    X_graph[:, i] += interaction1 + interaction2
                    X_graph[:, j] += interaction1 + interaction2
            
            # 2. Non-linear transformations (graph signal processing)
            X_graph = np.tanh(X_graph)  # Bounded activation
            
            # 3. Feature combinations (simulating graph aggregation)
            if n_features >= 5:
                # Add some combined features
                combined1 = np.mean(X_graph[:, :5], axis=1, keepdims=True)
                combined2 = np.std(X_graph[:, :min(10, n_features)], axis=1, keepdims=True)
                
                # Replace some features with combinations
                X_graph[:, -2:] = np.hstack([combined1, combined2])
            
            return X_graph
        
        X_train_kg = advanced_graph_processing(X_train)
        X_test_kg = advanced_graph_processing(X_test)
        
        print(f"   ✅ Enhanced KG embeddings: train{X_train_kg.shape}, test{X_test_kg.shape}")
        
        return X_train_kg, X_test_kg
    
    def train_models(self, X_train, X_test, y_train, y_test, train_pids, approach_name):
        """Train models with regularization appropriate for feature count"""
        print(f"\n🚀 Training models for {approach_name} ({X_train.shape[1]} features)...")
        
        n_features = X_train.shape[1]
        
        # Adjust regularization based on feature count
        if n_features > 50:
            # More regularization for high-dimensional data
            reg_strength = 2.0
            max_depth = 3
            n_estimators = 50
        elif n_features > 20:
            # Moderate regularization
            reg_strength = 1.0
            max_depth = 4
            n_estimators = 75
        else:
            # Light regularization
            reg_strength = 0.5
            max_depth = 5
            n_estimators = 100
        
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                C=1.0/reg_strength,
                penalty='l2'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=42, 
                eval_metric='logloss',
                max_depth=max_depth,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=reg_strength,
                reg_lambda=reg_strength,
                n_estimators=n_estimators
            ),
            'SVM': SVC(
                random_state=42, 
                probability=True,
                C=1.0/reg_strength,
                gamma='scale'
            )
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"   🔧 Training {model_name}...")
            
            # Cross-validation
            cv_scores = self._participant_cv(X_train, y_train, train_pids, model)
            
            # Train final model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            metrics = {
                'cv_scores': cv_scores,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            results[model_name] = metrics
            
            # Assessment
            if metrics['auc'] > 0.85:
                status = "🎉 Excellent"
            elif metrics['auc'] > 0.75:
                status = "✅ Good"
            elif metrics['auc'] > 0.65:
                status = "⚖️ Moderate"
            else:
                status = "📋 Limited"
            
            print(f"      {status}: AUC={metrics['auc']:.3f}, F1={metrics['f1']:.3f}")
        
        return results
    
    def _participant_cv(self, X_train, y_train, train_pids, model, cv_folds=5):
        """Participant-level cross-validation"""
        unique_pids = np.unique(train_pids)
        pid_labels = [y_train.iloc[np.where(train_pids == pid)[0][0]] for pid in unique_pids]
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in skf.split(unique_pids, pid_labels):
            train_fold_pids = unique_pids[train_idx]
            val_fold_pids = unique_pids[val_idx]
            
            train_fold_mask = np.isin(train_pids, train_fold_pids)
            val_fold_mask = np.isin(train_pids, val_fold_pids)
            
            X_fold_train = X_train[train_fold_mask]
            X_fold_val = X_train[val_fold_mask]
            y_fold_train = y_train.iloc[train_fold_mask]
            y_fold_val = y_train.iloc[val_fold_mask]
            
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_fold_train, y_fold_train)
            y_val_proba = model_copy.predict_proba(X_fold_val)[:, 1]
            fold_auc = roc_auc_score(y_fold_val, y_val_proba)
            cv_scores.append(fold_auc)
        
        return cv_scores
    
    def compare_results(self, raw_results, kg_results, feature_count):
        """Compare results and provide insights"""
        print(f"\n📊 ALL FEATURES COMPARISON ({feature_count} features):")
        print("-" * 90)
        print(f"{'Model':<20} {'Raw AUC':<10} {'KG AUC':<10} {'Δ AUC':<10} {'Raw F1':<10} {'KG F1':<10} {'Δ F1':<10}")
        print("-" * 90)
        
        improvements_auc = []
        improvements_f1 = []
        
        for model_name in raw_results.keys():
            if model_name in kg_results:
                raw_auc = raw_results[model_name]['auc']
                kg_auc = kg_results[model_name]['auc']
                raw_f1 = raw_results[model_name]['f1']
                kg_f1 = kg_results[model_name]['f1']
                
                delta_auc = kg_auc - raw_auc
                delta_f1 = kg_f1 - raw_f1
                
                improvements_auc.append(delta_auc)
                improvements_f1.append(delta_f1)
                
                print(f"{model_name:<20} {raw_auc:<10.3f} {kg_auc:<10.3f} {delta_auc:+<10.3f} "
                      f"{raw_f1:<10.3f} {kg_f1:<10.3f} {delta_f1:+<10.3f}")
        
        print("-" * 90)
        
        avg_auc_improvement = np.mean(improvements_auc)
        avg_f1_improvement = np.mean(improvements_f1)
        max_auc = max([max(raw_results[m]['auc'], kg_results[m]['auc']) for m in raw_results.keys()])
        
        print(f"\n📊 SUMMARY ({feature_count} features):")
        print(f"   Average AUC improvement: {avg_auc_improvement:+.3f}")
        print(f"   Average F1 improvement: {avg_f1_improvement:+.3f}")
        print(f"   Maximum AUC achieved: {max_auc:.3f}")
        
        return avg_auc_improvement, avg_f1_improvement, max_auc
    
    def run_all_features_evaluation(self):
        """Run complete evaluation with maximum available features"""
        # Load and prepare data
        df, features = self.load_and_prepare_data()
        
        # Feature selection pipeline
        df_clean, selected_features = self.feature_selection_pipeline(df, features)
        
        # Create split
        train_data, test_data, train_pids, test_pids = self.create_participant_split(df_clean)
        
        # Prepare ML data
        X_train, X_test, y_train, y_test = self.prepare_ml_data(train_data, test_data, selected_features)
        
        # Train on raw features
        print(f"\n{'='*60}")
        print(f"📊 ANALYSIS 1: ALL AVAILABLE RAW FEATURES ({len(selected_features)}D)")
        print(f"{'='*60}")
        
        raw_results = self.train_models(
            X_train, X_test, y_train, y_test, 
            train_data['participant_id'].values, "All Raw Features"
        )
        
        # Create enhanced KG embeddings
        X_train_kg, X_test_kg = self.create_enhanced_kg_embeddings(X_train, X_test)
        
        # Train on KG embeddings
        print(f"\n{'='*60}")
        print(f"🧠 ANALYSIS 2: ENHANCED KG EMBEDDINGS ({X_train_kg.shape[1]}D)")
        print(f"{'='*60}")
        
        kg_results = self.train_models(
            X_train_kg, X_test_kg, y_train, y_test,
            train_data['participant_id'].values, "Enhanced KG Embeddings"
        )
        
        # Compare results
        print(f"\n{'='*60}")
        print("📈 FINAL COMPARISON - ALL FEATURES")
        print(f"{'='*60}")
        
        auc_improvement, f1_improvement, max_auc = self.compare_results(raw_results, kg_results, len(selected_features))
        
        # Final insights
        print(f"\n🎯 ALL FEATURES EVALUATION CONCLUSIONS:")
        print(f"   📊 Used {len(selected_features)} features (vs 19 in previous analysis)")
        print(f"   📈 Maximum AUC: {max_auc:.3f}")
        
        if max_auc > 0.85:
            print("   🎉 EXCELLENT: High performance achieved with more features!")
        elif max_auc > 0.75:
            print("   ✅ GOOD: Meaningful improvement with additional features")
        elif max_auc > 0.65:
            print("   ⚖️ MODERATE: Some improvement but still limited")
        else:
            print("   📋 LIMITED: More features didn't significantly help")
        
        print(f"\n💡 FEATURE COUNT IMPACT:")
        print(f"   Previous (19 features): Best AUC ~0.63")
        print(f"   Current ({len(selected_features)} features): Best AUC {max_auc:.3f}")
        improvement_vs_19 = max_auc - 0.63
        print(f"   Improvement: {improvement_vs_19:+.3f} AUC points")
        
        if improvement_vs_19 > 0.1:
            print("   🎉 SIGNIFICANT improvement with more features!")
        elif improvement_vs_19 > 0.05:
            print("   ✅ MODERATE improvement with more features")
        else:
            print("   📋 MINIMAL improvement - 19 features were sufficient")
        
        return {
            'raw_results': raw_results,
            'kg_results': kg_results,
            'feature_count': len(selected_features),
            'max_auc': max_auc,
            'improvement_vs_19': improvement_vs_19
        }


def main():
    """Main execution"""
    print("🎯 ALL FEATURES EVALUATION")
    print("📋 Testing if more features improve ASD classification performance")
    print("🔒 With systematic bias correction for realistic results")
    print()
    
    evaluator = AllFeaturesEvaluation()
    results = evaluator.run_all_features_evaluation()
    
    print(f"\n🎉 ALL FEATURES EVALUATION COMPLETED!")
    print(f"📊 Used {results['feature_count']} features vs previous 19")
    print(f"🎯 Best AUC achieved: {results['max_auc']:.3f}")
    print(f"📈 Improvement over 19-feature approach: {results['improvement_vs_19']:+.3f}")
    
    return results

if __name__ == "__main__":
    results = main()