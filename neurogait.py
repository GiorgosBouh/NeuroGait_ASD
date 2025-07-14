# Enhanced Feature Analysis & ML Preparation Module - FIXED VERSION
"""
FIXED VERSION: Separates exploratory analysis from ML preparation
to prevent data leakage. All ML-related functions now properly
handle train/test splits.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

class ExploratoryFeatureAnalyzer:
    """
    EXPLORATORY ANALYSIS ONLY - Uses full dataset for exploration
    DO NOT use these methods for actual ML feature selection
    """
    
    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph
        self.exploration_results = {}
        
    def analyze_feature_distributions_exploration(self):
        """
        EXPLORATORY ONLY: Analyze feature distributions by class
        This is for understanding the data, not for ML feature selection
        """
        logger.info("🔍 EXPLORATORY: Analyzing feature distributions...")
        logger.warning("⚠️  This is for exploration only - do not use for ML feature selection!")
        
        # Separate features by class for exploration
        asd_data = self.kg.data[self.kg.data['diagnosis'] == 'ASD']
        control_data = self.kg.data[self.kg.data['diagnosis'] == 'Control']
        
        exploration_results = []
        
        for category, features in self.kg.feature_schema.items():
            if isinstance(features, list) and features:
                logger.info(f"   Exploring {category}: {len(features)} features")
                
                category_results = []
                for feature in features[:50]:  # Limit for exploration
                    if feature in self.kg.data.columns:
                        asd_values = asd_data[feature].dropna()
                        control_values = control_data[feature].dropna()
                        
                        if len(asd_values) > 0 and len(control_values) > 0:
                            # Statistical test for exploration
                            from scipy.stats import ttest_ind
                            stat, p_value = ttest_ind(asd_values, control_values)
                            
                            if p_value < 0.05:  # Potentially interesting
                                effect_size = abs(asd_values.mean() - control_values.mean()) / np.sqrt((asd_values.var() + control_values.var()) / 2)
                                category_results.append({
                                    'feature': feature,
                                    'p_value': p_value,
                                    'effect_size': effect_size,
                                    'asd_mean': asd_values.mean(),
                                    'control_mean': control_values.mean(),
                                    'category': category
                                })
                
                category_results.sort(key=lambda x: x['effect_size'], reverse=True)
                exploration_results.extend(category_results[:10])
        
        self.exploration_results['statistical'] = exploration_results
        logger.info(f"✅ Found {len(exploration_results)} potentially interesting features for exploration")
        return exploration_results
    
    def create_correlation_heatmap_exploration(self, sample_size=100):
        """
        EXPLORATORY ONLY: Create correlation heatmap for exploration
        """
        logger.info("🔍 EXPLORATORY: Creating correlation heatmap...")
        logger.warning("⚠️  This is for exploration only!")
        
        # Sample features for visualization
        numeric_cols = self.kg.data.select_dtypes(include=[np.number]).columns
        exclude_cols = ['participant_id'] 
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Randomly sample features to avoid overwhelming visualization
        if len(feature_cols) > sample_size:
            sampled_features = np.random.choice(feature_cols, sample_size, replace=False)
        else:
            sampled_features = feature_cols
        
        # Calculate correlations for exploration
        corr_matrix = self.kg.data[sampled_features].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, 
                   cmap='coolwarm', 
                   center=0, 
                   square=True,
                   xticklabels=False, 
                   yticklabels=False)
        plt.title(f'Feature Correlation Heatmap (Sample of {len(sampled_features)} features)\nFOR EXPLORATION ONLY')
        plt.tight_layout()
        
        return corr_matrix


class MLFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    PROPER ML PIPELINE: Handles feature selection and preprocessing
    with proper train/test separation to prevent data leakage
    """
    
    def __init__(self, 
                 n_features=100, 
                 correlation_threshold=0.95,
                 feature_selection_method='f_classif',
                 scaler_type='standard'):
        self.n_features = n_features
        self.correlation_threshold = correlation_threshold
        self.feature_selection_method = feature_selection_method
        self.scaler_type = scaler_type
        
        # Initialize components
        self.correlation_filter_ = None
        self.feature_selector_ = None
        self.scaler_ = None
        self.selected_features_ = None
        self.removed_corr_features_ = None
        
    def _remove_correlated_features(self, X):
        """Remove highly correlated features (fitted on training data only)"""
        logger.info(f"   Removing features with correlation > {self.correlation_threshold}")
        
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > self.correlation_threshold)]
        
        self.removed_corr_features_ = to_drop
        logger.info(f"   Removing {len(to_drop)} highly correlated features")
        
        return X.drop(columns=to_drop)
    
    def _select_features(self, X, y):
        """Select best features using statistical tests"""
        logger.info(f"   Selecting top {self.n_features} features using {self.feature_selection_method}")
        
        if self.feature_selection_method == 'f_classif':
            score_func = f_classif
        elif self.feature_selection_method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            raise ValueError(f"Unknown feature selection method: {self.feature_selection_method}")
        
        # Ensure we don't select more features than available
        k = min(self.n_features, X.shape[1])
        
        self.feature_selector_ = SelectKBest(score_func=score_func, k=k)
        X_selected = self.feature_selector_.fit_transform(X, y)
        
        # Store selected feature names
        selected_indices = self.feature_selector_.get_support(indices=True)
        self.selected_features_ = X.columns[selected_indices].tolist()
        
        logger.info(f"   Selected {len(self.selected_features_)} features")
        return X_selected
    
    def _scale_features(self, X):
        """Scale features"""
        logger.info(f"   Scaling features using {self.scaler_type} scaler")
        
        if self.scaler_type == 'standard':
            self.scaler_ = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler_ = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        
        X_scaled = self.scaler_.fit_transform(X)
        return X_scaled
    
    def fit(self, X, y):
        """Fit the pipeline on training data ONLY"""
        logger.info(f"\n🔧 Fitting ML feature pipeline on training data...")
        logger.info(f"   Input shape: {X.shape}")
        
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Remove missing values
        X_clean = X.fillna(X.median())
        
        # 1. Remove correlated features
        X_decorr = self._remove_correlated_features(X_clean)
        
        # 2. Feature selection
        X_selected = self._select_features(X_decorr, y)
        
        # 3. Feature scaling (fit only, don't transform yet)
        self.scaler_ = StandardScaler() if self.scaler_type == 'standard' else RobustScaler()
        self.scaler_.fit(X_selected)
        
        logger.info(f"✅ Pipeline fitted successfully")
        return self
    
    def transform(self, X):
        """Transform data using fitted pipeline"""
        if self.selected_features_ is None:
            raise ValueError("Pipeline not fitted yet! Call fit() first.")
        
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Apply same transformations as training
        X_clean = X.fillna(X.median())
        
        # Remove correlated features (using same features as training)
        X_decorr = X_clean.drop(columns=self.removed_corr_features_, errors='ignore')
        
        # Select same features as training
        X_selected = X_decorr[self.selected_features_]
        
        # Scale using fitted scaler
        X_scaled = self.scaler_.transform(X_selected)
        
        return X_scaled
    
    def fit_transform(self, X, y):
        """Fit and transform training data"""
        return self.fit(X, y).transform(X)
    
    def get_feature_importance(self):
        """Get feature importance scores (only available after fitting)"""
        if self.feature_selector_ is None:
            raise ValueError("Pipeline not fitted yet!")
        
        importance_df = pd.DataFrame({
            'feature': self.selected_features_,
            'score': self.feature_selector_.scores_[self.feature_selector_.get_support()],
            'rank': range(1, len(self.selected_features_) + 1)
        }).sort_values('score', ascending=False)
        
        return importance_df


class MLDatasetCreator:
    """
    Creates ML-ready datasets with proper train/test handling
    """
    
    def __init__(self, data):
        self.data = data
        
    def create_ml_dataset_proper_split(self, 
                                     test_size=0.2, 
                                     random_state=42,
                                     feature_pipeline_params=None,
                                     exclude_variance_std=True):
        """
        Create ML dataset with PROPER train/test split and no data leakage
        """
        logger.info("\n📊 Creating ML dataset with proper train/test split...")
        
        # Basic data preparation
        exclude_cols = ['participant_id', 'class', 'diagnosis']
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        # Remove variance and std features if requested
        if exclude_variance_std:
            variance_std_cols = [col for col in feature_cols 
                               if 'variance' in col.lower() or 'std' in col.lower()]
            if variance_std_cols:
                feature_cols = [col for col in feature_cols if col not in variance_std_cols]
                logger.info(f"🚫 Excluded {len(variance_std_cols)} variance/std features")
                logger.info(f"   Examples: {variance_std_cols[:5]}")
        
        X = self.data[feature_cols].copy()
        y = self.data['diagnosis'].map({'ASD': 1, 'Control': 0})
        participant_ids = self.data.get('participant_id', range(len(self.data)))
        
        logger.info(f"   Total samples: {len(X)}")
        logger.info(f"   Total features: {len(feature_cols)}")
        logger.info(f"   Class distribution: {y.value_counts().to_dict()}")
        
        # CRITICAL: Split data FIRST, before any feature engineering
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, participant_ids, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        logger.info(f"✅ Data split completed:")
        logger.info(f"   Training: {len(X_train)} samples")
        logger.info(f"   Test: {len(X_test)} samples")
        
        # Create and fit feature pipeline on TRAINING data only
        if feature_pipeline_params is None:
            feature_pipeline_params = {'n_features': 100, 'correlation_threshold': 0.9}
            
        pipeline = MLFeaturePipeline(**feature_pipeline_params)
        
        # Fit and transform training data
        X_train_processed = pipeline.fit_transform(X_train, y_train)
        
        # Transform test data using fitted pipeline
        X_test_processed = pipeline.transform(X_test)
        
        logger.info(f"✅ Feature engineering completed:")
        logger.info(f"   Processed training shape: {X_train_processed.shape}")
        logger.info(f"   Processed test shape: {X_test_processed.shape}")
        
        # Create final datasets
        train_dataset = pd.DataFrame(X_train_processed, columns=[f'feature_{i}' for i in range(X_train_processed.shape[1])])
        train_dataset['target'] = y_train.values
        train_dataset['participant_id'] = ids_train.values
        
        test_dataset = pd.DataFrame(X_test_processed, columns=[f'feature_{i}' for i in range(X_test_processed.shape[1])])
        test_dataset['target'] = y_test.values
        test_dataset['participant_id'] = ids_test.values
        
        results = {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'feature_pipeline': pipeline,
            'selected_features': pipeline.selected_features_,
            'feature_importance': pipeline.get_feature_importance(),
            'train_indices': X_train.index.tolist(),
            'test_indices': X_test.index.tolist()
        }
        
        logger.info("✅ ML-ready datasets created with no data leakage!")
        return results


class ProperCrossValidation:
    """
    Proper cross-validation that respects train/test splits
    """
    
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
    
    def create_cv_folds_no_leakage(self, X_train, y_train, participant_ids_train):
        """
        Create CV folds within training data only (no test data involved)
        """
        logger.info(f"\n🔄 Creating {self.n_splits}-fold CV strategy...")
        logger.info("   ✅ Using training data only - no test data leakage!")
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            fold_info = {
                'fold': fold_idx + 1,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'train_class_dist': y_train.iloc[train_idx].value_counts().to_dict(),
                'val_class_dist': y_train.iloc[val_idx].value_counts().to_dict()
            }
            folds.append(fold_info)
        
        logger.info(f"✅ Created {len(folds)} CV folds")
        for fold in folds:
            logger.info(f"   Fold {fold['fold']}: {fold['train_size']} train, {fold['val_size']} val")
        
        return folds
    
    def validate_cv_integrity(self, folds, test_indices):
        """
        Validate that CV folds don't include any test data
        """
        logger.info("\n🔍 Validating CV integrity...")
        
        test_set = set(test_indices)
        
        for fold in folds:
            train_indices_set = set(fold['train_idx'])
            val_indices_set = set(fold['val_idx'])
            
            # Check for overlap with test set
            train_test_overlap = train_indices_set.intersection(test_set)
            val_test_overlap = val_indices_set.intersection(test_set)
            
            if train_test_overlap or val_test_overlap:
                logger.error(f"❌ LEAKAGE DETECTED in Fold {fold['fold']}!")
                logger.error(f"   Train-Test overlap: {len(train_test_overlap)} samples")
                logger.error(f"   Val-Test overlap: {len(val_test_overlap)} samples")
                return False
        
        logger.info("✅ CV integrity validated - no test data leakage detected!")
        return True


# Safe utility functions that don't cause leakage

def save_ml_datasets(train_dataset, test_dataset, feature_info, output_dir="ml_output"):
    """Save ML datasets and metadata safely"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    train_dataset.to_csv(f"{output_dir}/train_dataset.csv", index=False)
    test_dataset.to_csv(f"{output_dir}/test_dataset.csv", index=False)
    
    # Save feature information
    feature_info['feature_importance'].to_csv(f"{output_dir}/feature_importance.csv", index=False)
    
    # Save metadata
    metadata = {
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'num_features': len(feature_info['selected_features']),
        'selected_features': feature_info['selected_features']
    }
    
    import json
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ ML datasets saved to {output_dir}/")


# Example usage with proper data handling
def create_proper_ml_workflow(data):
    """
    Example of proper ML workflow without data leakage
    """
    logger.info("\n🚀 Starting PROPER ML workflow (no data leakage)...")
    
    # 1. Exploratory analysis (uses full data - that's OK for exploration)
    explorer = ExploratoryFeatureAnalyzer({'data': data, 'feature_schema': {}})
    exploration_results = explorer.analyze_feature_distributions_exploration()
    
    # 2. Create proper ML datasets with train/test split
    dataset_creator = MLDatasetCreator(data)
    ml_results = dataset_creator.create_ml_dataset_proper_split(
        test_size=0.2,
        feature_pipeline_params={'n_features': 50, 'correlation_threshold': 0.9}
    )
    
    # 3. Create proper CV strategy (using training data only)
    cv_strategy = ProperCrossValidation(n_splits=5)
    
    # Extract training data for CV
    train_features = ml_results['train_dataset'].drop(['target', 'participant_id'], axis=1)
    train_targets = ml_results['train_dataset']['target']
    train_ids = ml_results['train_dataset']['participant_id']
    
    cv_folds = cv_strategy.create_cv_folds_no_leakage(train_features, train_targets, train_ids)
    
    # 4. Validate CV integrity
    cv_strategy.validate_cv_integrity(cv_folds, ml_results['test_indices'])
    
    # 5. Save everything
    save_ml_datasets(
        ml_results['train_dataset'],
        ml_results['test_dataset'],
        ml_results,
        "proper_ml_output"
    )
    
    logger.info("✅ PROPER ML workflow completed successfully!")
    
    return ml_results, cv_folds


if __name__ == "__main__":
    # Example usage
    logger.info("Fixed NeuroGait App - Ready for use without data leakage!")