#!/usr/bin/env python3
"""
Fix Augmentation Leakage - Participant-Level Splitting
Ensures no participant appears in both train and test sets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import warnings
import logging
from datetime import datetime
import os

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ParticipantLevelSplitter:
    """
    Handles participant-level splitting to prevent augmentation leakage
    """
    
    def __init__(self, samples_per_participant=8):
        self.samples_per_participant = samples_per_participant
        self.n_original_participants = None
        self.participant_ids = None
        
    def create_participant_ids(self, total_samples):
        """Create participant IDs assuming each participant has 8 augmented samples"""
        
        if total_samples % self.samples_per_participant != 0:
            raise ValueError(f"Total samples ({total_samples}) not divisible by samples_per_participant ({self.samples_per_participant})")
        
        self.n_original_participants = total_samples // self.samples_per_participant
        
        # Create participant IDs: [0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, ...]
        self.participant_ids = np.repeat(range(self.n_original_participants), self.samples_per_participant)
        
        logger.info(f"Created participant mapping:")
        logger.info(f"   Total samples: {total_samples}")
        logger.info(f"   Original participants: {self.n_original_participants}")
        logger.info(f"   Samples per participant: {self.samples_per_participant}")
        
        return self.participant_ids
    
    def participant_level_split(self, X, y, test_size=0.2, random_state=42):
        """
        Split data at participant level to prevent leakage
        """
        if self.participant_ids is None:
            self.create_participant_ids(len(X))
        
        logger.info(f"\n🔧 Performing participant-level split...")
        
        # Get unique participants and their labels (using first sample of each participant)
        unique_participants = np.arange(self.n_original_participants)
        
        # Get labels for stratification (take every 8th sample to get one per participant)
        participant_labels = y[::self.samples_per_participant]
        
        logger.info(f"   Participant label distribution: {np.bincount(participant_labels)}")
        
        # Split participants (not samples)
        train_participants, test_participants = train_test_split(
            unique_participants,
            test_size=test_size,
            random_state=random_state,
            stratify=participant_labels
        )
        
        logger.info(f"   Train participants: {len(train_participants)}")
        logger.info(f"   Test participants: {len(test_participants)}")
        
        # Get sample indices for train/test
        train_mask = np.isin(self.participant_ids, train_participants)
        test_mask = np.isin(self.participant_ids, test_participants)
        
        # Split the data
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        # Validation
        self.validate_no_leakage(train_participants, test_participants)
        
        logger.info(f"✅ Participant-level split completed:")
        logger.info(f"   Training samples: {len(X_train)} ({len(train_participants)} participants)")
        logger.info(f"   Test samples: {len(X_test)} ({len(test_participants)} participants)")
        logger.info(f"   Train class distribution: {np.bincount(y_train)}")
        logger.info(f"   Test class distribution: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test, train_participants, test_participants
    
    def validate_no_leakage(self, train_participants, test_participants):
        """Validate that no participant appears in both train and test"""
        
        overlap = set(train_participants).intersection(set(test_participants))
        
        if overlap:
            raise ValueError(f"❌ LEAKAGE DETECTED! Participants {overlap} appear in both train and test sets!")
        else:
            logger.info("✅ No participant leakage detected")
    
    def create_participant_cv_folds(self, X_train, y_train, train_participants, n_splits=5):
        """
        Create cross-validation folds at participant level
        """
        logger.info(f"\n🔄 Creating {n_splits}-fold CV at participant level...")
        
        # Get unique train participants and their labels
        unique_train_participants = np.unique(train_participants)
        participant_labels = []
        
        # Create mapping from participant ID to label using the first sample of each participant
        for pid in unique_train_participants:
            # Find samples belonging to this participant in the training set
            train_participant_ids = self.participant_ids[:len(X_train)]
            pid_mask = train_participant_ids == pid
            
            if np.any(pid_mask):
                # Get the first occurrence's label
                matching_indices = np.where(pid_mask)[0]
                first_index = matching_indices[0]
                participant_labels.append(y_train.iloc[first_index])
            else:
                raise ValueError(f"Participant {pid} not found in training data")
        
        participant_labels = np.array(participant_labels)
        
        # Create CV splits at participant level
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        cv_folds = []
        for fold_idx, (train_pids_idx, val_pids_idx) in enumerate(skf.split(unique_train_participants, participant_labels)):
            
            # Get actual participant IDs
            fold_train_pids = unique_train_participants[train_pids_idx]
            fold_val_pids = unique_train_participants[val_pids_idx]
            
            # Get sample indices for the current training data
            train_participant_ids = self.participant_ids[:len(X_train)]
            fold_train_mask = np.isin(train_participant_ids, fold_train_pids)
            fold_val_mask = np.isin(train_participant_ids, fold_val_pids)
            
            cv_folds.append({
                'fold': fold_idx + 1,
                'train_mask': fold_train_mask,
                'val_mask': fold_val_mask,
                'train_participants': fold_train_pids,
                'val_participants': fold_val_pids,
                'train_samples': np.sum(fold_train_mask),
                'val_samples': np.sum(fold_val_mask)
            })
            
            logger.info(f"   Fold {fold_idx + 1}: {len(fold_train_pids)} train participants ({np.sum(fold_train_mask)} samples), "
                       f"{len(fold_val_pids)} val participants ({np.sum(fold_val_mask)} samples)")
        
        return cv_folds

class NeuroGaitFixedLeakage:
    """
    NeuroGait analysis with proper participant-level splitting
    """
    
    def __init__(self):
        self.output_dir = f"neurogait_fixed_leakage_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.splitter = ParticipantLevelSplitter(samples_per_participant=8)
        
    def load_and_clean_data(self, csv_path='Final dataset.csv'):
        """Load and clean data (mean features only)"""
        logger.info(f"\n📊 Loading data from {csv_path}...")
        
        # Load data
        df = pd.read_csv(csv_path, sep=';', decimal=',')
        logger.info(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
        
        # Convert target
        if 'class' in df.columns:
            df['class'] = df['class'].map({'A': 1, 'T': 0})
            df = df.rename(columns={'class': 'diagnosis'})
            logger.info("✅ Converted target: 'A'->1 (ASD), 'T'->0 (Typical)")
        
        # Keep only mean features (eliminate variance/std)
        logger.info("\n🔧 Filtering to mean features only...")
        
        cols_to_keep = ['diagnosis']
        for col in df.columns:
            col_clean = col.strip()
            
            # Keep mean coordinate features
            if col_clean.startswith('mean-') and any(coord in col_clean for coord in ['-x-', '-y-', '-z-']):
                cols_to_keep.append(col)
            # Keep mean angle features  
            elif col_clean.startswith('mean ') and len(col_clean.split()) >= 2:
                cols_to_keep.append(col)
            # Keep ROM features
            elif col_clean.startswith('Rom'):
                cols_to_keep.append(col)
            # Keep gait parameters
            elif col_clean in ['MaxStLe', 'MaxStWi', 'StrLe', 'GaCT', 'StaT', 'SwiT', 'Velocity']:
                cols_to_keep.append(col)
            # Keep other single features
            elif col_clean in ['HaTiLPos', 'HaTiRPos', 'MaxDBFE', 'MinDBFE', 'Threshold']:
                cols_to_keep.append(col)
        
        df_filtered = df[cols_to_keep]
        
        # Basic cleaning
        df_filtered = df_filtered.dropna(axis=1, how='all')
        for col in df_filtered.columns:
            if col != 'diagnosis' and df_filtered[col].nunique() <= 1:
                df_filtered = df_filtered.drop(columns=[col])
        
        logger.info(f"✅ Kept {len(df_filtered.columns)-1} features after filtering")
        logger.info(f"📊 Class distribution: {df_filtered['diagnosis'].value_counts().to_dict()}")
        
        return df_filtered
    
    def create_feature_pipeline(self, n_features=30):
        """Create conservative feature selection pipeline"""
        
        class FeatureProcessor:
            def __init__(self, n_features):
                self.n_features = n_features
                self.scaler = StandardScaler()
                self.feature_selector = SelectKBest(f_classif, k=n_features)
                self.selected_features_ = None
                
            def fit(self, X, y):
                logger.info(f"\n🔧 Feature processing pipeline...")
                logger.info(f"   Input shape: {X.shape}")
                
                # Remove highly correlated features
                corr_matrix = X.corr().abs()
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                to_drop = [column for column in upper_triangle.columns 
                          if any(upper_triangle[column] > 0.8)]
                
                logger.info(f"   Removing {len(to_drop)} highly correlated features")
                self.corr_features_to_drop = to_drop
                X_decorr = X.drop(columns=to_drop)
                
                # Scale
                X_scaled = self.scaler.fit_transform(X_decorr)
                
                # Select features
                actual_k = min(self.n_features, X_scaled.shape[1])
                self.feature_selector.set_params(k=actual_k)
                X_selected = self.feature_selector.fit_transform(X_scaled, y)
                
                selected_indices = self.feature_selector.get_support(indices=True)
                self.selected_features_ = X_decorr.columns[selected_indices].tolist()
                
                logger.info(f"   Selected {len(self.selected_features_)} features")
                return self
                
            def transform(self, X):
                X_decorr = X.drop(columns=self.corr_features_to_drop, errors='ignore')
                X_scaled = self.scaler.transform(X_decorr)
                X_selected = self.feature_selector.transform(X_scaled)
                return X_selected
                
            def fit_transform(self, X, y):
                return self.fit(X, y).transform(X)
        
        return FeatureProcessor(n_features)
    
    def train_model_with_participant_cv(self, X_train, X_test, y_train, y_test, train_participants):
        """Train model with participant-level cross-validation"""
        logger.info(f"\n🚀 Training model with participant-level CV...")
        
        # Create CV folds at participant level
        cv_folds = self.splitter.create_participant_cv_folds(X_train, y_train, train_participants)
        
        # Configure conservative model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=2.0,
            reg_lambda=2.0,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Participant-level cross-validation
        cv_scores = []
        for fold in cv_folds:
            X_fold_train = X_train[fold['train_mask']]
            X_fold_val = X_train[fold['val_mask']]
            y_fold_train = y_train[fold['train_mask']]
            y_fold_val = y_train[fold['val_mask']]
            
            fold_model = xgb.XGBClassifier(**model.get_params())
            fold_model.fit(X_fold_train, y_fold_train)
            
            y_pred_proba = fold_model.predict_proba(X_fold_val)[:, 1]
            fold_auc = roc_auc_score(y_fold_val, y_pred_proba)
            cv_scores.append(fold_auc)
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        logger.info(f"   Participant-level CV AUC: {cv_mean:.4f} ± {cv_std:.4f}")
        
        # Train final model on all training data
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'cv_auc_mean': cv_mean,
            'cv_auc_std': cv_std,
            'test_auc': roc_auc_score(y_test, y_pred_proba),
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred),
            'test_recall': recall_score(y_test, y_pred),
            'test_f1': f1_score(y_test, y_pred),
        }
        
        logger.info(f"\n📊 Results (NO PARTICIPANT LEAKAGE):")
        logger.info(f"   CV AUC:      {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
        logger.info(f"   Test AUC:    {metrics['test_auc']:.4f}")
        logger.info(f"   Accuracy:    {metrics['test_accuracy']:.4f}")
        logger.info(f"   Precision:   {metrics['test_precision']:.4f}")
        logger.info(f"   Recall:      {metrics['test_recall']:.4f}")
        logger.info(f"   F1-score:    {metrics['test_f1']:.4f}")
        
        # Performance assessment
        if metrics['test_auc'] > 0.9:
            logger.warning("   ⚠️  Still very high - check for other issues")
        elif metrics['test_auc'] > 0.8:
            logger.info("   ✅ Good performance")
        elif metrics['test_auc'] > 0.7:
            logger.info("   ✅ Realistic performance")
        else:
            logger.info("   ℹ️  Lower performance - may be more realistic")
        
        return metrics, model
    
    def run_analysis(self):
        """Run complete analysis with participant-level splitting"""
        logger.info(f"\n🔍 Starting NeuroGait Analysis with FIXED LEAKAGE - {datetime.now()}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        
        try:
            # 1. Load data
            df = self.load_and_clean_data()
            
            # 2. Participant-level split (CRITICAL!)
            X = df.drop('diagnosis', axis=1)
            y = df['diagnosis']
            
            X_train, X_test, y_train, y_test, train_participants, test_participants = \
                self.splitter.participant_level_split(X, y, test_size=0.2, random_state=42)
            
            # 3. Feature processing
            feature_pipeline = self.create_feature_pipeline(n_features=25)
            X_train_processed = feature_pipeline.fit_transform(X_train, y_train)
            X_test_processed = feature_pipeline.transform(X_test)
            
            # 4. Train with participant-level CV
            metrics, model = self.train_model_with_participant_cv(
                X_train_processed, X_test_processed, y_train, y_test, train_participants
            )
            
            # 5. Save results
            import json
            with open(f"{self.output_dir}/results.json", 'w') as f:
                json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
            
            logger.info(f"\n✅ Analysis completed with NO PARTICIPANT LEAKAGE!")
            logger.info(f"📁 Results saved to: {self.output_dir}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            raise


def main():
    """Run the fixed analysis"""
    try:
        analyzer = NeuroGaitFixedLeakage()
        results = analyzer.run_analysis()
        
        print("\n" + "="*60)
        print("🏁 ANALYSIS COMPLETE - NO PARTICIPANT LEAKAGE")
        print("="*60)
        print(f"Test AUC: {results['test_auc']:.4f}")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"CV AUC: {results['cv_auc_mean']:.4f} ± {results['cv_auc_std']:.4f}")
        
        if results['test_auc'] < 0.85:
            print("✅ Realistic performance achieved!")
        else:
            print("⚠️  Still high - may have other issues")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()