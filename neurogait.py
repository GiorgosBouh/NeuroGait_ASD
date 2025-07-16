# Remove the problematic script
rm neurogait.py

# Create a clean, working version
cat > neurogait_clean.py << 'ENDFILE'
#!/usr/bin/env python3
"""
NeuroGait ASD ML Analysis - CLEAN VERSION WITH ADDITIONAL LEAKAGE PROTECTION
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
import json
from datetime import datetime
import os

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class NeuroGaitAnalysisClean:
    def __init__(self):
        self.output_dir = f"neurogait_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_and_aggressive_clean(self, csv_path='Final dataset.csv'):
        """Load data with aggressive cleaning to prevent leakage"""
        logger.info(f"\n📊 Loading data with aggressive cleaning...")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ Dataset not found: {csv_path}")
        
        df = pd.read_csv(csv_path, sep=';', decimal=',')
        logger.info(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
        
        # Convert target
        if 'class' in df.columns:
            df['class'] = df['class'].map({'A': 1, 'T': 0})
            df = df.rename(columns={'class': 'diagnosis'})
            logger.info("✅ Converted target: 'A'->1 (ASD), 'T'->0 (Typical)")
        
        # AGGRESSIVE FEATURE REMOVAL to prevent leakage
        logger.info("\n🚫 Aggressive feature removal...")
        
        # 1. Remove variance/std features
        variance_std_cols = [col for col in df.columns 
                           if 'variance' in col.lower() or 'std' in col.lower()]
        df = df.drop(columns=variance_std_cols)
        logger.info(f"   Removed {len(variance_std_cols)} variance/std features")
        
        # 2. Remove potentially suspicious features
        suspicious_patterns = [
            'threshold', 'count', 'score', 'index', 'id', 'label',
            'time', 'duration', 'length', 'gait', 'stance', 'swing'
        ]
        
        suspicious_cols = []
        for col in df.columns:
            if col == 'diagnosis':
                continue
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in suspicious_patterns):
                suspicious_cols.append(col)
        
        df = df.drop(columns=suspicious_cols)
        logger.info(f"   Removed {len(suspicious_cols)} potentially suspicious features")
        logger.info(f"   Examples: {suspicious_cols[:5]}")
        
        # 3. Keep only mean features and ROM features
        remaining_cols = ['diagnosis']
        for col in df.columns:
            if col == 'diagnosis':
                continue
            col_lower = col.lower()
            if col_lower.startswith('mean-') or col_lower.startswith('rom'):
                remaining_cols.append(col)
        
        df = df[remaining_cols]
        logger.info(f"   Kept only mean and ROM features: {len(df.columns)-1} features")
        
        # 4. Remove features that are too predictive (potential leakage)
        X_temp = df.drop('diagnosis', axis=1)
        y_temp = df['diagnosis']
        
        high_auc_features = []
        for col in X_temp.columns:
            try:
                if X_temp[col].nunique() > 1:  # Skip constant features
                    auc = roc_auc_score(y_temp, X_temp[col])
                    if auc > 0.85 or auc < 0.15:  # Too predictive
                        high_auc_features.append(col)
            except:
                continue
        
        if high_auc_features:
            df = df.drop(columns=high_auc_features)
            logger.info(f"   Removed {len(high_auc_features)} highly predictive features")
        
        # Final cleaning
        df = df.dropna(axis=1, how='all')
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'diagnosis' not in numeric_cols:
            numeric_cols.append('diagnosis')
        df = df[numeric_cols]
        
        # Remove constant features
        for col in df.columns:
            if col != 'diagnosis' and df[col].nunique() <= 1:
                df = df.drop(columns=[col])
        
        logger.info(f"✅ Final dataset: {len(df)} samples, {len(df.columns)-1} features")
        logger.info(f"📊 Class distribution: {df['diagnosis'].value_counts().to_dict()}")
        
        return df
    
    def create_conservative_pipeline(self, n_features=20):
        """Create very conservative feature pipeline"""
        
        class ConservativeSelector:
            def __init__(self, n_features):
                self.n_features = n_features
                self.scaler = StandardScaler()
                self.feature_selector = SelectKBest(f_classif, k=n_features)
                
            def fit(self, X, y):
                logger.info(f"\n🔧 Conservative feature selection...")
                logger.info(f"   Input shape: {X.shape}")
                
                # Very aggressive correlation removal
                corr_matrix = X.corr().abs()
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                to_drop = [column for column in upper_triangle.columns 
                          if any(upper_triangle[column] > 0.7)]  # Lower threshold
                
                logger.info(f"   Removing {len(to_drop)} correlated features (threshold=0.7)")
                self.corr_features_to_drop = to_drop
                X_decorr = X.drop(columns=to_drop)
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X_decorr)
                
                # Select fewer features
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
        
        return ConservativeSelector(n_features)
    
    def train_conservative_model(self, X_train, X_test, y_train, y_test):
        """Train very conservative model"""
        logger.info(f"\n🚀 Training conservative model...")
        
        # Very conservative XGBoost settings
        model = xgb.XGBClassifier(
            n_estimators=50,        # Reduced
            max_depth=3,            # Reduced
            learning_rate=0.01,     # Much lower
            subsample=0.6,          # Lower
            colsample_bytree=0.6,   # Lower
            reg_alpha=5.0,          # Much higher
            reg_lambda=5.0,         # Much higher
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        
        logger.info(f"   CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Train final model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'test_auc': roc_auc_score(y_test, y_pred_proba),
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred),
            'test_recall': recall_score(y_test, y_pred),
            'test_f1': f1_score(y_test, y_pred),
        }
        
        logger.info(f"\n📊 Results:")
        logger.info(f"   CV AUC:      {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
        logger.info(f"   Test AUC:    {metrics['test_auc']:.4f}")
        logger.info(f"   Accuracy:    {metrics['test_accuracy']:.4f}")
        logger.info(f"   Precision:   {metrics['test_precision']:.4f}")
        logger.info(f"   Recall:      {metrics['test_recall']:.4f}")
        logger.info(f"   F1-score:    {metrics['test_f1']:.4f}")
        
        if metrics['test_auc'] > 0.9:
            logger.warning("   ⚠️  Still very high performance - check data!")
        elif metrics['test_auc'] > 0.8:
            logger.info("   ✅ Good performance")
        elif metrics['test_auc'] > 0.7:
            logger.info("   ✅ Realistic performance")
        else:
            logger.info("   ℹ️  Lower performance - may be more realistic")
        
        return metrics
    
    def run_analysis(self):
        """Run conservative analysis"""
        logger.info(f"\n🔍 Starting Conservative NeuroGait Analysis - {datetime.now()}")
        
        try:
            # 1. Aggressive data cleaning
            df = self.load_and_aggressive_clean()
            
            # 2. Split data first
            X = df.drop('diagnosis', axis=1)
            y = df['diagnosis']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"   Split: {len(X_train)} train, {len(X_test)} test")
            
            # 3. Conservative feature processing
            pipeline = self.create_conservative_pipeline(n_features=20)
            X_train_processed = pipeline.fit_transform(X_train, y_train)
            X_test_processed = pipeline.transform(X_test)
            
            # 4. Train conservative model
            metrics = self.train_conservative_model(
                X_train_processed, X_test_processed, y_train, y_test
            )
            
            # 5. Save results
            with open(f"{self.output_dir}/results.json", 'w') as f:
                json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
            
            logger.info(f"\n✅ Analysis completed!")
            logger.info(f"📁 Results saved to: {self.output_dir}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            raise


def main():
    analyzer = NeuroGaitAnalysisClean()
    results = analyzer.run_analysis()
    
    print("\n" + "="*50)
    print("🏁 CONSERVATIVE ANALYSIS COMPLETE")
    print("="*50)
    print(f"Test AUC: {results['test_auc']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"CV AUC: {results['cv_auc_mean']:.4f} ± {results['cv_auc_std']:.4f}")
    
    return results


if __name__ == "__main__":
    main()
ENDFILE

# Run the clean version
python neurogait_clean.py