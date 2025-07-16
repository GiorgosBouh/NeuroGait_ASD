# Create the main analysis script
cat > neurogait_analysis.py << 'EOF'
#!/usr/bin/env python3
"""
NeuroGait ASD ML Analysis - COMPLETE WORKING VERSION
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
import warnings
import logging
import json
from datetime import datetime
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class NeuroGaitAnalysisFixed:
    def __init__(self):
        self.output_dir = f"neurogait_fixed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_and_basic_clean(self, csv_path='Final dataset.csv', exclude_variance_std=True):
        """Load data with minimal preprocessing"""
        logger.info(f"\n📊 Loading raw data from {csv_path}...")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ Dataset not found: {csv_path}")
        
        # Load with European decimal format
        df = pd.read_csv(csv_path, sep=';', decimal=',')
        logger.info(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
        
        # Convert target variable
        if 'class' in df.columns:
            if set(df['class'].unique()) == {'A', 'T'}:
                df['class'] = df['class'].map({'A': 1, 'T': 0})
                df = df.rename(columns={'class': 'diagnosis'})
                logger.info("✅ Converted target: 'A'->1 (ASD), 'T'->0 (Typical)")
            else:
                raise ValueError(f"❌ Unexpected class values: {df['class'].unique()}")
        else:
            raise ValueError("❌ No 'class' column found")
        
        # Remove variance and std features if requested
        if exclude_variance_std:
            variance_std_cols = [col for col in df.columns 
                               if 'variance' in col.lower() or 'std' in col.lower()]
            if variance_std_cols:
                df = df.drop(columns=variance_std_cols)
                logger.info(f"🚫 Excluded {len(variance_std_cols)} variance/std features")
        
        # Basic cleaning
        initial_cols = len(df.columns)
        df = df.dropna(axis=1, how='all')  # Remove all-NaN columns
        
        # Keep only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'diagnosis' not in numeric_cols:
            numeric_cols.append('diagnosis')
        df = df[numeric_cols]
        
        # Remove constant columns
        for col in df.columns:
            if col != 'diagnosis' and df[col].nunique() <= 1:
                df = df.drop(columns=[col])
        
        logger.info(f"✅ Basic cleaning: {initial_cols} -> {len(df.columns)} columns")
        logger.info(f"📊 Class distribution:\n{df['diagnosis'].value_counts()}")
        
        return df
    
    def create_feature_pipeline(self, n_features=50, correlation_threshold=0.9):
        """Create feature processing pipeline"""
        
        class FeatureSelector:
            def __init__(self, n_features, correlation_threshold):
                self.n_features = n_features
                self.correlation_threshold = correlation_threshold
                self.selected_features_ = None
                self.scaler = StandardScaler()
                self.feature_selector = SelectKBest(f_classif, k=n_features)
                
            def fit(self, X, y):
                logger.info(f"\n🔧 Fitting feature pipeline...")
                logger.info(f"   Input shape: {X.shape}")
                
                # Remove highly correlated features
                corr_matrix = X.corr().abs()
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                to_drop = [column for column in upper_triangle.columns 
                          if any(upper_triangle[column] > self.correlation_threshold)]
                
                logger.info(f"   Removing {len(to_drop)} highly correlated features")
                self.corr_features_to_drop = to_drop
                X_decorr = X.drop(columns=to_drop)
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X_decorr)
                
                # Select best features
                self.feature_selector.set_params(k=min(self.n_features, X_scaled.shape[1]))
                X_selected = self.feature_selector.fit_transform(X_scaled, y)
                
                # Store selected feature names
                selected_indices = self.feature_selector.get_support(indices=True)
                self.selected_features_ = X_decorr.columns[selected_indices].tolist()
                
                logger.info(f"   Final feature count: {len(self.selected_features_)}")
                return self
                
            def transform(self, X):
                if self.selected_features_ is None:
                    raise ValueError("Pipeline not fitted yet!")
                
                # Apply same transformations
                X_decorr = X.drop(columns=self.corr_features_to_drop, errors='ignore')
                X_scaled = self.scaler.transform(X_decorr)
                X_selected = self.feature_selector.transform(X_scaled)
                
                return X_selected
                
            def fit_transform(self, X, y):
                return self.fit(X, y).transform(X)
        
        return FeatureSelector(n_features, correlation_threshold)
    
    def train_and_evaluate_model(self, X_train, X_test, y_train, y_test, model_name="XGBoost"):
        """Train model with proper cross-validation"""
        logger.info(f"\n🚀 Training {model_name}...")
        logger.info(f"   Training set: {X_train.shape}")
        logger.info(f"   Test set: {X_test.shape}")
        
        # Configure model with regularization
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
        
        # Cross-validation on training set only
        logger.info("   Performing 5-fold cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        
        logger.info(f"   CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Train final model
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'test_auc': roc_auc_score(y_test, y_pred_proba),
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred),
            'test_recall': recall_score(y_test, y_pred),
            'test_f1': f1_score(y_test, y_pred),
        }
        
        # Print results
        logger.info(f"\n📊 {model_name} Results:")
        logger.info(f"   CV AUC:      {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
        logger.info(f"   Test AUC:    {metrics['test_auc']:.4f}")
        logger.info(f"   Accuracy:    {metrics['test_accuracy']:.4f}")
        logger.info(f"   Precision:   {metrics['test_precision']:.4f}")
        logger.info(f"   Recall:      {metrics['test_recall']:.4f}")
        logger.info(f"   F1-score:    {metrics['test_f1']:.4f}")
        
        # Check if results are realistic
        if metrics['test_auc'] > 0.95:
            logger.warning("   ⚠️  AUC > 0.95 - possible overfitting or leakage!")
        elif metrics['test_auc'] < 0.55:
            logger.warning("   ⚠️  AUC < 0.55 - model not learning")
        else:
            logger.info("   ✅ Performance in realistic range")
        
        return metrics, model
    
    def run_analysis(self, exclude_variance_std=True):
        """Run complete analysis with proper data handling"""
        logger.info(f"\n🔍 Starting Fixed NeuroGait Analysis - {datetime.now()}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        
        try:
            # 1. Load and basic cleaning
            df = self.load_and_basic_clean(exclude_variance_std=exclude_variance_std)
            
            # 2. Split data FIRST
            X = df.drop('diagnosis', axis=1)
            y = df['diagnosis']
            
            logger.info(f"\n✂️  Splitting data...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"   Training: {len(X_train)} samples")
            logger.info(f"   Test: {len(X_test)} samples")
            
            # 3. Feature engineering - fit on train only
            feature_pipeline = self.create_feature_pipeline(n_features=50, correlation_threshold=0.9)
            
            X_train_processed = feature_pipeline.fit_transform(X_train, y_train)
            X_test_processed = feature_pipeline.transform(X_test)
            
            logger.info(f"   Processed shapes: train {X_train_processed.shape}, test {X_test_processed.shape}")
            
            # 4. Train and evaluate model
            metrics, model = self.train_and_evaluate_model(
                X_train_processed, X_test_processed, y_train, y_test
            )
            
            # 5. Save results
            self.save_results(metrics)
            
            logger.info(f"\n✅ Analysis completed successfully!")
            logger.info(f"📁 Results saved to: {self.output_dir}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            raise
    
    def save_results(self, metrics):
        """Save results to JSON file"""
        # Convert numpy types to Python types
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.floating, np.integer)):
                serializable_metrics[key] = float(value)
            else:
                serializable_metrics[key] = value
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'Fixed NeuroGait Analysis (No Data Leakage)',
            'metrics': serializable_metrics
        }
        
        with open(f"{self.output_dir}/analysis_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"💾 Results saved to: {self.output_dir}/analysis_report.json")


def main():
    """Main function to run the analysis"""
    try:
        # Create analyzer instance
        analyzer = NeuroGaitAnalysisFixed()
        
        # Run analysis
        results = analyzer.run_analysis(exclude_variance_std=True)
        
        # Print summary
        print("\n" + "="*60)
        print("🏁 ANALYSIS COMPLETE")
        print("="*60)
        print(f"Test AUC: {results['test_auc']:.4f}")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Test F1-score: {results['test_f1']:.4f}")
        print(f"CV AUC: {results['cv_auc_mean']:.4f} ± {results['cv_auc_std']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Main analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
EOF

# Now run the complete analysis
python neurogait_analysis.py