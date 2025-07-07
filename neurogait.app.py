import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Add your neurogait_kg_builder to path
sys.path.append('.')  # Adjust path as needed

# Import your KG builder (adjust import based on your actual module structure)
# from neurogait_kg_builder import YourKGClass  

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuroGaitAnalyzer:
    """Analyzer specifically for your NeuroGait ASD dataset"""
    
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        self.xgb_model = xgb.XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.data = None
        self.feature_names = []
        
    def load_excel_dataset(self, filepath='Final dataset.xlsx'):
        """Load your Final dataset.xlsx file"""
        
        logger.info(f"Loading dataset from {filepath}")
        
        try:
            # Load the Excel file
            self.data = pd.read_excel(filepath)
            logger.info(f"Loaded {len(self.data)} rows, {len(self.data.columns)} columns")
            
            # Display basic info
            print("\nDataset Info:")
            print(f"Shape: {self.data.shape}")
            print(f"\nColumns: {list(self.data.columns)[:10]}...")  # First 10 columns
            
            # Check for target column (might be 'class', 'diagnosis', 'label', etc.)
            possible_target_cols = ['class', 'Class', 'diagnosis', 'Diagnosis', 
                                   'label', 'Label', 'target', 'Target']
            
            target_col = None
            for col in possible_target_cols:
                if col in self.data.columns:
                    target_col = col
                    break
            
            if target_col:
                print(f"\nTarget column found: '{target_col}'")
                print(f"Target distribution:\n{self.data[target_col].value_counts()}")
                
                # Convert target to binary if needed
                if self.data[target_col].dtype == 'object':
                    # Assuming 'A' or 'ASD' = 1, 'T' or 'Control' = 0
                    unique_vals = self.data[target_col].unique()
                    print(f"Unique target values: {unique_vals}")
                    
                    # Create mapping based on common patterns
                    if 'A' in unique_vals or 'ASD' in unique_vals:
                        target_mapping = {
                            'A': 1, 'ASD': 1, 'Autism': 1,
                            'T': 0, 'TD': 0, 'Control': 0, 'Typical': 0
                        }
                    else:
                        # Default mapping - adjust based on your data
                        target_mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                    
                    self.data['diagnosis_binary'] = self.data[target_col].map(target_mapping)
                    print(f"Target mapping: {target_mapping}")
            else:
                logger.warning("No target column found! Please specify the target column.")
                
            # Identify feature columns (numeric columns excluding target)
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove target and ID columns
            exclude_cols = ['diagnosis_binary', target_col, 'ID', 'id', 'participant_id', 
                           'Participant_ID', 'index']
            self.feature_names = [col for col in numeric_cols 
                                 if col not in exclude_cols and not col.startswith('Unnamed')]
            
            print(f"\nIdentified {len(self.feature_names)} feature columns")
            print(f"Sample features: {self.feature_names[:10]}...")
            
            # Check for participant ID column
            id_cols = ['ID', 'id', 'participant_id', 'Participant_ID', 'ParticipantID']
            self.id_col = None
            for col in id_cols:
                if col in self.data.columns:
                    self.id_col = col
                    print(f"\nParticipant ID column: '{self.id_col}'")
                    print(f"Unique participants: {self.data[self.id_col].nunique()}")
                    break
                    
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def prepare_ml_data(self, target_col='diagnosis_binary'):
        """Prepare data for ML with participant-level splitting"""
        
        if self.data is None:
            raise ValueError("No data loaded! Call load_excel_dataset() first.")
        
        # Get features and target
        X = self.data[self.feature_names].values
        y = self.data[target_col].values
        
        # Handle missing values
        if np.any(np.isnan(X)):
            logger.warning(f"Found {np.sum(np.isnan(X))} missing values. Filling with median.")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Participant-level split
        if self.id_col and self.id_col in self.data.columns:
            participant_ids = self.data[self.id_col].values
            
            # Get unique participants
            unique_participants = list(set(participant_ids))
            participant_labels = []
            
            for p_id in unique_participants:
                # Get label for this participant (use majority vote if multiple samples)
                p_mask = participant_ids == p_id
                p_labels = y[p_mask]
                participant_labels.append(int(np.round(np.mean(p_labels))))
            
            # Split by participants
            train_participants, test_participants = train_test_split(
                unique_participants,
                test_size=0.3,
                random_state=42,
                stratify=participant_labels
            )
            
            # Create masks
            train_mask = np.isin(participant_ids, train_participants)
            test_mask = np.isin(participant_ids, test_participants)
            
            logger.info(f"Participant-level split: {len(train_participants)} train, "
                       f"{len(test_participants)} test participants")
        else:
            # Regular split if no participant IDs
            logger.warning("No participant IDs found. Using regular train/test split.")
            train_mask = np.zeros(len(X), dtype=bool)
            test_mask = np.zeros(len(X), dtype=bool)
            
            indices = np.arange(len(X))
            train_idx, test_idx = train_test_split(
                indices, test_size=0.3, random_state=42, stratify=y
            )
            train_mask[train_idx] = True
            test_mask[test_idx] = True
        
        # Split data
        self.X_train = X[train_mask]
        self.X_test = X[test_mask]
        self.y_train = y[train_mask]
        self.y_test = y[test_mask]
        
        # Scale
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\nML Data Prepared:")
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        print(f"Class distribution (train): {np.bincount(self.y_train)}")
        print(f"Class distribution (test): {np.bincount(self.y_test)}")
    
    def train_and_evaluate(self):
        """Train models and evaluate performance"""
        
        print("\n" + "="*80)
        print("TRAINING MODELS")
        print("="*80)
        
        # Train Random Forest
        print("\nTraining Random Forest...")
        self.rf_model.fit(self.X_train_scaled, self.y_train)
        
        # Train XGBoost
        print("Training XGBoost...")
        self.xgb_model.fit(self.X_train_scaled, self.y_train)
        
        # Evaluate
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        results = {}
        
        for name, model in [('Random Forest', self.rf_model), ('XGBoost', self.xgb_model)]:
            y_pred = model.predict(self.X_test_scaled)
            y_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            print(f"\n{name}:")
            print("-" * 40)
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred, 
                                      target_names=['Control', 'ASD']))
            
            print("\nConfusion Matrix:")
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"TN: {cm[0,0]:3d}  FP: {cm[0,1]:3d}")
            print(f"FN: {cm[1,0]:3d}  TP: {cm[1,1]:3d}")
            
            # Store results
            results[name] = {
                'predictions': y_pred,
                'probabilities': y_proba,
                'confusion_matrix': cm
            }
        
        return results
    
    def analyze_feature_importance(self, top_n=30):
        """Analyze which gait features are most important"""
        
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        # Get feature importance from both models
        rf_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        xgb_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Print top features
        print(f"\nTop {top_n} Features - Random Forest:")
        print("-" * 60)
        for idx, row in rf_importance.head(top_n).iterrows():
            print(f"{row['feature']:40s}: {row['importance']:.4f}")
        
        print(f"\n\nTop {top_n} Features - XGBoost:")
        print("-" * 60)
        for idx, row in xgb_importance.head(top_n).iterrows():
            print(f"{row['feature']:40s}: {row['importance']:.4f}")
        
        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # RF plot
        rf_importance.head(top_n).plot(kind='barh', x='feature', y='importance', 
                                       ax=ax1, legend=False)
        ax1.set_title('Random Forest - Feature Importance', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Importance Score')
        ax1.invert_yaxis()
        
        # XGB plot
        xgb_importance.head(top_n).plot(kind='barh', x='feature', y='importance', 
                                        ax=ax2, legend=False)
        ax2.set_title('XGBoost - Feature Importance', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Importance Score')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('neurogait_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Find consensus top features
        print("\n" + "="*80)
        print("CONSENSUS TOP FEATURES (appear in both models' top 15)")
        print("="*80)
        
        rf_top15 = set(rf_importance.head(15)['feature'])
        xgb_top15 = set(xgb_importance.head(15)['feature'])
        consensus = rf_top15.intersection(xgb_top15)
        
        print(f"\nFeatures in both top 15 lists ({len(consensus)} features):")
        for feature in sorted(consensus):
            rf_rank = rf_importance[rf_importance['feature'] == feature].index[0] + 1
            xgb_rank = xgb_importance[xgb_importance['feature'] == feature].index[0] + 1
            print(f"  {feature:40s} (RF rank: {rf_rank}, XGB rank: {xgb_rank})")
        
        return rf_importance, xgb_importance
    
    def analyze_by_feature_groups(self):
        """Group features by type and analyze importance"""
        
        print("\n" + "="*80)
        print("FEATURE GROUP ANALYSIS")
        print("="*80)
        
        # Define feature groups based on common gait analysis patterns
        groups = {
            'Spatial': ['step', 'stride', 'length', 'width', 'distance'],
            'Temporal': ['time', 'duration', 'cadence', 'speed', 'velocity'],
            'Variability': ['std', 'cv', 'var', 'SD', 'CV'],
            'Asymmetry': ['asymmetry', 'asym', 'difference', 'ratio'],
            'Angular': ['angle', 'degree', 'rotation'],
            'Force': ['force', 'grf', 'pressure', 'load']
        }
        
        # Get RF importance
        rf_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.rf_model.feature_importances_
        })
        
        # Categorize features
        feature_groups = {}
        for feature in self.feature_names:
            feature_lower = feature.lower()
            assigned = False
            
            for group, keywords in groups.items():
                if any(keyword in feature_lower for keyword in keywords):
                    if group not in feature_groups:
                        feature_groups[group] = []
                    feature_groups[group].append(feature)
                    assigned = True
                    break
            
            if not assigned:
                if 'Other' not in feature_groups:
                    feature_groups['Other'] = []
                feature_groups['Other'].append(feature)
        
        # Calculate group importance
        group_importance = {}
        for group, features in feature_groups.items():
            importances = rf_importance[rf_importance['feature'].isin(features)]['importance']
            group_importance[group] = {
                'mean_importance': importances.mean(),
                'total_importance': importances.sum(),
                'n_features': len(features),
                'top_feature': features[np.argmax(importances.values)] if len(features) > 0 else None
            }
        
        # Print results
        sorted_groups = sorted(group_importance.items(), 
                             key=lambda x: x[1]['total_importance'], 
                             reverse=True)
        
        for group, stats in sorted_groups:
            print(f"\n{group} Features:")
            print(f"  Number of features: {stats['n_features']}")
            print(f"  Mean importance: {stats['mean_importance']:.4f}")
            print(f"  Total importance: {stats['total_importance']:.4f}")
            if stats['top_feature']:
                print(f"  Most important: {stats['top_feature']}")


def main():
    """Main execution function for your NeuroGait data"""
    
    print("NeuroGait ASD Analysis")
    print("="*80)
    
    # Initialize analyzer
    analyzer = NeuroGaitAnalyzer()
    
    # Load your dataset
    analyzer.load_excel_dataset('Final dataset.xlsx')
    
    # Prepare ML data
    analyzer.prepare_ml_data()
    
    # Train and evaluate
    results = analyzer.train_and_evaluate()
    
    # Analyze feature importance
    rf_imp, xgb_imp = analyzer.analyze_feature_importance(top_n=30)
    
    # Analyze by feature groups
    analyzer.analyze_by_feature_groups()
    
    # Save detailed results
    with open('neurogait_analysis_results.txt', 'w') as f:
        f.write("NeuroGait ASD Analysis Results\n")
        f.write("="*80 + "\n\n")
        
        f.write("Top 50 Features by Random Forest:\n")
        f.write("-"*60 + "\n")
        for idx, row in rf_imp.head(50).iterrows():
            f.write(f"{idx+1:3d}. {row['feature']:40s}: {row['importance']:.6f}\n")
        
        f.write("\n\nTop 50 Features by XGBoost:\n")
        f.write("-"*60 + "\n")
        for idx, row in xgb_imp.head(50).iterrows():
            f.write(f"{idx+1:3d}. {row['feature']:40s}: {row['importance']:.6f}\n")
    
    print("\n✓ Analysis complete! Results saved to:")
    print("  - neurogait_feature_importance.png")
    print("  - neurogait_analysis_results.txt")
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
    
    # Additional analysis you can run:
    # 1. Get predictions for new data
    # 2. Export models for deployment
    # 3. Generate SHAP explanations
    # etc.neurogait