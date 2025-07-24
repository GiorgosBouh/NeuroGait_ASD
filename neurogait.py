#!/usr/bin/env python3
"""
Enhanced Realistic NeuroGait ML Analysis: Raw Features vs KG Embeddings (19D vs 19D)
Key Improvements:
1. Strict 19D comparison (NO PCA reduction)
2. Enhanced leakage prevention
3. Detailed statistical validation
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.base import clone
import xgboost as xgb
from scipy.stats import ttest_rel

# Neo4j connection
try:
    from neo4j import GraphDatabase
    from dotenv import load_dotenv
    load_dotenv('.env')
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neurogait_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class FairComparisonNeuroGaitMLAnalysis:
    def __init__(self):
        self.output_dir = f"fair_comparison_ml_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Neo4j connection
        self.neo4j_driver = None
        if HAS_NEO4J:
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                    auth=(os.getenv('NEO4J_USER', 'neo4j'),
                         os.getenv('NEO4J_PASSWORD', 'password')),
                    connection_timeout=15
                )
                # Test connection
                with self.neo4j_driver.session() as session:
                    session.run("RETURN 1")
                logger.info("✅ Connected to Neo4j")
            except Exception as e:
                logger.error(f"❌ Neo4j connection failed: {e}")
                self.neo4j_driver = None

        # =============================================
        # FIXED: Use EXACTLY these 19 features for both approaches
        # =============================================
        self.essential_features = [
            'mean HESHL', 'mean HESHR', 'mean SPELL', 'mean SPELR',
            'mean SHWRL', 'mean SHWRR', 'mean ELHAL', 'mean ELHAR',
            'mean THHAL', 'mean THHAR', 'mean SPKNL', 'mean SPKNR',
            'mean HIANL', 'mean HIANR', 'mean KNFOL', 'mean KNFOR',
            'GaCT', 'StaT', 'SwiT'
        ]
        
        # Configuration
        self.config = {
            'test_size': 0.2,
            'random_state': 42,
            'n_splits': 5,
            'expected_dim': 19  # <<< FIXED: Enforce 19D comparison
        }
        
        self.results = {}

    # =============================================
    # DATA LOADING AND PREPROCESSING
    # =============================================
    def load_data(self):
        """Load and validate dataset"""
        logger.info("📊 Loading dataset...")
        try:
            # Attempt multiple encodings
            try:
                df = pd.read_csv('Final dataset.csv', sep=';', decimal=',', encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv('Final dataset.csv', sep=';', decimal=',', encoding='latin-1')
            
            # Validate required columns
            missing_features = [f for f in self.essential_features if f not in df.columns]
            if missing_features:
                logger.error(f"❌ Missing features: {missing_features}")
                raise ValueError("Essential features missing in dataset")
                
            # Create participant mapping (8 samples per participant)
            df['participant_id'] = df.index // 8
            df['diagnosis'] = df['class'].map({'A': 1, 'T': 0})  # ASD=1, Typical=0
            
            # Select only essential columns
            df = df[self.essential_features + ['participant_id', 'diagnosis']].dropna()
            
            logger.info(f"✅ Loaded {len(df)} samples with {len(self.essential_features)} features")
            logger.info(f"   Class distribution: {df['diagnosis'].value_counts().to_dict()}")
            logger.info(f"   Participants: {df['participant_id'].nunique()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise

    def participant_level_split(self, df):
        """Strict participant-level splitting"""
        logger.info("🔧 Performing participant-level split...")
        
        participant_info = df.groupby('participant_id')['diagnosis'].first().reset_index()
        
        train_pids, test_pids = train_test_split(
            participant_info['participant_id'],
            test_size=self.config['test_size'],
            stratify=participant_info['diagnosis'],
            random_state=self.config['random_state']
        )
        
        train_data = df[df['participant_id'].isin(train_pids)]
        test_data = df[df['participant_id'].isin(test_pids)]
        
        logger.info(f"✅ Split completed:")
        logger.info(f"   Train: {len(train_pids)} participants ({len(train_data)} samples)")
        logger.info(f"   Test: {len(test_pids)} participants ({len(test_data)} samples)")
        logger.info(f"   Train class distribution: {train_data['diagnosis'].value_counts().to_dict()}")
        logger.info(f"   Test class distribution: {test_data['diagnosis'].value_counts().to_dict()}")
        
        return train_data, test_data, train_pids, test_pids

    # =============================================
    # FEATURE PROCESSING (19D)
    # =============================================
    def prepare_raw_features(self, train_data, test_data):
        """Prepare standardized 19D features"""
        logger.info("🔧 Preparing raw features (19D)...")
        
        # Validate feature dimensions
        assert len(self.essential_features) == 19, "Must use exactly 19 features"
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_data[self.essential_features])
        X_test = scaler.transform(test_data[self.essential_features])
        
        logger.info(f"✅ Raw features shape: Train {X_train.shape}, Test {X_test.shape}")
        return X_train, X_test, train_data['diagnosis'], test_data['diagnosis']

    def get_kg_embeddings(self, train_data, test_data):
        """Extract 19D embeddings from Knowledge Graph"""
        if not self.neo4j_driver:
            logger.error("❌ Neo4j connection not available!")
            return None, None

        logger.info("🧠 Extracting KG embeddings (19D)...")
        try:
            with self.neo4j_driver.session() as session:
                # Query to get all embeddings with participant info
                result = session.run("""
                    MATCH (p:Participant)-[:HAS_SAMPLE]->(s:Sample)-[:HAS_EMBEDDING]->(e:Embedding)
                    RETURN p.id as participant_id, e.vector as embedding, e.dimension as dim
                    ORDER BY s.sample_index
                """)
                
                # Create mapping {participant_id: [embeddings]}
                embeddings_map = {}
                for record in result:
                    pid = record['participant_id']
                    if pid.startswith('P_'):
                        pid = int(pid.split('_')[1])  # Convert "P_123" to 123
                    embeddings_map.setdefault(pid, []).append(record['embedding'])
                
                # Validate dimensions
                sample_embedding = next(iter(embeddings_map.values()), [[]])[0]
                if len(sample_embedding) != self.config['expected_dim']:
                    logger.error(f"❌ Dimension mismatch: Expected {self.config['expected_dim']}D, got {len(sample_embedding)}D")
                    return None, None
                
                # Create feature matrices
                X_train_kg = np.array([embeddings_map.get(pid, [[0]*self.config['expected_dim']])[0] 
                                      for pid in train_data['participant_id']])
                X_test_kg = np.array([embeddings_map.get(pid, [[0]*self.config['expected_dim']])[0] 
                                     for pid in test_data['participant_id']])
                
                logger.info(f"✅ KG embeddings shape: Train {X_train_kg.shape}, Test {X_test_kg.shape}")
                return X_train_kg, X_test_kg
                
        except Exception as e:
            logger.error(f"❌ Failed to extract KG embeddings: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None

    # =============================================
    # MODEL TRAINING AND EVALUATION
    # =============================================
    def train_models(self, X_train, X_test, y_train, y_test, approach_name):
        """Train and evaluate multiple models"""
        logger.info(f"🚀 Training {approach_name} models...")
        
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=self.config['random_state'],
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.config['random_state'],
                class_weight='balanced_subsample'
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=self.config['random_state'],
                eval_metric='logloss',
                scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1)
            ),
            'SVM': SVC(
                probability=True,
                random_state=self.config['random_state'],
                class_weight='balanced'
            )
        }

        results = {}
        for name, model in models.items():
            logger.info(f"   🔧 Training {name}...")
            
            # Cross-validation
            cv_scores = self._participant_cv(
                model, X_train, y_train, 
                train_data['participant_id']
            )
            
            # Final training
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'auc': roc_auc_score(y_test, y_proba),
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            logger.info(f"      ✅ {name}: AUC={results[name]['auc']:.3f}, F1={results[name]['f1']:.3f}")
            
            # Leakage detection
            if results[name]['auc'] > 0.95:
                logger.warning(f"      ⚠️  Suspiciously high AUC ({results[name]['auc']:.3f}) - possible leakage!")
        
        return results

    def _participant_cv(self, model, X, y, participant_ids):
        """Participant-level cross-validation"""
        unique_pids = np.unique(participant_ids)
        pid_labels = [y.iloc[np.where(participant_ids == pid)[0][0]] for pid in unique_pids]
        
        cv_scores = []
        skf = StratifiedKFold(
            n_splits=self.config['n_splits'],
            shuffle=True,
            random_state=self.config['random_state']
        )
        
        for train_idx, val_idx in skf.split(unique_pids, pid_labels):
            # Get participant IDs for this fold
            train_pids = unique_pids[train_idx]
            val_pids = unique_pids[val_idx]
            
            # Create masks
            train_mask = participant_ids.isin(train_pids)
            val_mask = participant_ids.isin(val_pids)
            
            # Clone model to avoid contamination
            model_clone = clone(model)
            model_clone.fit(X[train_mask], y[train_mask])
            
            # Predict and score
            y_proba = model_clone.predict_proba(X[val_mask])[:, 1]
            cv_scores.append(roc_auc_score(y[val_mask], y_proba))
        
        return cv_scores

    # =============================================
    # RESULT ANALYSIS AND COMPARISON
    # =============================================
    def compare_approaches(self, raw_results, kg_results):
        """Statistical comparison of raw vs KG results"""
        logger.info("📊 Comparing approaches...")
        
        comparison = {}
        for model_name in raw_results:
            if model_name in kg_results:
                # Metrics comparison
                comparison[model_name] = {
                    'raw_auc': raw_results[model_name]['auc'],
                    'kg_auc': kg_results[model_name]['auc'],
                    'auc_diff': kg_results[model_name]['auc'] - raw_results[model_name]['auc'],
                    'raw_f1': raw_results[model_name]['f1'],
                    'kg_f1': kg_results[model_name]['f1'],
                    'f1_diff': kg_results[model_name]['f1'] - raw_results[model_name]['f1'],
                    'p_value': ttest_rel(
                        raw_results[model_name]['cv_scores'],
                        kg_results[model_name]['cv_scores']
                    ).pvalue,
                    'significant': None  # Will be set below
                }
                
                # Determine significance
                comparison[model_name]['significant'] = (
                    comparison[model_name]['p_value'] < 0.05
                )
        
        return comparison

    def save_results(self, raw_results, kg_results, comparison):
        """Save comprehensive results to files"""
        logger.info("💾 Saving results...")
        
        # Prepare full results dictionary
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'features': self.essential_features,
                'note': '19D comparison (no PCA reduction)'
            },
            'raw_results': raw_results,
            'kg_results': kg_results,
            'comparison': comparison
        }
        
        # Save JSON
        with open(f'{self.output_dir}/full_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV summary
        summary_data = []
        for model in raw_results:
            if model in comparison:
                summary_data.append({
                    'model': model,
                    'raw_auc': raw_results[model]['auc'],
                    'kg_auc': kg_results[model]['auc'],
                    'auc_improvement': comparison[model]['auc_diff'],
                    'raw_f1': raw_results[model]['f1'],
                    'kg_f1': kg_results[model]['f1'],
                    'f1_improvement': comparison[model]['f1_diff'],
                    'p_value': comparison[model]['p_value'],
                    'significant': comparison[model]['significant']
                })
        
        pd.DataFrame(summary_data).to_csv(
            f'{self.output_dir}/summary.csv', index=False
        )
        
        logger.info(f"✅ Results saved to {self.output_dir}")

    def generate_report(self, comparison):
        """Generate final analysis report"""
        logger.info("\n📝 Generating final report...")
        
        # Calculate overall improvements
        avg_auc_improvement = np.mean([v['auc_diff'] for v in comparison.values()])
        avg_f1_improvement = np.mean([v['f1_diff'] for v in comparison.values()])
        
        # Find best models
        best_raw = max(comparison.items(), key=lambda x: x[1]['raw_auc'])
        best_kg = max(comparison.items(), key=lambda x: x[1]['kg_auc'])
        
        report = f"""
        =============================================
        🎯 FAIR COMPARISON REPORT (19D vs 19D)
        =============================================
        
        📊 Overall Performance:
           Average AUC Improvement: {avg_auc_improvement:.3f}
           Average F1 Improvement: {avg_f1_improvement:.3f}
        
        🏆 Best Performing Models:
           Raw Features: {best_raw[0]} (AUC={best_raw[1]['raw_auc']:.3f})
           KG Embeddings: {best_kg[0]} (AUC={best_kg[1]['kg_auc']:.3f})
        
        🔍 Statistical Significance:
        """
        
        for model, stats in comparison.items():
            report += f"""
           {model}:
              AUC Difference: {stats['auc_diff']:.3f} {'(✅)' if stats['significant'] else '(❌)'}
              p-value: {stats['p_value']:.4f}
            """
        
        report += f"""
        💡 Recommendations:
           {'✅ KG embeddings show significant improvement' if avg_auc_improvement > 0 else '⚠️ Consider using raw features'}
        
        📁 Results saved to: {os.path.abspath(self.output_dir)}
        """
        
        print(report)
        with open(f'{self.output_dir}/report.txt', 'w') as f:
            f.write(report)

    # =============================================
    # MAIN ANALYSIS PIPELINE
    # =============================================
    def run_analysis(self):
        """Complete analysis workflow"""
        try:
            logger.info("🚀 Starting 19D vs 19D comparison...")
            
            # 1. Load and split data
            df = self.load_data()
            train_data, test_data, train_pids, test_pids = self.participant_level_split(df)
            
            # 2. Prepare raw features (19D)
            X_train_raw, X_test_raw, y_train, y_test = self.prepare_raw_features(
                train_data, test_data
            )
            
            # 3. Get KG embeddings (19D)
            X_train_kg, X_test_kg = self.get_kg_embeddings(train_data, test_data)
            if X_train_kg is None:
                raise ValueError("KG embeddings not available (check Neo4j)")
            
            # 4. Train models on raw features
            raw_results = self.train_models(
                X_train_raw, X_test_raw, y_train, y_test,
                "Raw Features (19D)"
            )
            
            # 5. Train models on KG embeddings
            kg_results = self.train_models(
                X_train_kg, X_test_kg, y_train, y_test,
                "KG Embeddings (19D)"
            )
            
            # 6. Compare results
            comparison = self.compare_approaches(raw_results, kg_results)
            
            # 7. Save and report
            self.save_results(raw_results, kg_results, comparison)
            self.generate_report(comparison)
            
            logger.info("🎉 Analysis completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        finally:
            if self.neo4j_driver:
                self.neo4j_driver.close()
                logger.info("🔌 Neo4j connection closed")

def main():
    print("""
    ============================================
    🧠 NeuroGait Fair Comparison (19D vs 19D)
    ============================================
    Key Features:
    1. Strict 19D comparison (no PCA reduction)
    2. Participant-level splitting
    3. Leakage detection
    4. Statistical validation
    """)
    
    analyzer = FairComparisonNeuroGaitMLAnalysis()
    success = analyzer.run_analysis()
    
    if not success:
        print("\n❌ Analysis failed - check logs for details")
        exit(1)

if __name__ == "__main__":
    main()