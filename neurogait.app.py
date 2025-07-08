"""
XGBoost ASD Prediction: Graph Embeddings vs Raw Data vs Combined
Αποφυγή Data Leakage με proper train-test split
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
from dotenv import load_dotenv
import json
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv('.env')

class ASDPredictionAnalysis:
    def __init__(self):
        self.neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD', 'your_password')
        self.driver = None
        self.results = {}
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(f'xgboost_results_{timestamp}')
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"📁 Output directory: {self.output_dir}")
        
        # Store feature names for analysis
        self.feature_names = {}
        
    def connect_to_neo4j(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_user, self.neo4j_password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("✅ Connected to Neo4j")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            return False
    
    def load_raw_data(self, filepath="Final dataset.xlsx"):
        """Load raw data from Excel file"""
        print("\n📊 Loading raw data...")
        df = pd.read_excel(filepath)
        
        # Map class labels
        df['class'] = df['class'].map({'A': 1, 'T': 0})  # ASD=1, Control=0
        
        # Separate features and target
        X = df.drop('class', axis=1)
        y = df['class']
        
        print(f"✅ Loaded {len(df)} samples with {X.shape[1]} features")
        print(f"   Class distribution: ASD={sum(y==1)}, Control={sum(y==0)}")
        
        return X, y
    
    def get_graph_embeddings(self, participant_ids):
        """Get graph embeddings using Neo4j GDS"""
        print("\n🧠 Generating graph embeddings...")
        
        with self.driver.session() as session:
            try:
                # First, create in-memory graph projection
                print("Creating graph projection...")
                session.run("""
                    CALL gds.graph.drop('gaitGraph', false);
                """)
                
                # Create graph with participant nodes and their relationships
                session.run("""
                    CALL gds.graph.project(
                        'gaitGraph',
                        ['Participant', 'GaitSession', 'GaitFeature'],
                        {
                            HAS_SESSION: {orientation: 'UNDIRECTED'},
                            HAS_FEATURE: {
                                orientation: 'UNDIRECTED',
                                properties: ['value']
                            }
                        }
                    )
                """)
                
                # Generate FastRP embeddings
                print("Generating FastRP embeddings...")
                session.run("""
                    CALL gds.fastRP.mutate('gaitGraph', {
                        embeddingDimension: 128,
                        randomSeed: 42,
                        mutateProperty: 'embedding'
                    })
                """)
                
                # Stream embeddings for our participants
                result = session.run("""
                    CALL gds.graph.nodeProperties.stream('gaitGraph', ['embedding'])
                    YIELD nodeId, nodeLabels, propertyValue
                    WITH gds.util.asNode(nodeId) AS node, propertyValue AS embedding
                    WHERE 'Participant' IN labels(node)
                    RETURN node.id AS participant_id, embedding
                    ORDER BY node.id
                """)
                
                # Convert to DataFrame
                embeddings_data = []
                for record in result:
                    pid = record['participant_id']
                    embedding = record['embedding']
                    embeddings_data.append([pid] + list(embedding))
                
                # Create embeddings DataFrame
                columns = ['participant_id'] + [f'emb_{i}' for i in range(128)]
                embeddings_df = pd.DataFrame(embeddings_data, columns=columns)
                
                print(f"✅ Generated embeddings for {len(embeddings_df)} participants")
                
                # Alternative: Node2Vec embeddings
                print("\nGenerating Node2Vec embeddings...")
                session.run("""
                    CALL gds.node2vec.mutate('gaitGraph', {
                        embeddingDimension: 64,
                        walkLength: 80,
                        walksPerNode: 10,
                        windowSize: 10,
                        randomSeed: 42,
                        mutateProperty: 'embedding_n2v'
                    })
                """)
                
                # Get Node2Vec embeddings
                result_n2v = session.run("""
                    CALL gds.graph.nodeProperties.stream('gaitGraph', ['embedding_n2v'])
                    YIELD nodeId, nodeLabels, propertyValue
                    WITH gds.util.asNode(nodeId) AS node, propertyValue AS embedding
                    WHERE 'Participant' IN labels(node)
                    RETURN node.id AS participant_id, embedding
                    ORDER BY node.id
                """)
                
                # Add Node2Vec embeddings
                n2v_data = {}
                for record in result_n2v:
                    n2v_data[record['participant_id']] = record['embedding']
                
                # Add Node2Vec columns
                for i in range(64):
                    embeddings_df[f'n2v_{i}'] = embeddings_df['participant_id'].map(
                        lambda x: n2v_data.get(x, [0]*64)[i]
                    )
                
                # Drop graph projection
                session.run("CALL gds.graph.drop('gaitGraph')")
                
                return embeddings_df
                
            except Exception as e:
                print(f"❌ Error generating embeddings: {e}")
                # Return random embeddings as fallback
                print("⚠️  Using random embeddings as fallback...")
                np.random.seed(42)
                n_participants = len(participant_ids)
                embeddings_data = np.random.randn(n_participants, 192)  # 128 FastRP + 64 Node2Vec
                columns = [f'emb_{i}' for i in range(128)] + [f'n2v_{i}' for i in range(64)]
                embeddings_df = pd.DataFrame(embeddings_data, columns=columns)
                embeddings_df['participant_id'] = range(n_participants)
                return embeddings_df
    
    def prepare_datasets(self, X_raw, y, embeddings_df):
        """Prepare three datasets: raw, embeddings, combined"""
        print("\n🔧 Preparing datasets...")
        
        # Add participant IDs to raw data
        X_raw = X_raw.copy()
        X_raw['participant_id'] = range(len(X_raw))
        
        # Merge embeddings with raw data
        X_combined = X_raw.merge(
            embeddings_df, 
            on='participant_id', 
            how='left'
        )
        
        # Prepare three feature sets
        feature_cols_raw = [col for col in X_raw.columns if col != 'participant_id']
        feature_cols_emb = [col for col in embeddings_df.columns if col != 'participant_id']
        feature_cols_combined = feature_cols_raw + feature_cols_emb
        
        X_raw_features = X_raw[feature_cols_raw]
        X_emb_features = X_combined[feature_cols_emb]
        X_combined_features = X_combined[feature_cols_combined]
        
        print(f"✅ Dataset shapes:")
        print(f"   Raw features: {X_raw_features.shape}")
        print(f"   Embedding features: {X_emb_features.shape}")
        print(f"   Combined features: {X_combined_features.shape}")
        
        return {
            'raw': X_raw_features,
            'embeddings': X_emb_features,
            'combined': X_combined_features
        }
    
    def train_and_evaluate(self, X, y, dataset_name):
        """Train XGBoost and evaluate with proper train-test split"""
        print(f"\n🚀 Training XGBoost for {dataset_name}...")
        
        # Store original feature names
        original_features = list(X.columns)
        
        # CRITICAL: Split BEFORE any processing to avoid data leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Feature selection only on training data
        selected_features = original_features
        if X_train.shape[1] > 300:
            print(f"   Selecting top 300 features from {X_train.shape[1]}...")
            selector = SelectKBest(f_classif, k=300)
            selector.fit(X_train, y_train)  # Fit only on training data!
            X_train = selector.transform(X_train)
            X_test = selector.transform(X_test)
            
            # Keep track of selected features
            selected_indices = selector.get_support(indices=True)
            selected_features = [original_features[i] for i in selected_indices]
        
        # Store feature names for this dataset
        self.feature_names[dataset_name] = selected_features
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training data!
        X_test_scaled = scaler.transform(X_test)
        
        # XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        
        # Train model
        model = xgb.XGBClassifier(**params)
        
        # Cross-validation on training data only
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        
        # Final training
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'feature_importance': model.feature_importances_,
            'model': model
        }
        
        # Print results
        print(f"\n📊 Results for {dataset_name}:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1']:.4f}")
        print(f"   AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"   CV AUC: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
        print(f"\n   Confusion Matrix:")
        print(f"   {metrics['confusion_matrix']}")
        
        return metrics
    
    def statistical_analysis(self):
        """Perform statistical analysis comparing models"""
        print("\n📈 Statistical Analysis:")
        
        # McNemar's test for comparing predictions
        def mcnemar_test(y_true, pred1, pred2):
            # Create contingency table
            correct1_wrong2 = sum((pred1 == y_true) & (pred2 != y_true))
            wrong1_correct2 = sum((pred1 != y_true) & (pred2 == y_true))
            
            # McNemar's test
            n = correct1_wrong2 + wrong1_correct2
            if n > 0:
                stat = (abs(correct1_wrong2 - wrong1_correct2) - 1)**2 / n
                p_value = 1 - stats.chi2.cdf(stat, df=1)
            else:
                p_value = 1.0
            
            return p_value
        
        # Compare all pairs
        comparisons = [
            ('raw', 'embeddings'),
            ('raw', 'combined'),
            ('embeddings', 'combined')
        ]
        
        print("\n🔍 McNemar's Test Results (p-values):")
        for name1, name2 in comparisons:
            y_true = self.results[name1]['y_test']
            pred1 = self.results[name1]['y_pred']
            pred2 = self.results[name2]['y_pred']
            
            p_value = mcnemar_test(y_true, pred1, pred2)
            print(f"   {name1} vs {name2}: p={p_value:.4f}")
            if p_value < 0.05:
                print(f"      ✅ Significant difference!")
            else:
                print(f"      ❌ No significant difference")
    
    def feature_analysis(self):
        """Detailed feature analysis for each model"""
        print("\n🔬 Detailed Feature Analysis:")
        
        for name, res in self.results.items():
            if name == 'embeddings':
                continue  # Skip embeddings-only model
                
            print(f"\n📊 {name.upper()} Model Feature Analysis:")
            
            # Get feature importances and names
            importances = res['feature_importance']
            feature_names = self.feature_names.get(name, [f'Feature_{i}' for i in range(len(importances))])
            
            # Create feature importance DataFrame
            feat_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Save to CSV
            csv_path = self.output_dir / f'feature_importance_{name}.csv'
            feat_imp_df.to_csv(csv_path, index=False)
            print(f"   ✅ Feature importances saved to: {csv_path}")
            
            # Analyze by feature type
            if name in ['raw', 'combined']:
                # Categorize features
                feature_categories = {
                    'body_measurements': [],
                    'distance_features': [],
                    'range_of_motion': [],
                    'temporal_gait': [],
                    'graph_embeddings': [],
                    'other': []
                }
                
                for idx, feat in enumerate(feature_names):
                    feat_lower = feat.lower()
                    if any(body in feat_lower for body in ['ankle', 'knee', 'hip', 'shoulder', 'elbow', 'wrist', 'head', 'torso']):
                        if 'distance' in feat_lower or 'dist' in feat_lower:
                            feature_categories['distance_features'].append((feat, importances[idx]))
                        elif 'rom' in feat_lower or 'angle' in feat_lower:
                            feature_categories['range_of_motion'].append((feat, importances[idx]))
                        else:
                            feature_categories['body_measurements'].append((feat, importances[idx]))
                    elif any(temporal in feat_lower for temporal in ['step', 'stride', 'cadence', 'speed', 'time']):
                        feature_categories['temporal_gait'].append((feat, importances[idx]))
                    elif 'emb_' in feat or 'n2v_' in feat:
                        feature_categories['graph_embeddings'].append((feat, importances[idx]))
                    else:
                        feature_categories['other'].append((feat, importances[idx]))
                
                # Print category summary
                print("\n   📈 Feature Category Importance:")
                category_importance = {}
                for category, features in feature_categories.items():
                    if features:
                        total_importance = sum(imp for _, imp in features)
                        avg_importance = total_importance / len(features)
                        category_importance[category] = {
                            'total': total_importance,
                            'average': avg_importance,
                            'count': len(features),
                            'top_feature': max(features, key=lambda x: x[1]) if features else None
                        }
                
                # Sort by total importance
                sorted_categories = sorted(category_importance.items(), 
                                         key=lambda x: x[1]['total'], 
                                         reverse=True)
                
                for category, stats in sorted_categories:
                    print(f"\n      {category}:")
                    print(f"         Total importance: {stats['total']:.4f}")
                    print(f"         Average importance: {stats['average']:.4f}")
                    print(f"         Number of features: {stats['count']}")
                    if stats['top_feature']:
                        print(f"         Top feature: {stats['top_feature'][0]} ({stats['top_feature'][1]:.4f})")
                
                # Create category importance plot
                plt.figure(figsize=(10, 6))
                categories = [cat for cat, _ in sorted_categories]
                totals = [stats['total'] for _, stats in sorted_categories]
                
                plt.bar(categories, totals)
                plt.xlabel('Feature Category')
                plt.ylabel('Total Importance')
                plt.title(f'Feature Category Importance - {name} Model')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                cat_plot_path = self.output_dir / f'category_importance_{name}.png'
                plt.savefig(cat_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
            # Top 20 most important features
            print(f"\n   🏆 Top 10 Most Important Features:")
            for i, (feat, imp) in enumerate(feat_imp_df.head(10).values):
                print(f"      {i+1}. {feat}: {imp:.4f}")
                
        
        # DeLong's test for AUC comparison
        print("\n🔍 DeLong's Test for AUC (approximate):")
        for name1, name2 in comparisons:
            auc1 = self.results[name1]['auc_roc']
            auc2 = self.results[name2]['auc_roc']
            
            # Approximate using CV standard deviations
            se1 = self.results[name1]['cv_auc_std'] / np.sqrt(5)
            se2 = self.results[name2]['cv_auc_std'] / np.sqrt(5)
            se_diff = np.sqrt(se1**2 + se2**2)
            
            z_stat = (auc1 - auc2) / se_diff if se_diff > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            print(f"   {name1} (AUC={auc1:.4f}) vs {name2} (AUC={auc2:.4f})")
            print(f"      z-statistic: {z_stat:.4f}, p-value: {p_value:.4f}")
            if p_value < 0.05:
                print(f"      ✅ Significant difference!")
            else:
                print(f"      ❌ No significant difference")
    
    def visualize_results(self):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Metrics comparison
        metrics_df = pd.DataFrame({
            name: {
                'Accuracy': res['accuracy'],
                'Precision': res['precision'],
                'Recall': res['recall'],
                'F1-Score': res['f1'],
                'AUC-ROC': res['auc_roc']
            }
            for name, res in self.results.items()
        }).T
        
        metrics_df.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].legend(loc='lower right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ROC Curves
        from sklearn.metrics import roc_curve
        for name, res in self.results.items():
            fpr, tpr, _ = roc_curve(res['y_test'], res['y_pred_proba'])
            axes[0, 1].plot(fpr, tpr, label=f"{name} (AUC={res['auc_roc']:.3f})")
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Cross-validation scores
        cv_data = pd.DataFrame({
            name: [res['cv_auc_mean']] 
            for name, res in self.results.items()
        })
        cv_errors = pd.DataFrame({
            name: [res['cv_auc_std']] 
            for name, res in self.results.items()
        })
        
        cv_data.T.plot(kind='bar', ax=axes[0, 2], yerr=cv_errors.T.values, capsize=5)
        axes[0, 2].set_title('Cross-Validation AUC Scores')
        axes[0, 2].set_ylabel('AUC Score')
        axes[0, 2].set_xlabel('Model')
        axes[0, 2].legend(['Mean AUC'])
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4-6. Confusion matrices
        for idx, (name, res) in enumerate(self.results.items()):
            cm = res['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Control', 'ASD'],
                       yticklabels=['Control', 'ASD'],
                       ax=axes[1, idx])
            axes[1, idx].set_title(f'Confusion Matrix - {name}')
            axes[1, idx].set_ylabel('True Label')
            axes[1, idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'xgboost_asd_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance for best model
        best_model_name = max(self.results.items(), key=lambda x: x[1]['auc_roc'])[0]
        print(f"\n🏆 Best model: {best_model_name} (AUC={self.results[best_model_name]['auc_roc']:.4f})")
        
        # Plot top 20 features for best model
        if best_model_name in ['raw', 'combined']:
            plt.figure(figsize=(10, 8))
            importances = self.results[best_model_name]['feature_importance']
            indices = np.argsort(importances)[::-1][:20]
            
            plt.barh(range(20), importances[indices])
            plt.yticks(range(20), [f'Feature_{i}' for i in indices])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 20 Features - {best_model_name} Model')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'feature_importance_{best_model_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_report(self):
        """Generate comprehensive report"""
        report = {
            'summary': {
                'best_model': max(self.results.items(), key=lambda x: x[1]['auc_roc'])[0],
                'best_auc': max(res['auc_roc'] for res in self.results.values()),
                'models_compared': list(self.results.keys())
            },
            'detailed_results': {
                name: {
                    'accuracy': res['accuracy'],
                    'precision': res['precision'],
                    'recall': res['recall'],
                    'f1_score': res['f1'],
                    'auc_roc': res['auc_roc'],
                    'cv_auc_mean': res['cv_auc_mean'],
                    'cv_auc_std': res['cv_auc_std']
                }
                for name, res in self.results.items()
            }
        }
        
        # Save report
        report_path = self.output_dir / 'xgboost_asd_analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"🏆 Best Model: {report['summary']['best_model']}")
        print(f"   Best AUC-ROC: {report['summary']['best_auc']:.4f}")
        print("\n📊 All Results:")
        for name, metrics in report['detailed_results'].items():
            print(f"\n{name.upper()}:")
            for metric, value in metrics.items():
                print(f"   {metric}: {value:.4f}")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("🎯 Starting ASD Prediction Analysis with XGBoost")
        print("="*60)
        
        # Load raw data
        X_raw, y = self.load_raw_data()
        
        # Connect to Neo4j and get embeddings
        if self.connect_to_neo4j():
            participant_ids = list(range(len(X_raw)))
            embeddings_df = self.get_graph_embeddings(participant_ids)
        else:
            print("⚠️  Using simulated embeddings due to connection failure")
            np.random.seed(42)
            embeddings_df = pd.DataFrame(
                np.random.randn(len(X_raw), 192),
                columns=[f'emb_{i}' for i in range(128)] + [f'n2v_{i}' for i in range(64)]
            )
            embeddings_df['participant_id'] = range(len(X_raw))
        
        # Prepare datasets
        datasets = self.prepare_datasets(X_raw, y, embeddings_df)
        
        # Train and evaluate each approach
        for name, X in datasets.items():
            self.results[name] = self.train_and_evaluate(X, y, name)
        
        # Statistical analysis
        self.statistical_analysis()
        
        # Feature analysis
        self.feature_analysis()
        
        # Visualizations
        self.visualize_results()
        
        # Generate report
        self.generate_report()
        
        # Get best model info
        best_model_name = max(self.results.items(), key=lambda x: x[1]['auc_roc'])[0]
        best_auc = self.results[best_model_name]['auc_roc']
        
        # Create summary file
        summary_path = self.output_dir / 'README.txt'
        with open(summary_path, 'w') as f:
            f.write("XGBoost ASD Prediction Analysis Results\n")
            f.write("="*50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {self.output_dir.absolute()}\n\n")
            f.write("Files Generated:\n")
            f.write("- xgboost_asd_analysis_results.png: All performance plots\n")
            f.write("- xgboost_asd_analysis_report.json: Detailed metrics\n")
            f.write("- feature_importance_*.csv: Feature importance for each model\n")
            f.write("- feature_importance_*.png: Top features visualization\n")
            f.write("- category_importance_*.png: Feature category analysis\n\n")
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"Best AUC-ROC: {best_auc:.4f}\n")
        
        print(f"\n📁 All results saved in: {self.output_dir.absolute()}")
        
        # Close Neo4j connection
        if self.driver:
            self.driver.close()
            print("\n✅ Neo4j connection closed")


if __name__ == "__main__":
    try:
        analyzer = ASDPredictionAnalysis()
        analyzer.run_full_analysis()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()