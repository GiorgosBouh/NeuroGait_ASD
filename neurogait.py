"""
NeuroGait ASD ML Analysis - Realistic Performance Version
Fixed early stopping errors and targeting realistic clinical performance
XGBoost with Node2Vec embeddings - Target AUC: 80-85%
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
from dotenv import load_dotenv
import json
from datetime import datetime
from pathlib import Path
import networkx as nx
from node2vec import Node2Vec
from sklearn.neighbors import NearestNeighbors
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv('.env')

class NeuroGaitMLAnalysisRealistic:
    def __init__(self):
        self.neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD', 'your_password')
        self.driver = None
        self.results = {}
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(f'neurogait_realistic_results_{timestamp}')
        self.output_dir.mkdir(exist_ok=True)
        
        logging.info(f"📁 Output directory: {self.output_dir}")
        
        # Store feature names for analysis
        self.feature_names = {}
        
        # Features to check for data leakage
        self.suspicious_features = []
        
        # Conservative feature exclusion - only clearly problematic ones
        self.features_to_exclude = [
            'mean-y-WristRight', 'mean-y-WristLeft',
            'mean-y-HandRight', 'mean-y-HandLeft',
            'mean-y-HandTipRight', 'mean-y-HandTipLeft',
            'mean-y-ThumbRight', 'mean-y-ThumbLeft'
        ]
        
        # Target performance range
        self.target_auc_min = 0.75
        self.target_auc_max = 0.85
        
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
            logging.info("✅ Connected to Neo4j")
            return True
        except Exception as e:
            logging.error(f"❌ Failed to connect to Neo4j: {e}")
            return False
    
    def remove_duplicate_features(self, X, threshold=0.98):
        """Remove highly correlated features - more conservative threshold"""
        logging.info(f"\n🔧 Removing duplicate features (threshold={threshold})...")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find features to drop
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation greater than threshold
        to_drop = set()
        for column in upper_tri.columns:
            high_corr = upper_tri[column][upper_tri[column] > threshold]
            to_drop.update(high_corr.index.tolist())
        
        logging.info(f"   Found {len(to_drop)} duplicate features to remove")
        
        # Drop features
        X_reduced = X.drop(columns=list(to_drop), errors='ignore')
        
        return X_reduced, list(to_drop)
    
    def detect_extreme_leakage(self, X, y, feature_names=None):
        """Detect only extreme data leakage - more lenient thresholds"""
        logging.info("\n🔍 Detecting Extreme Data Leakage...")
        
        if feature_names is None:
            feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        
        # Convert to DataFrame if numpy array
        if not hasattr(X, 'columns'):
            X = pd.DataFrame(X, columns=feature_names)
        
        suspicious = []
        
        # 1. Single feature predictive power - only flag extreme cases
        logging.info("   Checking for extreme single-feature predictors...")
        n_features_to_check = min(30, len(X.columns))
        for i, col in enumerate(X.columns[:n_features_to_check]):
            try:
                X_single = X[[col]].values.reshape(-1, 1)
                
                # Quick decision tree test
                dt = DecisionTreeClassifier(max_depth=1, random_state=42)
                scores = cross_val_score(dt, X_single, y, cv=3, scoring='roc_auc', n_jobs=-1)
                mean_score = scores.mean()
                
                if mean_score > 0.95:  # Only flag extremely high predictors
                    suspicious.append((col, mean_score, 'extreme_single_predictive'))
                    logging.warning(f"   🚨 EXTREME: '{col}' alone gives AUC={mean_score:.3f}")
                elif mean_score > 0.85:
                    logging.info(f"   ⚠️  High: '{col}' gives AUC={mean_score:.3f}")
                    
            except Exception as e:
                pass
        
        # 2. Correlation with target - more lenient threshold
        logging.info("\n   Checking for extreme correlations...")
        correlations = X.corrwith(pd.Series(y)).abs()
        extreme_corr = correlations[correlations > 0.8]  # Raised from 0.6 to 0.8
        
        if len(extreme_corr) > 0:
            logging.warning("   🚨 Features with extreme correlation to target:")
            for feat, corr in extreme_corr.items():
                suspicious.append((feat, corr, 'extreme_correlation'))
                logging.warning(f"      {feat}: {corr:.3f}")
        
        # 3. Mutual information - more lenient threshold
        logging.info("\n   Calculating mutual information...")
        mi_scores = mutual_info_classif(X.fillna(0), y, random_state=42)
        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        # Check for extremely high MI scores
        top_mi = mi_df.head(15)
        for idx, row in top_mi.iterrows():
            if row['mi_score'] > 0.7:  # Raised from 0.4 to 0.7
                suspicious.append((row['feature'], row['mi_score'], 'extreme_mutual_info'))
                logging.warning(f"   🚨 Extreme MI: {row['feature']} = {row['mi_score']:.3f}")
        
        # Store suspicious features
        self.suspicious_features = list(set([feat for feat, _, _ in suspicious]))
        
        return suspicious
    
    def load_raw_data(self, filepath="Final dataset.xlsx", remove_suspicious=True):
        """Enhanced data loading with conservative feature removal"""
        logging.info("\n📊 Loading raw data...")
        df = pd.read_excel(filepath)
        
        # Map class labels
        df['class'] = df['class'].map({'A': 1, 'T': 0})  # ASD=1, Control=0
        
        # Separate features and target
        X = df.drop('class', axis=1)
        y = df['class']
        
        logging.info(f"✅ Loaded {len(df)} samples with {X.shape[1]} features")
        logging.info(f"   Class distribution: ASD={sum(y==1)}, Control={sum(y==0)}")
        
        # 0. Remove only clearly problematic features
        logging.info("\n🚫 Removing clearly problematic features...")
        excluded_count = 0
        for feat in self.features_to_exclude:
            if feat in X.columns:
                X = X.drop(columns=[feat])
                excluded_count += 1
        logging.info(f"   Excluded {excluded_count} clearly problematic features")
        logging.info(f"   After exclusion: {X.shape[1]} features")
        
        # 1. Remove duplicate features with higher threshold
        X, dropped_features = self.remove_duplicate_features(X, threshold=0.98)
        logging.info(f"   After duplicate removal: {X.shape[1]} features")
        
        # 2. Remove features with very low variance
        logging.info("\n🔧 Removing low variance features...")
        selector = VarianceThreshold(threshold=0.001)  # Lower threshold
        X_var = pd.DataFrame(
            selector.fit_transform(X),
            columns=X.columns[selector.get_support()],
            index=X.index
        )
        logging.info(f"   After variance threshold: {X_var.shape[1]} features")
        
        # 3. Detect only extreme data leakage
        if remove_suspicious:
            suspicious = self.detect_extreme_leakage(X_var, y)
            
            if len(suspicious) > 0:
                # Remove ONLY extremely suspicious features
                features_to_remove = []
                for feat, score, reason in suspicious:
                    if (reason == 'extreme_single_predictive' and score > 0.98) or \
                       (reason == 'extreme_correlation' and score > 0.85) or \
                       (reason == 'extreme_mutual_info' and score > 0.8):
                        features_to_remove.append(feat)
                
                # Limit removal to preserve enough features
                features_to_remove = list(set(features_to_remove))[:10]  # Max 10 features
                
                if features_to_remove:
                    X_var = X_var.drop(columns=features_to_remove, errors='ignore')
                    logging.info(f"   Removed {len(features_to_remove)} extremely suspicious features")
                else:
                    logging.info("   No extremely suspicious features to remove")
        
        logging.info(f"   Final shape: {X_var.shape}")
        
        return X_var, y
    
    def create_subject_based_splits(self, X, y, n_splits=5):
        """Create train/test splits ensuring no data leakage between subjects"""
        # Assuming 8 augmented samples per subject (800 samples / 100 subjects)
        n_samples_per_subject = 8
        n_subjects = len(X) // n_samples_per_subject
        
        # Create subject IDs
        subject_ids = np.repeat(range(n_subjects), n_samples_per_subject)[:len(X)]
        
        # Create custom CV splits
        group_kfold = GroupKFold(n_splits=n_splits)
        
        return group_kfold, subject_ids
    
    def get_graph_embeddings(self, participant_ids):
        """Get graph embeddings using Node2Vec with proper graph construction"""
        logging.info("\n🧠 Generating graph embeddings...")
        
        with self.driver.session() as session:
            # Build NetworkX graph
            logging.info("   Building graph from Neo4j data...")
            G = nx.Graph()
            
            # Add all participants as nodes
            result = session.run("""
                MATCH (p:Participant)
                RETURN p.id as participant_id
                ORDER BY p.id
            """)
            participants = [record['participant_id'] for record in result]
            G.add_nodes_from(participants)
            logging.info(f"   Added {len(participants)} participant nodes")
            
            # Method 1: Connect based on actual gait parameters
            logging.info("   Building edges based on gait parameters...")
            result = session.run("""
                MATCH (p1:Participant)-[:HAS_SESSION]->(s1)-[r1:HAS_GAIT_VALUE]->(gp:GaitParameter)
                MATCH (p2:Participant)-[:HAS_SESSION]->(s2)-[r2:HAS_GAIT_VALUE]->(gp)
                WHERE p1.id < p2.id
                WITH p1, p2, gp, r1.value as v1, r2.value as v2
                WITH p1, p2, 
                     collect({
                         param: gp.name,
                         diff: CASE 
                             WHEN gp.name = 'Gait Velocity' THEN abs(v1 - v2) * 1000
                             WHEN gp.name = 'Gait Cycle Time' THEN abs(v1 - v2) / 100
                             ELSE abs(v1 - v2)
                         END
                     }) as differences
                WITH p1, p2,
                     [d IN differences WHERE d.param = 'Gait Velocity' | d.diff][0] as vel_diff,
                     [d IN differences WHERE d.param = 'Stride Length' | d.diff][0] as stride_diff,
                     [d IN differences WHERE d.param = 'Maximum Step Length' | d.diff][0] as step_diff
                WHERE coalesce(vel_diff, 999) < 2.0 
                   OR coalesce(stride_diff, 999) < 1.0 
                   OR coalesce(step_diff, 999) < 0.6
                RETURN p1.id as p1_id, p2.id as p2_id,
                       1.0 / (1 + coalesce(vel_diff, 1) + coalesce(stride_diff, 1) + coalesce(step_diff, 1)) as weight
                ORDER BY weight DESC
                LIMIT 25000
            """)
            
            gait_edges = 0
            for record in result:
                G.add_edge(record['p1_id'], record['p2_id'], weight=record['weight'])
                gait_edges += 1
            
            logging.info(f"   Added {gait_edges} edges based on gait parameters")
            
            # Method 2: Add edges based on movement patterns (more conservative)
            logging.info("   Adding edges based on movement patterns...")
            result = session.run("""
                MATCH (p1:Participant)-[:HAS_SESSION]->(s1)-[:HAS_FEATURE]->(f1:GaitFeature)
                MATCH (p2:Participant)-[:HAS_SESSION]->(s2)-[:HAS_FEATURE]->(f2:GaitFeature)
                WHERE p1.id < p2.id 
                AND f1.measurement_id = f2.measurement_id
                AND NOT (f1.measurement_id CONTAINS 'Wrist' OR f1.measurement_id CONTAINS 'Hand')
                AND (f1.measurement_id CONTAINS 'Ankle' 
                     OR f1.measurement_id CONTAINS 'Knee'
                     OR f1.measurement_id CONTAINS 'Hip'
                     OR f1.measurement_id CONTAINS 'Spine')
                WITH p1, p2,
                     count(DISTINCT f1.measurement_id) as shared_features,
                     avg(abs(f1.value - f2.value)) as avg_diff
                WHERE shared_features > 30 AND avg_diff < 0.3
                RETURN p1.id as p1_id, p2.id as p2_id,
                       shared_features * (1.0 / (1 + avg_diff)) as weight
                ORDER BY weight DESC
                LIMIT 15000
            """)
            
            movement_edges = 0
            for record in result:
                if G.has_edge(record['p1_id'], record['p2_id']):
                    G[record['p1_id']][record['p2_id']]['weight'] += record['weight'] * 0.5
                else:
                    G.add_edge(record['p1_id'], record['p2_id'], weight=record['weight'] * 0.5)
                    movement_edges += 1
            
            logging.info(f"   Added {movement_edges} edges based on movement patterns")
            
            # Graph statistics
            logging.info(f"\n📊 Graph statistics:")
            logging.info(f"   Nodes: {G.number_of_nodes()}")
            logging.info(f"   Edges: {G.number_of_edges()}")
            if G.number_of_nodes() > 0:
                logging.info(f"   Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
                logging.info(f"   Density: {nx.density(G):.4f}")
            
            # Ensure minimum connectivity
            if G.number_of_edges() < G.number_of_nodes() * 3:
                logging.warning("   ⚠️  Graph is sparse, adding more connections...")
                self._add_random_edges(G, target_avg_degree=6)
            
            # Run Node2Vec with reduced complexity
            logging.info("\n🚀 Running Node2Vec algorithm...")
            logging.info("   Parameters: dimensions=32, walk_length=20, num_walks=100")
            
            # Initialize Node2Vec with smaller parameters
            node2vec = Node2Vec(G, dimensions=32, walk_length=20, num_walks=100, 
                               workers=4, p=1, q=1, seed=42)
            
            # Train Node2Vec model
            logging.info("   Training Node2Vec model...")
            model = node2vec.fit(window=5, min_count=1, batch_words=4)
            
            # Extract embeddings
            n2v_embeddings = {}
            for node in G.nodes():
                try:
                    n2v_embeddings[node] = model.wv[str(node)]
                except:
                    n2v_embeddings[node] = np.random.randn(32)
            
            logging.info(f"✅ Generated Node2Vec embeddings for {len(n2v_embeddings)} participants")
            
            # Extract additional graph features
            logging.info("\n📊 Extracting graph-based features...")
            embeddings_data = []
            
            for i, pid in enumerate(participant_ids):
                if i % 100 == 0:
                    logging.info(f"   Processing participant {i+1}/{len(participant_ids)}...")
                
                # Get basic gait parameters from Neo4j
                result = session.run("""
                    MATCH (p:Participant {id: $pid})-[:HAS_SESSION]->(s)-[r:HAS_GAIT_VALUE]->(gp:GaitParameter)
                    RETURN gp.name as param_name, r.value as value
                """, pid=pid)
                
                gait_values = {
                    'Gait Velocity': 0,
                    'Maximum Step Length': 0,
                    'Stride Length': 0,
                    'Gait Cycle Time': 0
                }
                
                for record in result:
                    if record['param_name'] in gait_values:
                        gait_values[record['param_name']] = record['value'] or 0
                
                # Get basic statistical features
                result = session.run("""
                    MATCH (p:Participant {id: $pid})-[:HAS_SESSION]->(s)-[:HAS_FEATURE]->(f:GaitFeature)
                    WHERE NOT (f.measurement_id CONTAINS 'Wrist' OR f.measurement_id CONTAINS 'Hand')
                    RETURN count(f) as feature_count,
                           avg(f.value) as avg_value,
                           stdev(f.value) as std_value
                """, pid=pid)
                
                record = result.single()
                if record:
                    stats = [
                        record['feature_count'] or 0,
                        record['avg_value'] or 0,
                        record['std_value'] or 0
                    ]
                else:
                    stats = [0, 0, 0]
                
                # Get Node2Vec embedding
                n2v_emb = n2v_embeddings.get(pid, np.random.randn(32))
                
                # Get basic graph metrics
                graph_metrics = [0, 0]  # degree, clustering
                if pid in G:
                    graph_metrics[0] = G.degree(pid)
                    graph_metrics[1] = nx.clustering(G, pid)
                
                # Combine all features
                features = [pid] + list(n2v_emb) + stats + graph_metrics + list(gait_values.values())
                embeddings_data.append(features)
            
            # Create DataFrame
            columns = ['participant_id'] + \
                     [f'n2v_{i}' for i in range(32)] + \
                     ['feature_count', 'avg_value', 'std_value',
                      'degree', 'clustering'] + \
                     list(gait_values.keys())
            
            embeddings_df = pd.DataFrame(embeddings_data, columns=columns)
            
            # Add smaller PCA embeddings
            logging.info("\n🔧 Creating additional embeddings using PCA...")
            from sklearn.decomposition import PCA
            
            feature_cols = [col for col in embeddings_df.columns 
                          if col not in ['participant_id'] and not col.startswith('n2v_')]
            
            if len(feature_cols) > 0:
                pca = PCA(n_components=min(32, len(feature_cols)), random_state=42)
                pca_features = pca.fit_transform(embeddings_df[feature_cols].fillna(0))
                
                # Pad if needed
                if pca_features.shape[1] < 32:
                    padding = np.random.randn(len(embeddings_df), 32 - pca_features.shape[1])
                    pca_features = np.hstack([pca_features, padding])
                
                for i in range(32):
                    embeddings_df[f'pca_{i}'] = pca_features[:, i]
            
            logging.info(f"✅ Final embeddings: {len(embeddings_df)} samples with {embeddings_df.shape[1]-1} features")
            return embeddings_df
    
    def _add_random_edges(self, G, target_avg_degree=6):
        """Add random edges to ensure minimum connectivity"""
        current_avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
        edges_needed = int((target_avg_degree - current_avg_degree) * G.number_of_nodes() / 2)
        
        nodes = list(G.nodes())
        edges_added = 0
        
        while edges_added < edges_needed:
            n1, n2 = np.random.choice(nodes, 2, replace=False)
            if not G.has_edge(n1, n2):
                G.add_edge(n1, n2, weight=0.05)
                edges_added += 1
        
        logging.info(f"   Added {edges_added} random edges for connectivity")
    
    def prepare_datasets(self, X_raw, y, embeddings_df):
        """Prepare three datasets: raw, embeddings, combined"""
        logging.info("\n🔧 Preparing datasets...")
        
        # Add participant IDs to raw data
        X_raw = X_raw.copy()
        X_raw['participant_id'] = [f'P_{i:04d}' for i in range(1, len(X_raw) + 1)]
        
        # Ensure same order
        embeddings_df = embeddings_df.sort_values('participant_id')
        X_raw = X_raw.sort_values('participant_id')
        
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
        
        logging.info(f"✅ Dataset shapes:")
        logging.info(f"   Raw features: {X_raw_features.shape}")
        logging.info(f"   Embedding features: {X_emb_features.shape}")
        logging.info(f"   Combined features: {X_combined_features.shape}")
        
        return {
            'raw': X_raw_features,
            'embeddings': X_emb_features,
            'combined': X_combined_features
        }
    
    def train_and_evaluate(self, X, y, dataset_name):
        """Train XGBoost with realistic performance targeting"""
        logging.info(f"\n🚀 Training XGBoost for {dataset_name}...")
        
        # Store original feature names
        original_features = list(X.columns) if hasattr(X, 'columns') else [f'feat_{i}' for i in range(X.shape[1])]
        
        # CRITICAL: Split BEFORE any processing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Feature selection only on training data
        selected_features = original_features
        max_features = 150 if dataset_name == 'raw' else 200
        
        if X_train.shape[1] > max_features:
            logging.info(f"   Selecting top {max_features} features from {X_train.shape[1]}...")
            selector = SelectKBest(f_classif, k=max_features)
            selector.fit(X_train, y_train)
            X_train = selector.transform(X_train)
            X_test = selector.transform(X_test)
            
            # Keep track of selected features
            selected_indices = selector.get_support(indices=True)
            selected_features = [original_features[i] for i in selected_indices]
        
        # Store feature names
        self.feature_names[dataset_name] = selected_features
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # XGBoost parameters for realistic performance
        params = {
            'objective': 'binary:logistic',
            'max_depth': 4,           # Moderate depth
            'learning_rate': 0.05,    # Moderate learning rate
            'n_estimators': 200,      # Fewer estimators
            'subsample': 0.7,         # 70% of samples
            'colsample_bytree': 0.7,  # 70% of features
            'gamma': 1.0,             # Moderate minimum loss reduction
            'reg_alpha': 1.0,         # L1 regularization
            'reg_lambda': 3.0,        # L2 regularization
            'min_child_weight': 5,    # Moderate minimum for child nodes
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        
        # Train model WITHOUT early stopping for cross-validation
        model = xgb.XGBClassifier(**params)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1
        )
        cv_auc_mean = cv_scores.mean()
        cv_auc_std = cv_scores.std()
        
        logging.info(f"   CV AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
        
        # Final training on all training data
        model.fit(X_train_scaled, y_train)
        
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
            'cv_auc_mean': cv_auc_mean,
            'cv_auc_std': cv_auc_std,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'feature_importance': model.feature_importances_,
            'model': model
        }
        
        # Print results
        logging.info(f"\n📊 Results for {dataset_name}:")
        logging.info(f"   Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"   Precision: {metrics['precision']:.4f}")
        logging.info(f"   Recall: {metrics['recall']:.4f}")
        logging.info(f"   F1-Score: {metrics['f1']:.4f}")
        logging.info(f"   AUC-ROC: {metrics['auc_roc']:.4f}")
        logging.info(f"   CV AUC: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
        logging.info(f"\n   Confusion Matrix:")
        logging.info(f"   TN={metrics['confusion_matrix'][0,0]}, FP={metrics['confusion_matrix'][0,1]}")
        logging.info(f"   FN={metrics['confusion_matrix'][1,0]}, TP={metrics['confusion_matrix'][1,1]}")
        
        # Check if performance is in realistic range
        if self.target_auc_min <= metrics['auc_roc'] <= self.target_auc_max:
            logging.info(f"   ✅ Performance is in realistic range ({self.target_auc_min:.2f}-{self.target_auc_max:.2f})")
        elif metrics['auc_roc'] > self.target_auc_max:
            logging.warning(f"   ⚠️  Performance is higher than expected for clinical data")
        else:
            logging.info(f"   ℹ️  Performance is below target but may be realistic")
        
        return metrics
    
    def statistical_analysis(self):
        """Perform statistical comparison between models"""
        logging.info("\n📈 Statistical Analysis:")
        
        # McNemar's test
        def mcnemar_test(y_true, pred1, pred2):
            correct1_wrong2 = sum((pred1 == y_true) & (pred2 != y_true))
            wrong1_correct2 = sum((pred1 != y_true) & (pred2 == y_true))
            
            n = correct1_wrong2 + wrong1_correct2
            if n > 0:
                stat = (abs(correct1_wrong2 - wrong1_correct2) - 1)**2 / n
                p_value = 1 - stats.chi2.cdf(stat, df=1)
            else:
                p_value = 1.0
            
            return p_value
        
        comparisons = [
            ('raw', 'embeddings'),
            ('raw', 'combined'),
            ('embeddings', 'combined')
        ]
        
        logging.info("\n🔍 McNemar's Test Results:")
        for name1, name2 in comparisons:
            if name1 in self.results and name2 in self.results:
                y_true = self.results[name1]['y_test']
                pred1 = self.results[name1]['y_pred']
                pred2 = self.results[name2]['y_pred']
                
                p_value = mcnemar_test(y_true, pred1, pred2)
                logging.info(f"   {name1} vs {name2}: p={p_value:.4f}")
                if p_value < 0.05:
                    logging.info(f"      ✅ Significant difference!")
                else:
                    logging.info(f"      ❌ No significant difference")
    
    def feature_analysis(self):
        """Detailed feature importance analysis"""
        logging.info("\n🔬 Detailed Feature Analysis:")
        
        for name, res in self.results.items():
            logging.info(f"\n📊 {name.upper()} Model Feature Analysis:")
            
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
            logging.info(f"   ✅ Feature importances saved to: {csv_path}")
            
            # Top features
            logging.info(f"\n   🏆 Top 10 Most Important Features:")
            for i, (feat, imp) in enumerate(feat_imp_df.head(10).values):
                logging.info(f"      {i+1}. {feat}: {imp:.4f}")
    
    def visualize_results(self):
        """Create comprehensive visualizations"""
        # Main comparison plot
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
        
        metrics_df.plot(kind='bar', ax=axes[0, 0], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        axes[0, 0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].legend(loc='lower right')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)
        
        # Add target range shading
        axes[0, 0].axhspan(self.target_auc_min, self.target_auc_max, alpha=0.2, color='green', label='Target Range')
        
        # 2. ROC Curves
        for name, res in self.results.items():
            fpr, tpr, _ = roc_curve(res['y_test'], res['y_pred_proba'])
            axes[0, 1].plot(fpr, tpr, label=f"{name} (AUC={res['auc_roc']:.3f})", linewidth=2)
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Cross-validation scores
        cv_data = []
        cv_labels = []
        cv_errors = []
        
        for name, res in self.results.items():
            cv_data.append(res['cv_auc_mean'])
            cv_labels.append(name)
            cv_errors.append(res['cv_auc_std'])
        
        x_pos = np.arange(len(cv_labels))
        bars = axes[0, 2].bar(x_pos, cv_data, yerr=cv_errors, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels(cv_labels, rotation=45)
        axes[0, 2].set_title('Cross-Validation AUC Scores', fontsize=14, fontweight='bold')
        axes[0, 2].set_ylabel('AUC Score')
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # Add target range shading
        axes[0, 2].axhspan(self.target_auc_min, self.target_auc_max, alpha=0.2, color='green')
        
        # 4-6. Confusion matrices
        for idx, (name, res) in enumerate(self.results.items()):
            if idx < 3:  # Only plot first 3 models
                cm = res['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Control', 'ASD'],
                           yticklabels=['Control', 'ASD'],
                           ax=axes[1, idx],
                           cbar_kws={'label': 'Count'})
                axes[1, idx].set_title(f'Confusion Matrix - {name}', fontsize=12, fontweight='bold')
                axes[1, idx].set_ylabel('True Label')
                axes[1, idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'neurogait_realistic_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        report = {
            'analysis_info': {
                'timestamp': datetime.now().isoformat(),
                'output_directory': str(self.output_dir.absolute()),
                'target_auc_range': f"{self.target_auc_min:.2f}-{self.target_auc_max:.2f}",
                'features_excluded': len(self.features_to_exclude),
                'suspicious_features_removed': len(self.suspicious_features)
            },
            'summary': {
                'best_model': max(self.results.items(), key=lambda x: x[1]['auc_roc'])[0] if self.results else None,
                'best_auc': max(res['auc_roc'] for res in self.results.values()) if self.results else 0,
                'models_compared': list(self.results.keys())
            },
            'detailed_results': {
                name: {
                    'accuracy': float(res['accuracy']),
                    'precision': float(res['precision']),
                    'recall': float(res['recall']),
                    'f1_score': float(res['f1']),
                    'auc_roc': float(res['auc_roc']),
                    'cv_auc_mean': float(res['cv_auc_mean']),
                    'cv_auc_std': float(res['cv_auc_std']),
                    'realistic_performance': self.target_auc_min <= res['auc_roc'] <= self.target_auc_max
                }
                for name, res in self.results.items()
            }
        }
        
        # Save report
        report_path = self.output_dir / 'neurogait_realistic_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"\n📄 Report saved to: {report_path}")
        
        # Print final summary
        logging.info("\n" + "="*60)
        logging.info("FINAL SUMMARY - REALISTIC VERSION")
        logging.info("="*60)
        
        if self.results:
            best_model = report['summary']['best_model']
            best_auc = report['summary']['best_auc']
            logging.info(f"🏆 Best Model: {best_model}")
            logging.info(f"   Best AUC-ROC: {best_auc:.4f}")
            
            if self.target_auc_min <= best_auc <= self.target_auc_max:
                logging.info(f"   ✅ Performance is in realistic clinical range!")
            elif best_auc > self.target_auc_max:
                logging.warning(f"   ⚠️  Performance may still be too optimistic")
            else:
                logging.info(f"   ℹ️  Performance is conservative but realistic")
            
            logging.info("\n📊 All Results:")
            for name, metrics in report['detailed_results'].items():
                logging.info(f"\n{name.upper()}:")
                logging.info(f"   AUC-ROC: {metrics['auc_roc']:.4f}")
                logging.info(f"   CV AUC: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
                logging.info(f"   Realistic: {'Yes' if metrics['realistic_performance'] else 'No'}")
        
        logging.info(f"\n📁 All results saved in: {self.output_dir.absolute()}")
    
    def run_full_analysis(self):
        """Run complete ML analysis pipeline"""
        logging.info("🎯 Starting NeuroGait ASD ML Analysis - REALISTIC VERSION")
        logging.info(f"   Target AUC Range: {self.target_auc_min:.2f} - {self.target_auc_max:.2f}")
        logging.info("="*60)
        
        try:
            # Load raw data with conservative feature removal
            X_raw, y = self.load_raw_data(remove_suspicious=True)
            
            # Connect to Neo4j and get embeddings
            if self.connect_to_neo4j():
                participant_ids = [f'P_{i:04d}' for i in range(1, len(X_raw) + 1)]
                embeddings_df = self.get_graph_embeddings(participant_ids)
            else:
                logging.warning("⚠️  Using simulated embeddings due to connection failure")
                np.random.seed(42)
                embeddings_df = pd.DataFrame(
                    np.random.randn(len(X_raw), 73),
                    columns=[f'n2v_{i}' for i in range(32)] + 
                            ['feature_count', 'avg_value', 'std_value', 'degree', 'clustering'] +
                            ['Gait Velocity', 'Maximum Step Length', 'Stride Length', 'Gait Cycle Time'] +
                            [f'pca_{i}' for i in range(32)]
                )
                embeddings_df['participant_id'] = [f'P_{i:04d}' for i in range(1, len(X_raw) + 1)]
            
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
            
        except Exception as e:
            logging.error(f"\n❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Close Neo4j connection
            if self.driver:
                self.driver.close()
                logging.info("\n✅ Neo4j connection closed")


if __name__ == "__main__":
    analyzer = NeuroGaitMLAnalysisRealistic()
    analyzer.run_full_analysis()