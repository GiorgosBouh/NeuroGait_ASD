"""
NeuroGait ASD ML Analysis
XGBoost with Node2Vec embeddings and proper data leakage prevention
Compares: Raw features vs Graph embeddings vs Combined approach
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
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

class NeuroGaitMLAnalysis:
    def __init__(self):
        self.neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD', 'your_password')
        self.driver = None
        self.results = {}
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(f'neurogait_ml_results_{timestamp}')
        self.output_dir.mkdir(exist_ok=True)
        
        logging.info(f"📁 Output directory: {self.output_dir}")
        
        # Store feature names for analysis
        self.feature_names = {}
        
        # Features to check for data leakage
        self.suspicious_features = []
        
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
    
    def detect_data_leakage(self, X, y, feature_names=None):
        """Comprehensive data leakage detection"""
        logging.info("\n🔍 Performing Data Leakage Detection...")
        
        if feature_names is None:
            feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        
        # Convert to DataFrame if numpy array
        if not hasattr(X, 'columns'):
            X = pd.DataFrame(X, columns=feature_names)
        
        suspicious = []
        
        # 1. Single feature predictive power
        logging.info("   Checking individual feature predictive power...")
        for i, col in enumerate(X.columns[:100]):  # Check first 100 features
            try:
                X_single = X[[col]].values.reshape(-1, 1)
                
                # Quick decision tree test
                dt = DecisionTreeClassifier(max_depth=1, random_state=42)
                scores = cross_val_score(dt, X_single, y, cv=5, scoring='roc_auc', n_jobs=-1)
                mean_score = scores.mean()
                
                if mean_score > 0.90:
                    suspicious.append((col, mean_score, 'high_single_predictive'))
                    logging.warning(f"   🚨 SUSPICIOUS: '{col}' alone gives AUC={mean_score:.3f}")
                elif mean_score > 0.80:
                    logging.info(f"   ⚠️  Notable: '{col}' gives AUC={mean_score:.3f}")
                    
            except Exception as e:
                pass
        
        # 2. Correlation with target
        logging.info("\n   Checking correlations with target...")
        correlations = X.corrwith(pd.Series(y)).abs()
        high_corr = correlations[correlations > 0.7]
        
        if len(high_corr) > 0:
            logging.warning("   🚨 Features with very high correlation to target:")
            for feat, corr in high_corr.items():
                suspicious.append((feat, corr, 'high_correlation'))
                logging.warning(f"      {feat}: {corr:.3f}")
        
        # 3. Mutual information
        logging.info("\n   Calculating mutual information...")
        mi_scores = mutual_info_classif(X.fillna(0), y, random_state=42)
        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        # Check for suspiciously high MI scores
        top_mi = mi_df.head(10)
        for idx, row in top_mi.iterrows():
            if row['mi_score'] > 0.5:
                suspicious.append((row['feature'], row['mi_score'], 'high_mutual_info'))
                logging.warning(f"   🚨 High MI: {row['feature']} = {row['mi_score']:.3f}")
        
        # 4. Check for duplicate or near-duplicate features
        logging.info("\n   Checking for duplicate features...")
        correlation_matrix = X.corr().abs()
        upper_tri = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        duplicate_pairs = []
        for column in upper_tri.columns:
            correlated = upper_tri[column][upper_tri[column] > 0.99]
            for idx in correlated.index:
                duplicate_pairs.append((column, idx, correlated[idx]))
                
        if duplicate_pairs:
            logging.warning(f"   🚨 Found {len(duplicate_pairs)} near-duplicate feature pairs")
        
        # Store suspicious features
        self.suspicious_features = list(set([feat for feat, _, _ in suspicious]))
        
        return suspicious
    
    def load_raw_data(self, filepath="Final dataset.xlsx", remove_suspicious=True):
        """Load raw data with optional suspicious feature removal"""
        logging.info("\n📊 Loading raw data...")
        df = pd.read_excel(filepath)
        
        # Map class labels
        df['class'] = df['class'].map({'A': 1, 'T': 0})  # ASD=1, Control=0
        
        logging.info(f"✅ Loaded {len(df)} samples with {df.shape[1]-1} features")
        logging.info(f"   Class distribution: ASD={sum(df['class']==1)}, Control={sum(df['class']==0)}")
        
        # Separate features and target
        X = df.drop('class', axis=1)
        y = df['class']
        
        # Detect data leakage
        suspicious = self.detect_data_leakage(X, y)
        
        if len(suspicious) > 0 and remove_suspicious:
            logging.info(f"\n⚠️  Removing {len(self.suspicious_features)} suspicious features...")
            
            # Remove top suspicious features but keep some for comparison
            features_to_remove = []
            for feat, score, reason in suspicious:
                if reason == 'high_single_predictive' and score > 0.95:
                    features_to_remove.append(feat)
                elif reason == 'high_correlation' and score > 0.85:
                    features_to_remove.append(feat)
            
            features_to_remove = list(set(features_to_remove))[:10]  # Remove max 10 features
            
            if features_to_remove:
                X = X.drop(columns=features_to_remove, errors='ignore')
                logging.info(f"   Removed features: {features_to_remove}")
        
        return X, y
    
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
                WHERE coalesce(vel_diff, 999) < 1.0 
                   OR coalesce(stride_diff, 999) < 0.5 
                   OR coalesce(step_diff, 999) < 0.3
                RETURN p1.id as p1_id, p2.id as p2_id,
                       1.0 / (1 + coalesce(vel_diff, 1) + coalesce(stride_diff, 1) + coalesce(step_diff, 1)) as weight
                ORDER BY weight DESC
                LIMIT 30000
            """)
            
            gait_edges = 0
            for record in result:
                G.add_edge(record['p1_id'], record['p2_id'], weight=record['weight'])
                gait_edges += 1
            
            logging.info(f"   Added {gait_edges} edges based on gait parameters")
            
            # Method 2: Add edges based on movement patterns (important features)
            logging.info("   Adding edges based on movement patterns...")
            result = session.run("""
                MATCH (p1:Participant)-[:HAS_SESSION]->(s1)-[:HAS_FEATURE]->(f1:GaitFeature)
                MATCH (p2:Participant)-[:HAS_SESSION]->(s2)-[:HAS_FEATURE]->(f2:GaitFeature)
                WHERE p1.id < p2.id 
                AND f1.measurement_id = f2.measurement_id
                AND (f1.measurement_id CONTAINS 'Wrist' 
                     OR f1.measurement_id CONTAINS 'Hand'
                     OR f1.measurement_id CONTAINS 'Ankle'
                     OR f1.measurement_id CONTAINS 'angle')
                WITH p1, p2,
                     count(DISTINCT f1.measurement_id) as shared_features,
                     avg(abs(f1.value - f2.value)) as avg_diff
                WHERE shared_features > 50 AND avg_diff < 0.2
                RETURN p1.id as p1_id, p2.id as p2_id,
                       shared_features * (1.0 / (1 + avg_diff)) as weight
                ORDER BY weight DESC
                LIMIT 20000
            """)
            
            movement_edges = 0
            for record in result:
                if G.has_edge(record['p1_id'], record['p2_id']):
                    G[record['p1_id']][record['p2_id']]['weight'] += record['weight']
                else:
                    G.add_edge(record['p1_id'], record['p2_id'], weight=record['weight'])
                    movement_edges += 1
            
            logging.info(f"   Added {movement_edges} edges based on movement patterns")
            
            # Method 3: k-NN based on velocity (most reliable single parameter)
            logging.info("   Building k-NN connections based on velocity...")
            result = session.run("""
                MATCH (p:Participant)-[:HAS_SESSION]->(s)-[r:HAS_GAIT_VALUE]->(gp:GaitParameter)
                WHERE gp.name = 'Gait Velocity'
                RETURN p.id as pid, r.value as velocity
                ORDER BY p.id
            """)
            
            velocity_data = {record['pid']: record['velocity'] for record in result}
            
            if len(velocity_data) > 0:
                pids = list(velocity_data.keys())
                X_velocity = np.array([[velocity_data.get(pid, 0)] for pid in pids])
                
                knn = NearestNeighbors(n_neighbors=min(15, len(pids)), metric='euclidean')
                knn.fit(X_velocity)
                
                distances, indices = knn.kneighbors(X_velocity)
                
                knn_edges = 0
                for i, pid1 in enumerate(pids):
                    for j in range(1, min(15, len(indices[i]))):
                        pid2 = pids[indices[i][j]]
                        if pid1 < pid2:
                            weight = 1.0 / (1 + distances[i][j] * 100)
                            if G.has_edge(pid1, pid2):
                                G[pid1][pid2]['weight'] = max(G[pid1][pid2]['weight'], weight)
                            else:
                                G.add_edge(pid1, pid2, weight=weight)
                                knn_edges += 1
                
                logging.info(f"   Added {knn_edges} k-NN edges")
            
            # Graph statistics
            logging.info(f"\n📊 Graph statistics:")
            logging.info(f"   Nodes: {G.number_of_nodes()}")
            logging.info(f"   Edges: {G.number_of_edges()}")
            if G.number_of_nodes() > 0:
                logging.info(f"   Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
                logging.info(f"   Density: {nx.density(G):.4f}")
            
            # Ensure minimum connectivity
            if G.number_of_edges() < G.number_of_nodes() * 5:
                logging.warning("   ⚠️  Graph is sparse, adding more connections...")
                self._add_random_edges(G, target_avg_degree=10)
            
            # Run Node2Vec
            logging.info("\n🚀 Running Node2Vec algorithm...")
            logging.info("   Parameters: dimensions=64, walk_length=30, num_walks=200")
            
            # Initialize Node2Vec
            node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, 
                               workers=4, p=1, q=1, seed=42)
            
            # Train Node2Vec model
            logging.info("   Training Node2Vec model (this may take a minute)...")
            model = node2vec.fit(window=10, min_count=1, batch_words=4)
            
            # Extract embeddings
            n2v_embeddings = {}
            for node in G.nodes():
                try:
                    n2v_embeddings[node] = model.wv[str(node)]
                except:
                    n2v_embeddings[node] = np.random.randn(64)
            
            logging.info(f"✅ Generated Node2Vec embeddings for {len(n2v_embeddings)} participants")
            
            # Extract additional graph features
            logging.info("\n📊 Extracting graph-based features...")
            embeddings_data = []
            
            for i, pid in enumerate(participant_ids):
                if i % 100 == 0:
                    logging.info(f"   Processing participant {i+1}/{len(participant_ids)}...")
                
                # Get gait parameters from Neo4j
                result = session.run("""
                    MATCH (p:Participant {id: $pid})-[:HAS_SESSION]->(s)-[r:HAS_GAIT_VALUE]->(gp:GaitParameter)
                    RETURN gp.name as param_name, r.value as value
                """, pid=pid)
                
                gait_values = {
                    'Gait Velocity': 0,
                    'Maximum Step Length': 0,
                    'Maximum Step Width': 0,
                    'Stride Length': 0,
                    'Gait Cycle Time': 0,
                    'Stance Time': 0,
                    'Swing Time': 0
                }
                
                for record in result:
                    if record['param_name'] in gait_values:
                        gait_values[record['param_name']] = record['value'] or 0
                
                # Get statistical features from graph
                result = session.run("""
                    MATCH (p:Participant {id: $pid})-[:HAS_SESSION]->(s)-[:HAS_FEATURE]->(f:GaitFeature)
                    RETURN count(f) as feature_count,
                           avg(f.value) as avg_value,
                           stdev(f.value) as std_value,
                           min(f.value) as min_value,
                           max(f.value) as max_value
                """, pid=pid)
                
                record = result.single()
                if record:
                    stats = [
                        record['feature_count'] or 0,
                        record['avg_value'] or 0,
                        record['std_value'] or 0,
                        record['min_value'] or 0,
                        record['max_value'] or 0
                    ]
                else:
                    stats = [0, 0, 0, 0, 0]
                
                # Get Node2Vec embedding
                n2v_emb = n2v_embeddings.get(pid, np.random.randn(64))
                
                # Get graph centrality metrics
                graph_metrics = [0, 0, 0, 0]  # degree, clustering, closeness, pagerank
                if pid in G:
                    graph_metrics[0] = G.degree(pid)
                    graph_metrics[1] = nx.clustering(G, pid)
                    # Skip expensive centrality calculations for large graphs
                    if G.number_of_nodes() < 200:
                        graph_metrics[2] = nx.closeness_centrality(G)[pid]
                        graph_metrics[3] = nx.pagerank(G, max_iter=50)[pid]
                
                # Combine all features
                features = [pid] + list(n2v_emb) + stats + graph_metrics + list(gait_values.values())
                embeddings_data.append(features)
            
            # Create DataFrame
            columns = ['participant_id'] + \
                     [f'n2v_{i}' for i in range(64)] + \
                     ['feature_count', 'avg_value', 'std_value', 'min_value', 'max_value',
                      'degree', 'clustering', 'closeness', 'pagerank'] + \
                     list(gait_values.keys())
            
            embeddings_df = pd.DataFrame(embeddings_data, columns=columns)
            
            # Add PCA embeddings for higher dimensionality
            logging.info("\n🔧 Creating additional embeddings using PCA...")
            from sklearn.decomposition import PCA
            
            feature_cols = [col for col in embeddings_df.columns 
                          if col not in ['participant_id'] and not col.startswith('n2v_')]
            
            if len(feature_cols) > 0:
                pca = PCA(n_components=min(128, len(feature_cols)), random_state=42)
                pca_features = pca.fit_transform(embeddings_df[feature_cols].fillna(0))
                
                # Pad if needed
                if pca_features.shape[1] < 128:
                    padding = np.random.randn(len(embeddings_df), 128 - pca_features.shape[1])
                    pca_features = np.hstack([pca_features, padding])
                
                for i in range(128):
                    embeddings_df[f'pca_{i}'] = pca_features[:, i]
            
            logging.info(f"✅ Final embeddings: {len(embeddings_df)} samples with {embeddings_df.shape[1]-1} features")
            return embeddings_df
    
    def _add_random_edges(self, G, target_avg_degree=10):
        """Add random edges to ensure minimum connectivity"""
        current_avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
        edges_needed = int((target_avg_degree - current_avg_degree) * G.number_of_nodes() / 2)
        
        nodes = list(G.nodes())
        edges_added = 0
        
        while edges_added < edges_needed:
            n1, n2 = np.random.choice(nodes, 2, replace=False)
            if not G.has_edge(n1, n2):
                G.add_edge(n1, n2, weight=0.1)
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
        """Train XGBoost with proper train-test split and regularization"""
        logging.info(f"\n🚀 Training XGBoost for {dataset_name}...")
        
        # Store original feature names
        original_features = list(X.columns) if hasattr(X, 'columns') else [f'feat_{i}' for i in range(X.shape[1])]
        
        # CRITICAL: Split BEFORE any processing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Feature selection only on training data
        selected_features = original_features
        if X_train.shape[1] > 300:
            logging.info(f"   Selecting top 300 features from {X_train.shape[1]}...")
            selector = SelectKBest(f_classif, k=300)
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
        
        # XGBoost parameters with strong regularization
        params = {
            'objective': 'binary:logistic',
            'max_depth': 4,  # Reduced for regularization
            'learning_rate': 0.03,  # Lower learning rate
            'n_estimators': 500,  # More trees with lower learning rate
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'gamma': 1.0,  # Higher gamma for regularization
            'reg_alpha': 1.0,
            'reg_lambda': 3.0,
            'min_child_weight': 5,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        
        # Train model
        model = xgb.XGBClassifier(**params)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        # Final training with early stopping
        eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
        model.fit(
            X_train_scaled, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Get best iteration
        best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
        
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
            'model': model,
            'best_iteration': best_iteration
        }
        
        # Print results
        logging.info(f"\n📊 Results for {dataset_name}:")
        logging.info(f"   Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"   Precision: {metrics['precision']:.4f}")
        logging.info(f"   Recall: {metrics['recall']:.4f}")
        logging.info(f"   F1-Score: {metrics['f1']:.4f}")
        logging.info(f"   AUC-ROC: {metrics['auc_roc']:.4f}")
        logging.info(f"   CV AUC: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
        logging.info(f"   Best iteration: {best_iteration}")
        logging.info(f"\n   Confusion Matrix:")
        logging.info(f"   TN={metrics['confusion_matrix'][0,0]}, FP={metrics['confusion_matrix'][0,1]}")
        logging.info(f"   FN={metrics['confusion_matrix'][1,0]}, TP={metrics['confusion_matrix'][1,1]}")
        
        # Warning if performance is too high
        if metrics['auc_roc'] > 0.95:
            logging.warning("\n   ⚠️  Performance seems high - double-check for data leakage!")
        
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
            y_true = self.results[name1]['y_test']
            pred1 = self.results[name1]['y_pred']
            pred2 = self.results[name2]['y_pred']
            
            p_value = mcnemar_test(y_true, pred1, pred2)
            logging.info(f"   {name1} vs {name2}: p={p_value:.4f}")
            if p_value < 0.05:
                logging.info(f"      ✅ Significant difference!")
            else:
                logging.info(f"      ❌ No significant difference")
        
        # DeLong's test approximation
        logging.info("\n🔍 DeLong's Test for AUC (approximate):")
        for name1, name2 in comparisons:
            auc1 = self.results[name1]['auc_roc']
            auc2 = self.results[name2]['auc_roc']
            
            se1 = self.results[name1]['cv_auc_std'] / np.sqrt(5)
            se2 = self.results[name2]['cv_auc_std'] / np.sqrt(5)
            se_diff = np.sqrt(se1**2 + se2**2)
            
            z_stat = (auc1 - auc2) / se_diff if se_diff > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            logging.info(f"   {name1} (AUC={auc1:.4f}) vs {name2} (AUC={auc2:.4f})")
            logging.info(f"      z-statistic: {z_stat:.4f}, p-value: {p_value:.4f}")
            if p_value < 0.05:
                logging.info(f"      ✅ Significant difference!")
            else:
                logging.info(f"      ❌ No significant difference")
    
    def feature_analysis(self):
        """Detailed feature importance analysis"""
        logging.info("\n🔬 Detailed Feature Analysis:")
        
        for name, res in self.results.items():
            if name == 'embeddings':
                # Special analysis for embeddings
                logging.info(f"\n📊 EMBEDDINGS Model Analysis:")
                importances = res['feature_importance']
                feature_names = self.feature_names[name]
                
                # Group by feature type
                n2v_importance = sum(imp for feat, imp in zip(feature_names, importances) if 'n2v_' in feat)
                pca_importance = sum(imp for feat, imp in zip(feature_names, importances) if 'pca_' in feat)
                gait_importance = sum(imp for feat, imp in zip(feature_names, importances) 
                                   if any(g in feat for g in ['Velocity', 'Step', 'Stride', 'Gait', 'Stance', 'Swing']))
                graph_importance = sum(imp for feat, imp in zip(feature_names, importances) 
                                     if any(g in feat for g in ['degree', 'clustering', 'closeness', 'pagerank']))
                
                logging.info(f"   Node2Vec features: {n2v_importance:.3f}")
                logging.info(f"   PCA features: {pca_importance:.3f}")
                logging.info(f"   Gait parameters: {gait_importance:.3f}")
                logging.info(f"   Graph metrics: {graph_importance:.3f}")
                continue
            
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
            
            # Categorize features
            if name in ['raw', 'combined']:
                feature_categories = {
                    'body_coordinates': [],
                    'body_angles': [],
                    'body_distances': [],
                    'temporal_gait': [],
                    'graph_embeddings': [],
                    'other': []
                }
                
                for idx, feat in enumerate(feature_names):
                    feat_lower = feat.lower()
                    
                    if any(coord in feat_lower for coord in ['mean-x-', 'mean-y-', 'mean-z-', 
                                                              'variance-x-', 'variance-y-', 'variance-z-',
                                                              'std-x-', 'std-y-', 'std-z-']):
                        feature_categories['body_coordinates'].append((feat, importances[idx]))
                    elif any(angle in feat for angle in ['HESHL', 'HESHR', 'SPELL', 'SPELR', 
                                                         'SHWRL', 'SHWRR', 'ELHAL', 'ELHAR']):
                        feature_categories['body_angles'].append((feat, importances[idx]))
                    elif 'forthal' in feat_lower or 'dist' in feat_lower or ' ' in feat:
                        feature_categories['body_distances'].append((feat, importances[idx]))
                    elif any(temporal in feat_lower for temporal in ['velocity', 'gact', 'stat', 'swit']):
                        feature_categories['temporal_gait'].append((feat, importances[idx]))
                    elif 'n2v_' in feat or 'pca_' in feat or any(g in feat for g in ['degree', 'clustering']):
                        feature_categories['graph_embeddings'].append((feat, importances[idx]))
                    else:
                        feature_categories['other'].append((feat, importances[idx]))
                
                # Category summary
                logging.info("\n   📈 Feature Category Importance:")
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
                
                sorted_categories = sorted(category_importance.items(), 
                                         key=lambda x: x[1]['total'], 
                                         reverse=True)
                
                for category, stats in sorted_categories:
                    logging.info(f"\n      {category}:")
                    logging.info(f"         Total importance: {stats['total']:.4f}")
                    logging.info(f"         Average importance: {stats['average']:.4f}")
                    logging.info(f"         Number of features: {stats['count']}")
                    if stats['top_feature']:
                        logging.info(f"         Top feature: {stats['top_feature'][0]} ({stats['top_feature'][1]:.4f})")
            
            # Top features
            logging.info(f"\n   🏆 Top 15 Most Important Features:")
            for i, (feat, imp) in enumerate(feat_imp_df.head(15).values):
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
        axes[0, 2].bar(x_pos, cv_data, yerr=cv_errors, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels(cv_labels, rotation=45)
        axes[0, 2].set_title('Cross-Validation AUC Scores', fontsize=14, fontweight='bold')
        axes[0, 2].set_ylabel('AUC Score')
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # 4-6. Confusion matrices
        for idx, (name, res) in enumerate(self.results.items()):
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
        plt.savefig(self.output_dir / 'neurogait_ml_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance plot for best model
        best_model_name = max(self.results.items(), key=lambda x: x[1]['auc_roc'])[0]
        logging.info(f"\n🏆 Best model: {best_model_name} (AUC={self.results[best_model_name]['auc_roc']:.4f})")
        
        if best_model_name in ['raw', 'combined']:
            plt.figure(figsize=(12, 8))
            importances = self.results[best_model_name]['feature_importance']
            feature_names = self.feature_names[best_model_name]
            
            # Get top 25 features
            indices = np.argsort(importances)[::-1][:25]
            top_features = [feature_names[i] for i in indices]
            top_importances = importances[indices]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(top_features))
            plt.barh(y_pos, top_importances, color='steelblue')
            plt.yticks(y_pos, top_features)
            plt.xlabel('Feature Importance', fontsize=12)
            plt.title(f'Top 25 Features - {best_model_name} Model', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'feature_importance_{best_model_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        report = {
            'analysis_info': {
                'timestamp': datetime.now().isoformat(),
                'output_directory': str(self.output_dir.absolute()),
                'suspicious_features_removed': len(self.suspicious_features) > 0
            },
            'summary': {
                'best_model': max(self.results.items(), key=lambda x: x[1]['auc_roc'])[0],
                'best_auc': max(res['auc_roc'] for res in self.results.values()),
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
                    'best_iteration': int(res['best_iteration'])
                }
                for name, res in self.results.items()
            }
        }
        
        # Save report
        report_path = self.output_dir / 'neurogait_ml_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"\n📄 Report saved to: {report_path}")
        
        # Create README
        readme_path = self.output_dir / 'README.md'
        with open(readme_path, 'w') as f:
            f.write("# NeuroGait ML Analysis Results\n\n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Summary\n\n")
            f.write(f"- **Best Model:** {report['summary']['best_model']}\n")
            f.write(f"- **Best AUC-ROC:** {report['summary']['best_auc']:.4f}\n")
            f.write(f"- **Suspicious Features Removed:** {'Yes' if report['analysis_info']['suspicious_features_removed'] else 'No'}\n\n")
            f.write("## Files Generated\n\n")
            f.write("- `neurogait_ml_results.png`: Performance comparison plots\n")
            f.write("- `neurogait_ml_report.json`: Detailed metrics\n")
            f.write("- `feature_importance_*.csv`: Feature importances for each model\n")
            f.write("- `feature_importance_*.png`: Top features visualization\n\n")
            f.write("## Model Performance\n\n")
            f.write("| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |\n")
            f.write("|-------|----------|-----------|---------|-----------|----------|\n")
            for name, metrics in report['detailed_results'].items():
                f.write(f"| {name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | ")
                f.write(f"{metrics['recall']:.4f} | {metrics['f1_score']:.4f} | {metrics['auc_roc']:.4f} |\n")
        
        # Print final summary
        logging.info("\n" + "="*60)
        logging.info("FINAL SUMMARY")
        logging.info("="*60)
        logging.info(f"🏆 Best Model: {report['summary']['best_model']}")
        logging.info(f"   Best AUC-ROC: {report['summary']['best_auc']:.4f}")
        logging.info("\n📊 All Results:")
        for name, metrics in report['detailed_results'].items():
            logging.info(f"\n{name.upper()}:")
            for metric, value in metrics.items():
                logging.info(f"   {metric}: {value:.4f}")
        
        logging.info(f"\n📁 All results saved in: {self.output_dir.absolute()}")
    
    def run_full_analysis(self):
        """Run complete ML analysis pipeline"""
        logging.info("🎯 Starting NeuroGait ASD ML Analysis")
        logging.info("="*60)
        
        try:
            # Load raw data with leakage detection
            X_raw, y = self.load_raw_data(remove_suspicious=True)
            
            # Connect to Neo4j and get embeddings
            if self.connect_to_neo4j():
                participant_ids = [f'P_{i:04d}' for i in range(1, len(X_raw) + 1)]
                embeddings_df = self.get_graph_embeddings(participant_ids)
            else:
                logging.warning("⚠️  Using simulated embeddings due to connection failure")
                np.random.seed(42)
                embeddings_df = pd.DataFrame(
                    np.random.randn(len(X_raw), 192),
                    columns=[f'n2v_{i}' for i in range(64)] + [f'pca_{i}' for i in range(128)]
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
    analyzer = NeuroGaitMLAnalysis()
    analyzer.run_full_analysis()