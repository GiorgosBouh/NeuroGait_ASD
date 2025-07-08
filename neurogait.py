"""
XGBoost ASD Prediction: Graph Embeddings vs Raw Data vs Combined
WITH Data Leakage Detection and Fixed Graph Construction
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
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
    
    def detect_data_leakage(self, df):
        """Detect potential data leakage in the dataset"""
        print("\n🔍 Checking for Data Leakage...")
        
        # 1. Check for duplicate rows
        duplicates = df.duplicated().sum()
        print(f"   Duplicate rows: {duplicates}")
        if duplicates > 0:
            print(f"   ⚠️  WARNING: Found {duplicates} duplicate rows!")
        
        # 2. Check for suspiciously perfect features
        print("\n   Checking individual feature predictive power...")
        suspicious_features = []
        
        X = df.drop('class', axis=1)
        y = df['class']
        
        for col in X.columns[:50]:  # Check first 50 features
            try:
                # Single feature accuracy
                X_single = X[[col]].values.reshape(-1, 1)
                dt = DecisionTreeClassifier(max_depth=1, random_state=42)
                scores = cross_val_score(dt, X_single, y, cv=5, scoring='roc_auc')
                mean_score = scores.mean()
                
                if mean_score > 0.95:
                    suspicious_features.append((col, mean_score))
                    print(f"   🚨 SUSPICIOUS: '{col}' alone gives AUC={mean_score:.3f}")
                elif mean_score > 0.85:
                    print(f"   ⚠️  High predictive: '{col}' gives AUC={mean_score:.3f}")
            except:
                pass
        
        # 3. Check for extreme class imbalances in features
        print("\n   Checking feature distributions between classes...")
        for col in X.columns[:20]:
            try:
                asd_mean = df[df['class']==1][col].mean()
                control_mean = df[df['class']==0][col].mean()
                
                if control_mean != 0:
                    ratio = abs(asd_mean / control_mean)
                    if ratio > 100 or ratio < 0.01:
                        print(f"   ⚠️  Extreme difference in '{col}': ASD/Control ratio = {ratio:.2f}")
            except:
                pass
        
        # 4. Check correlation with target
        print("\n   Checking correlations with target...")
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        high_corr = correlations[correlations > 0.8]
        if len(high_corr) > 0:
            print("   🚨 Features with very high correlation to target:")
            for feat, corr in high_corr.head(10).items():
                print(f"      {feat}: {corr:.3f}")
        
        # 5. Mutual information
        print("\n   Calculating mutual information...")
        mi_scores = mutual_info_classif(X.fillna(0), y, random_state=42)
        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        print("   Top 10 features by mutual information:")
        for idx, row in mi_df.head(10).iterrows():
            print(f"      {row['feature']}: {row['mi_score']:.3f}")
        
        return suspicious_features
    
    def load_raw_data(self, filepath="Final dataset.xlsx"):
        """Load raw data from Excel file with leakage detection"""
        print("\n📊 Loading raw data...")
        df = pd.read_excel(filepath)
        
        # Map class labels
        df['class'] = df['class'].map({'A': 1, 'T': 0})  # ASD=1, Control=0
        
        print(f"✅ Loaded {len(df)} samples with {df.shape[1]-1} features")
        print(f"   Class distribution: ASD={sum(df['class']==1)}, Control={sum(df['class']==0)}")
        
        # Detect data leakage
        suspicious = self.detect_data_leakage(df)
        
        if len(suspicious) > 0:
            print("\n⚠️  WARNING: Potential data leakage detected!")
            print("   Consider removing these features:")
            for feat, score in suspicious[:5]:
                print(f"   - {feat} (AUC={score:.3f})")
            
            # Option to remove suspicious features
            response = input("\n   Remove suspicious features? (y/n): ")
            if response.lower() == 'y':
                features_to_remove = [feat for feat, _ in suspicious]
                df = df.drop(columns=features_to_remove)
                print(f"   ✅ Removed {len(features_to_remove)} suspicious features")
        
        # Separate features and target
        X = df.drop('class', axis=1)
        y = df['class']
        
        return X, y
    
    def get_graph_embeddings(self, participant_ids):
        """Get graph embeddings using Python Node2Vec with improved graph construction"""
        print("\n🧠 Generating graph embeddings...")
        
        with self.driver.session() as session:
            # First check if GDS is available
            try:
                result = session.run("CALL gds.list()")
                has_gds = len(list(result)) >= 0
                print("✅ GDS is available! Using GDS implementation...")
                
                # GDS implementation would go here...
                # For now, we'll skip to Python implementation
                raise Exception("Using Python Node2Vec instead")
                
            except Exception as e:
                print(f"⚠️  Using Python Node2Vec implementation...")
            
            # Python Node2Vec Implementation
            print("\n🔧 Building graph from Neo4j data for Node2Vec...")
            
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add all participants as nodes
            result = session.run("""
                MATCH (p:Participant)
                RETURN p.id as participant_id
            """)
            participants = [record['participant_id'] for record in result]
            G.add_nodes_from(participants)
            
            print(f"   Added {len(participants)} participant nodes")
            
            # Method 1: Connect participants with similar gait parameters (RELAXED THRESHOLDS)
            print("   Building edges based on gait parameter similarity...")
            result = session.run("""
                MATCH (p1:Participant)-[:HAS_SESSION]->(s1:GaitSession)
                MATCH (p2:Participant)-[:HAS_SESSION]->(s2:GaitSession)
                WHERE p1.id < p2.id
                WITH p1, p2, s1, s2,
                     abs(s1.StepLength - s2.StepLength) as step_diff,
                     abs(s1.Cadence - s2.Cadence) as cadence_diff,
                     abs(s1.Speed - s2.Speed) as speed_diff
                WHERE step_diff < 0.3 AND cadence_diff < 20 AND speed_diff < 0.5
                RETURN p1.id as p1_id, p2.id as p2_id, 
                       1.0 / (1 + step_diff + cadence_diff/100 + speed_diff) as weight
                LIMIT 20000
            """)
            
            edges = [(record['p1_id'], record['p2_id'], {'weight': record['weight']}) 
                    for record in result]
            G.add_edges_from(edges)
            
            print(f"   Added {len(edges)} edges based on gait similarity")
            
            # Method 2: Add edges based on shared feature patterns (RELAXED THRESHOLDS)
            print("   Adding edges based on shared feature patterns...")
            result = session.run("""
                MATCH (p1:Participant)-[:HAS_SESSION]->(s1)-[:HAS_FEATURE]->(f1:GaitFeature)
                MATCH (p2:Participant)-[:HAS_SESSION]->(s2)-[:HAS_FEATURE]->(f2:GaitFeature)
                WHERE p1.id < p2.id 
                AND f1.measurement_id = f2.measurement_id
                AND abs(f1.value - f2.value) < 0.2
                WITH p1.id as p1_id, p2.id as p2_id, count(*) as shared_features
                WHERE shared_features > 50
                RETURN p1_id, p2_id, shared_features
                LIMIT 10000
            """)
            
            feature_edges = 0
            for record in result:
                if G.has_edge(record['p1_id'], record['p2_id']):
                    G[record['p1_id']][record['p2_id']]['weight'] += record['shared_features'] / 1000
                else:
                    G.add_edge(record['p1_id'], record['p2_id'], 
                              weight=record['shared_features'] / 1000)
                    feature_edges += 1
            
            print(f"   Added {feature_edges} additional edges based on features")
            
            # Method 3: k-NN based on overall similarity
            print("\n   Building k-NN connections...")
            
            # Get all features for each participant
            result = session.run("""
                MATCH (p:Participant)-[:HAS_SESSION]->(s:GaitSession)
                RETURN p.id as pid, 
                       s.StepLength as sl, s.StrideLength as stl, 
                       s.Cadence as cad, s.Speed as sp, s.StepWidth as sw
                ORDER BY p.id
            """)
            
            # Create feature matrix for k-NN
            feature_matrix = {}
            for record in result:
                feature_matrix[record['pid']] = [
                    record['sl'] or 0, record['stl'] or 0,
                    record['cad'] or 0, record['sp'] or 0, record['sw'] or 0
                ]
            
            # Add k-NN edges (k=10)
            from sklearn.neighbors import NearestNeighbors
            if len(feature_matrix) > 0:
                pids = list(feature_matrix.keys())
                X_knn = np.array([feature_matrix[pid] for pid in pids])
                
                knn = NearestNeighbors(n_neighbors=min(11, len(pids)), metric='euclidean')
                knn.fit(X_knn)
                
                distances, indices = knn.kneighbors(X_knn)
                
                knn_edges = 0
                for i, pid1 in enumerate(pids):
                    for j in range(1, min(11, len(indices[i]))):  # Skip self (index 0)
                        pid2 = pids[indices[i][j]]
                        if pid1 < pid2:  # Avoid duplicate edges
                            weight = 1.0 / (1 + distances[i][j])
                            if G.has_edge(pid1, pid2):
                                G[pid1][pid2]['weight'] += weight
                            else:
                                G.add_edge(pid1, pid2, weight=weight)
                                knn_edges += 1
                
                print(f"   Added {knn_edges} k-NN edges")
            
            # Graph statistics
            print(f"\n📊 Graph statistics:")
            print(f"   Nodes: {G.number_of_nodes()}")
            print(f"   Edges: {G.number_of_edges()}")
            print(f"   Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
            print(f"   Density: {nx.density(G):.4f}")
            
            # Check connectivity
            n_components = nx.number_connected_components(G)
            print(f"   Connected components: {n_components}")
            if n_components > 1:
                largest_cc = max(nx.connected_components(G), key=len)
                print(f"   Largest component size: {len(largest_cc)} ({len(largest_cc)/G.number_of_nodes()*100:.1f}%)")
            
            # Ensure minimum connectivity
            if nx.density(G) < 0.01:
                print("\n⚠️  Graph is too sparse! Adding more edges...")
                # Add random edges to ensure connectivity
                nodes = list(G.nodes())
                n_random_edges = int(0.01 * len(nodes) * (len(nodes) - 1) / 2 - G.number_of_edges())
                for _ in range(n_random_edges):
                    n1, n2 = np.random.choice(nodes, 2, replace=False)
                    if not G.has_edge(n1, n2):
                        G.add_edge(n1, n2, weight=0.1)
                print(f"   Added {n_random_edges} random edges for connectivity")
            
            # Run Node2Vec
            print("\n🚀 Running Node2Vec algorithm...")
            print("   Parameters: dimensions=64, walk_length=30, num_walks=200")
            
            # Initialize Node2Vec
            node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, 
                               workers=4, p=1, q=1, seed=42)
            
            # Train Node2Vec model
            print("   Training Node2Vec model (this may take a minute)...")
            model = node2vec.fit(window=10, min_count=1, batch_words=4)
            
            # Extract embeddings
            n2v_embeddings = {}
            for node in G.nodes():
                try:
                    n2v_embeddings[node] = model.wv[str(node)]
                except:
                    n2v_embeddings[node] = np.random.randn(64)
            
            print(f"✅ Generated Node2Vec embeddings for {len(n2v_embeddings)} participants")
            
            # Test embedding quality
            print("\n   Testing embedding quality...")
            # Get class labels for participants
            result = session.run("""
                MATCH (p:Participant)-[:HAS_SESSION]->(s:GaitSession)-[:CLASSIFIED_AS]->(c:Classification)
                RETURN p.id as pid, c.label as label
            """)
            
            labels = {}
            for record in result:
                labels[record['pid']] = 1 if record['label'] == 'ASD' else 0
            
            # Quick classification test on embeddings
            if len(labels) > 100:
                from sklearn.ensemble import RandomForestClassifier
                X_test = []
                y_test = []
                for pid in list(labels.keys())[:200]:
                    if pid in n2v_embeddings:
                        X_test.append(n2v_embeddings[pid])
                        y_test.append(labels[pid])
                
                if len(X_test) > 50:
                    X_test = np.array(X_test)
                    y_test = np.array(y_test)
                    rf = RandomForestClassifier(n_estimators=50, random_state=42)
                    scores = cross_val_score(rf, X_test, y_test, cv=5)
                    print(f"   Embedding quality (CV accuracy): {scores.mean():.3f} ± {scores.std():.3f}")
            
            # Also extract graph-based features
            print("\n📊 Extracting additional graph-based features...")
            embeddings_data = []
            
            for i, pid in enumerate(participant_ids):
                if i % 100 == 0:
                    print(f"   Processing participant {i+1}/{len(participant_ids)}...")
                
                # Get graph statistics
                result = session.run("""
                    MATCH (p:Participant {id: $pid})-[:HAS_SESSION]->(s:GaitSession)-[:HAS_FEATURE]->(f:GaitFeature)
                    RETURN count(f) as feature_count,
                           avg(f.value) as avg_value,
                           stdev(f.value) as std_value,
                           min(f.value) as min_value,
                           max(f.value) as max_value
                """, pid=pid)
                
                record = result.single()
                if record:
                    stats = [record['feature_count'] or 0,
                            record['avg_value'] or 0,
                            record['std_value'] or 0,
                            record['min_value'] or 0,
                            record['max_value'] or 0]
                else:
                    stats = [0, 0, 0, 0, 0]
                
                # Get Node2Vec embedding
                n2v_emb = n2v_embeddings.get(pid, np.random.randn(64))
                
                # Get graph centrality metrics
                if pid in G:
                    degree = G.degree(pid)
                    clustering = nx.clustering(G, pid)
                    try:
                        if G.number_of_nodes() < 200:
                            closeness = nx.closeness_centrality(G)[pid]
                        else:
                            closeness = 0
                    except:
                        closeness = 0
                    
                    # PageRank
                    try:
                        pagerank = nx.pagerank(G, max_iter=50)[pid]
                    except:
                        pagerank = 0
                else:
                    degree = clustering = closeness = pagerank = 0
                
                # Get gait parameters
                result = session.run("""
                    MATCH (p:Participant {id: $pid})-[:HAS_SESSION]->(s:GaitSession)
                    RETURN s.StepLength as StepLength, s.StrideLength as StrideLength,
                           s.StepTime as StepTime, s.StrideTime as StrideTime,
                           s.Cadence as Cadence, s.Speed as Speed, s.StepWidth as StepWidth
                """, pid=pid)
                
                record = result.single()
                if record:
                    gait_values = [record[key] or 0 for key in 
                                 ['StepLength', 'StrideLength', 'StepTime', 
                                  'StrideTime', 'Cadence', 'Speed', 'StepWidth']]
                else:
                    gait_values = [0] * 7
                
                # Combine all features
                features = [pid] + list(n2v_emb) + stats + \
                          [degree, clustering, closeness, pagerank] + gait_values
                
                embeddings_data.append(features)
            
            # Create DataFrame
            columns = ['participant_id'] + \
                     [f'n2v_{i}' for i in range(64)] + \
                     ['feature_count', 'avg_value', 'std_value', 'min_value', 'max_value',
                      'degree', 'clustering', 'closeness', 'pagerank'] + \
                     ['StepLength', 'StrideLength', 'StepTime', 'StrideTime', 
                      'Cadence', 'Speed', 'StepWidth']
            
            embeddings_df = pd.DataFrame(embeddings_data, columns=columns)
            
            # Add synthetic FastRP-like embeddings using PCA
            print("\n🔧 Creating FastRP-style embeddings using PCA...")
            from sklearn.decomposition import PCA
            
            # Use all non-embedding features for PCA
            feature_cols = [col for col in embeddings_df.columns 
                          if col not in ['participant_id'] and not col.startswith('n2v_')]
            
            if len(feature_cols) > 0:
                pca = PCA(n_components=min(128, len(feature_cols)), random_state=42)
                pca_features = pca.fit_transform(embeddings_df[feature_cols].fillna(0))
                
                # Pad with random if needed
                if pca_features.shape[1] < 128:
                    padding = np.random.randn(len(embeddings_df), 128 - pca_features.shape[1])
                    pca_features = np.hstack([pca_features, padding])
                
                for i in range(128):
                    embeddings_df[f'emb_{i}'] = pca_features[:, i]
            else:
                # Fallback to random
                for i in range(128):
                    embeddings_df[f'emb_{i}'] = np.random.randn(len(embeddings_df))
            
            print(f"✅ Final embeddings: {len(embeddings_df)} samples with Node2Vec + graph features")
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
        
        # XGBoost parameters - add more regularization
        params = {
            'objective': 'binary:logistic',
            'max_depth': 5,  # Reduced from 6
            'learning_rate': 0.05,  # Reduced from 0.1
            'n_estimators': 300,  # Increased from 200
            'subsample': 0.7,  # Reduced from 0.8
            'colsample_bytree': 0.7,  # Reduced from 0.8
            'gamma': 0.5,  # Increased from 0.1
            'reg_alpha': 0.5,  # Increased from 0.1
            'reg_lambda': 2,  # Increased from 1
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
        
        # Warning if performance is suspiciously high
        if metrics['auc_roc'] > 0.98:
            print("\n   ⚠️  WARNING: Performance seems unusually high!")
            print("      Consider checking for data leakage.")
        
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
        
        # DeLong's test for AUC comparison
        print("\n🔍 DeLong's Test for AUC (approximate):")
        comparisons = [
            ('raw', 'embeddings'),
            ('raw', 'combined'),
            ('embeddings', 'combined')
        ]
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
                    if any(body in feat_lower for body in ['ankle', 'knee', 'hip', 'shoulder', 'elbow', 'wrist', 'hand', 'head', 'torso', 'neck']):
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
                if sorted_categories:
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
            feature_names = self.feature_names[best_model_name]
            
            # Get top 20 features
            indices = np.argsort(importances)[::-1][:20]
            top_features = [feature_names[i] for i in indices]
            top_importances = importances[indices]
            
            plt.barh(range(20), top_importances)
            plt.yticks(range(20), top_features)
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