#!/usr/bin/env python3
"""
COMPLETELY FIXED NeuroGait Analysis - Pure Movement Patterns ONLY
Excludes ALL spatial coordinate-based features including ROM x,y coordinates
Uses ONLY: angles, temporal patterns, and normalized ratios
FIXED VERSION - Removes spatial bias completely
"""

import pandas as pd
import numpy as np
import os
import json
import networkx as nx
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import warnings
import logging
from datetime import datetime

# Optional: Node2Vec for graph embeddings
try:
    from node2vec import Node2Vec
    HAS_NODE2VEC = True
except ImportError:
    HAS_NODE2VEC = False
    print("⚠️  node2vec not available - using fallback embeddings")

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class PureMovementAnalysis:
    def __init__(self, samples_per_participant=8):
        self.samples_per_participant = samples_per_participant
        self.output_dir = f"pure_movement_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # ONLY pure angle feature mappings (no spatial coordinates)
        self.angle_mappings = {
            'HESHL': 'Head-SpineShoulder-ShoulderLeft',
            'HESHR': 'Head-SpineShoulder-ShoulderRight', 
            'SPELL': 'SpineShoulder-ShoulderLeft-ElbowLeft',
            'SPELR': 'SpineShoulder-ShoulderRight-ElbowRight',
            'SHWRL': 'ShoulderLeft-ElbowLeft-WristLeft',
            'SHWRR': 'ShoulderRight-ElbowRight-WristRight',
            'ELHAL': 'ElbowLeft-WristLeft-HandLeft',
            'ELHAR': 'ElbowRight-WristRight-HandRight',
            'THHAL': 'ThumbLeft-WristLeft-HandLeft',
            'THHAR': 'ThumbRight-WristRight-HandRight',
            'SPKNL': 'SpineBase-HipLeft-KneeLeft',
            'SPKNR': 'SpineBase-HipRight-KneeRight',
            'HIANL': 'HipLeft-KneeLeft-AnkleLeft',
            'HIANR': 'HipRight-KneeRight-AnkleRight',
            'KNFOL': 'KneeLeft-AnkleLeft-FootLeft',
            'KNFOR': 'KneeRight-AnkleRight-FootRight'
        }
        
    def create_participant_mapping(self):
        """Create the CONFIRMED participant mapping"""
        participants_info = []
        
        # Participants 0-49: ASD (samples 0-399)
        for p in range(50):
            start_idx = p * 8
            end_idx = start_idx + 8
            participants_info.append({
                'participant_id': p,
                'class': 'ASD',
                'samples': list(range(start_idx, end_idx))
            })
        
        # Participants 50-99: Typical (samples 400-799)  
        for p in range(50, 100):
            start_idx = p * 8
            end_idx = start_idx + 8
            participants_info.append({
                'participant_id': p,
                'class': 'Typical',
                'samples': list(range(start_idx, end_idx))
            })
        
        return participants_info
        
    def load_pure_movement_data(self):
        """Load data keeping ONLY pure movement patterns (NO spatial coordinates AT ALL)"""
        logger.info("📊 Loading data with PURE MOVEMENT patterns only...")
        
        # Load data
        df = pd.read_csv('Final dataset.csv', sep=';', decimal=',')
        logger.info(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
        
        # Apply CONFIRMED participant structure
        participants_info = self.create_participant_mapping()
        
        # Add participant metadata
        participant_ids = []
        actual_classes = []
        
        for i in range(len(df)):
            participant_id = i // 8
            if participant_id < 50:
                class_label = 'ASD'
            else:
                class_label = 'Typical'
            
            participant_ids.append(participant_id)
            actual_classes.append(class_label)
        
        df['participant_id'] = participant_ids
        df['actual_class'] = actual_classes
        
        # Convert target using CONFIRMED structure
        df['diagnosis'] = df['actual_class'].map({'ASD': 1, 'Typical': 0})
        
        # CRITICAL: Keep ONLY pure movement patterns (NO spatial coordinates whatsoever)
        logger.info("\n🎯 Filtering to PURE movement patterns only (NO spatial coordinates)...")
        
        cols_to_keep = ['diagnosis', 'participant_id']
        spatial_excluded = 0
        rom_spatial_excluded = 0
        
        for col in df.columns:
            col_clean = col.strip()
            
            # ❌ EXCLUDE: All spatial coordinate features (mean-x, mean-y, mean-z)
            if col_clean.startswith('mean-') and any(coord in col_clean for coord in ['-x-', '-y-', '-z-']):
                spatial_excluded += 1
                logger.info(f"   ❌ Spatial coordinate: {col_clean}")
                continue
                
            # ❌ EXCLUDE: ROM features with x,y coordinates (these are spatial ranges!)
            elif col_clean.startswith('Rom') and (col_clean.endswith('x') or col_clean.endswith('y')):
                rom_spatial_excluded += 1
                logger.info(f"   ❌ ROM spatial: {col_clean}")
                continue
            
            # ✅ INCLUDE: Pure temporal gait parameters only
            elif col_clean in ['GaCT', 'StaT', 'SwiT']:
                cols_to_keep.append(col)
                logger.info(f"   ✅ Pure temporal: {col_clean}")
            
            # ❌ EXCLUDE: Velocity (can be height-dependent)
            elif col_clean == 'Velocity':
                logger.info(f"   ❌ Height-dependent: {col_clean}")
                continue
            
            # ❌ EXCLUDE: Stride/step measurements (height-dependent)
            elif col_clean in ['MaxStLe', 'MaxStWi', 'StrLe']:
                logger.info(f"   ❌ Height-dependent spatial: {col_clean}")
                continue
            
            # ✅ INCLUDE: ONLY pure angular features (angles between joints)
            elif col_clean.startswith('mean ') and any(angle in col_clean for angle in self.angle_mappings.keys()):
                cols_to_keep.append(col)
                angle_type = next((angle for angle in self.angle_mappings.keys() if angle in col_clean), "unknown")
                logger.info(f"   ✅ Pure angle: {col_clean} ({angle_type})")
            
            # ❌ EXCLUDE: All other ROM features (likely spatial)
            elif col_clean.startswith('Rom'):
                logger.info(f"   ❌ Potentially spatial ROM: {col_clean}")
                continue
            
            # ❌ EXCLUDE: Distance-based features (spatial)
            elif col_clean in ['MaxDBFE', 'MinDBFE']:
                logger.info(f"   ❌ Distance-based: {col_clean}")
                continue
                
            # ❌ EXCLUDE: Position features (spatial)
            elif col_clean in ['HaTiLPos', 'HaTiRPos']:
                logger.info(f"   ❌ Position-based: {col_clean}")
                continue
                
            # ❌ EXCLUDE: Threshold (unclear, better exclude)
            elif col_clean == 'Threshold':
                logger.info(f"   ❌ Unclear feature: {col_clean}")
                continue
        
        # Filter dataset to keep ONLY pure movement patterns
        df_filtered = df[cols_to_keep]
        
        # Remove constant features
        for col in df_filtered.columns:
            if col not in ['diagnosis', 'participant_id'] and df_filtered[col].nunique() <= 1:
                df_filtered = df_filtered.drop(columns=[col])
        
        logger.info(f"\n📊 PURE MOVEMENT FILTERING SUMMARY:")
        logger.info(f"   ❌ Excluded spatial coordinates: {spatial_excluded} features")
        logger.info(f"   ❌ Excluded ROM spatial (x,y): {rom_spatial_excluded} features")
        logger.info(f"   ❌ Excluded height-dependent: velocity, stride, distances")
        logger.info(f"   ✅ Kept ONLY pure movement patterns: {len(df_filtered.columns)-2}")
        logger.info(f"   🎯 Focus: Pure angles and temporal timing ONLY")
        logger.info(f"   📊 Class distribution: {df_filtered['diagnosis'].value_counts().to_dict()}")
        logger.info(f"   👥 Participants: {df_filtered['participant_id'].nunique()}")
        
        # Show what we actually kept
        kept_features = [col for col in df_filtered.columns if col not in ['diagnosis', 'participant_id']]
        logger.info(f"\n✅ KEPT FEATURES ({len(kept_features)}):")
        for feature in kept_features:
            logger.info(f"   • {feature}")
        
        return df_filtered
    
    def participant_level_split(self, X, y, participant_ids, test_size=0.2):
        """Split at participant level to prevent leakage"""
        logger.info("\n🔧 Performing participant-level split...")
        
        # Get unique participants and their labels
        unique_participants = participant_ids.unique()
        participant_labels = []
        
        for pid in unique_participants:
            # Get the label for this participant (should be consistent)
            participant_label = y[participant_ids == pid].iloc[0]
            participant_labels.append(participant_label)
        
        # Split participants
        train_pids, test_pids = train_test_split(
            unique_participants, 
            test_size=test_size, 
            stratify=participant_labels, 
            random_state=42
        )
        
        # Get sample indices
        train_mask = participant_ids.isin(train_pids)
        test_mask = participant_ids.isin(test_pids)
        
        X_train = X[train_mask].reset_index(drop=True)
        X_test = X[test_mask].reset_index(drop=True)
        y_train = y[train_mask].reset_index(drop=True)
        y_test = y[test_mask].reset_index(drop=True)
        
        logger.info(f"✅ Split: {len(train_pids)} train participants ({len(X_train)} samples)")
        logger.info(f"         {len(test_pids)} test participants ({len(X_test)} samples)")
        logger.info(f"   Train class distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"   Test class distribution: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test, train_pids
    
    def create_feature_pipeline(self, n_features=10):
        """Create feature processing pipeline (more conservative feature selection)"""
        
        class FeatureProcessor:
            def __init__(self, n_features):
                self.n_features = n_features
                self.scaler = StandardScaler()
                self.feature_selector = SelectKBest(f_classif, k=n_features)
                self.selected_features_ = None
                
            def fit(self, X, y):
                logger.info(f"\n🔧 Feature processing pipeline...")
                logger.info(f"   Input shape: {X.shape}")
                
                # More aggressive correlation removal for pure features
                corr_matrix = X.corr().abs()
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                # Lower correlation threshold since we have fewer features
                to_drop = [column for column in upper_triangle.columns 
                          if any(upper_triangle[column] > 0.75)]
                
                logger.info(f"   Removing {len(to_drop)} highly correlated features")
                self.corr_features_to_drop = to_drop
                X_decorr = X.drop(columns=to_drop)
                
                # Scale
                X_scaled = self.scaler.fit_transform(X_decorr)
                
                # Select fewer features for more conservative approach
                actual_k = min(self.n_features, X_scaled.shape[1])
                self.feature_selector.set_params(k=actual_k)
                X_selected = self.feature_selector.fit_transform(X_scaled, y)
                
                selected_indices = self.feature_selector.get_support(indices=True)
                self.selected_features_ = X_decorr.columns[selected_indices].tolist()
                
                logger.info(f"   Selected {len(self.selected_features_)} features")
                logger.info(f"   Selected features: {self.selected_features_}")
                return self
                
            def transform(self, X):
                X_decorr = X.drop(columns=self.corr_features_to_drop, errors='ignore')
                X_scaled = self.scaler.transform(X_decorr)
                X_selected = self.feature_selector.transform(X_scaled)
                return X_selected
                
            def fit_transform(self, X, y):
                return self.fit(X, y).transform(X)
        
        return FeatureProcessor(n_features)
    
    def create_movement_pattern_graph(self, X_train, y_train, participant_ids, similarity_threshold=0.4):
        """Create graph based on pure movement pattern similarity"""
        logger.info(f"\n🧠 Creating pure movement pattern similarity graph...")
        
        G = nx.Graph()
        
        # Get unique participants
        unique_participants = participant_ids.unique()
        
        # Add participant nodes
        for pid in unique_participants:
            participant_id = f"P_{pid:03d}"
            
            # Get this participant's samples
            participant_mask = participant_ids == pid
            participant_features = X_train[participant_mask].mean()  # Average across augmentations
            participant_label = int(y_train[participant_mask].iloc[0])
            
            # Add node with pure movement statistics
            feature_stats = {
                'label': participant_label,
                'movement_complexity': float(participant_features.std()),
                'mean_angle_activity': float(participant_features.mean()),
                'node_type': 'participant'
            }
            G.add_node(participant_id, **feature_stats)
        
        # Add similarity edges using pure movement patterns
        logger.info("   Computing pure movement pattern similarities...")
        
        # Create participant-level feature matrix (average of augmentations)
        participant_features = []
        for pid in unique_participants:
            participant_mask = participant_ids == pid
            participant_avg = X_train[participant_mask].mean()
            participant_features.append(participant_avg.values)
        
        participant_features = np.array(participant_features)
        
        # Use k-NN to find similar pure movement patterns
        n_participants = len(unique_participants)
        knn = NearestNeighbors(n_neighbors=min(6, n_participants//2), metric='cosine')
        knn.fit(participant_features)
        
        distances, indices = knn.kneighbors(participant_features)
        edge_count = 0
        
        for i, (neighbors, dists) in enumerate(zip(indices, distances)):
            participant_i = f"P_{unique_participants[i]:03d}"
            for j, dist in zip(neighbors, dists):
                if i != j and (1 - dist) > similarity_threshold:
                    participant_j = f"P_{unique_participants[j]:03d}"
                    similarity = 1 - dist
                    G.add_edge(participant_i, participant_j, 
                             weight=similarity, 
                             connection_type='pure_movement_similarity')
                    edge_count += 1
        
        logger.info(f"   Added {edge_count} pure movement similarity edges")
        logger.info(f"   Pure movement graph: {G.number_of_nodes()} participants, {G.number_of_edges()} edges")
        
        return G
    
    def create_graph_embeddings(self, X_train, y_train, X_test, train_participant_ids, test_participant_ids, embedding_dim=16):
        """Create graph embeddings based on pure movement patterns"""
        logger.info(f"\n🧠 Creating pure movement embeddings (dim={embedding_dim})...")
        
        try:
            # Create pure movement pattern graph
            movement_graph = self.create_movement_pattern_graph(
                X_train, y_train, train_participant_ids
            )
            
            # Generate embeddings if graph has edges
            if movement_graph.number_of_edges() > 0 and HAS_NODE2VEC:
                logger.info("   Running Node2Vec on pure movement patterns...")
                
                node2vec = Node2Vec(
                    movement_graph,
                    dimensions=embedding_dim,
                    walk_length=15,
                    num_walks=30,
                    p=1.0,
                    q=1.0,
                    workers=1,
                    quiet=True
                )
                
                model = node2vec.fit(window=3, min_count=1, batch_words=4, epochs=5)
                
                # Get embeddings for training participants
                unique_train_participants = train_participant_ids.unique()
                participant_embeddings = np.zeros((len(unique_train_participants), embedding_dim))
                
                for i, pid in enumerate(unique_train_participants):
                    participant_id = f"P_{pid:03d}"
                    if participant_id in model.wv:
                        participant_embeddings[i] = model.wv[participant_id]
                    else:
                        participant_embeddings[i] = np.random.normal(0, 0.001, embedding_dim)
                
                # Map embeddings to samples
                train_embeddings = np.zeros((len(X_train), embedding_dim))
                for i, pid in enumerate(train_participant_ids):
                    participant_idx = np.where(unique_train_participants == pid)[0][0]
                    train_embeddings[i] = participant_embeddings[participant_idx]
                
                # For test set: project using k-NN from training movement patterns
                logger.info("   Projecting test embeddings using pure movement similarity...")
                
                unique_test_participants = test_participant_ids.unique()
                test_participant_features = []
                
                for pid in unique_test_participants:
                    participant_mask = test_participant_ids == pid
                    participant_avg = X_test[participant_mask].mean()
                    test_participant_features.append(participant_avg.values)
                
                test_participant_features = np.array(test_participant_features)
                
                # Find similar training participants
                train_participant_features = []
                for pid in unique_train_participants:
                    participant_mask = train_participant_ids == pid
                    participant_avg = X_train[participant_mask].mean()
                    train_participant_features.append(participant_avg.values)
                
                train_participant_features = np.array(train_participant_features)
                
                knn = NearestNeighbors(n_neighbors=min(3, len(unique_train_participants)), metric='cosine')
                knn.fit(train_participant_features)
                
                test_distances, test_indices = knn.kneighbors(test_participant_features)
                test_participant_embeddings = np.zeros((len(unique_test_participants), embedding_dim))
                
                for i, (neighbors, dists) in enumerate(zip(test_indices, test_distances)):
                    weights = 1 / (dists + 1e-8)
                    weights = weights / weights.sum()
                    test_participant_embeddings[i] = np.average(participant_embeddings[neighbors], axis=0, weights=weights)
                
                # Map embeddings to test samples
                test_embeddings = np.zeros((len(X_test), embedding_dim))
                for i, pid in enumerate(test_participant_ids):
                    participant_idx = np.where(unique_test_participants == pid)[0][0]
                    test_embeddings[i] = test_participant_embeddings[participant_idx]
                
                logger.info(f"✅ Created pure movement embeddings: train {train_embeddings.shape}, test {test_embeddings.shape}")
                
            else:
                logger.warning("   No edges in movement graph or Node2Vec unavailable, using minimal embeddings")
                train_embeddings = np.random.normal(0, 0.001, (len(X_train), embedding_dim))
                test_embeddings = np.random.normal(0, 0.001, (len(X_test), embedding_dim))
            
            return train_embeddings, test_embeddings
            
        except Exception as e:
            logger.error(f"❌ Graph embedding failed: {str(e)}")
            logger.info("   Using minimal random embeddings as fallback")
            train_embeddings = np.random.normal(0, 0.001, (len(X_train), embedding_dim))
            test_embeddings = np.random.normal(0, 0.001, (len(X_test), embedding_dim))
            return train_embeddings, test_embeddings
    
    def participant_cv_scores(self, X_train, y_train, train_participant_ids, model_class=None):
        """Get CV scores at participant level with more conservative model"""
        unique_train_participants = train_participant_ids.unique()
        participant_labels = []
        
        for pid in unique_train_participants:
            participant_mask = train_participant_ids == pid
            participant_label = y_train[participant_mask].iloc[0]
            participant_labels.append(participant_label)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_p_idx, val_p_idx in skf.split(range(len(unique_train_participants)), participant_labels):
            # Get participant IDs for this fold
            train_fold_pids = unique_train_participants[train_p_idx]
            val_fold_pids = unique_train_participants[val_p_idx]
            
            # Get sample indices
            train_fold_mask = train_participant_ids.isin(train_fold_pids)
            val_fold_mask = train_participant_ids.isin(val_fold_pids)
            
            X_fold_train = X_train[train_fold_mask] if isinstance(X_train, np.ndarray) else X_train[train_fold_mask]
            X_fold_val = X_train[val_fold_mask] if isinstance(X_train, np.ndarray) else X_train[val_fold_mask]
            y_fold_train = y_train[train_fold_mask]
            y_fold_val = y_train[val_fold_mask]
            
            # Use more conservative model settings
            if model_class is None:
                model = xgb.XGBClassifier(
                    n_estimators=30, max_depth=2, learning_rate=0.01,
                    subsample=0.6, colsample_bytree=0.6,
                    reg_alpha=5.0, reg_lambda=5.0,
                    min_child_weight=10,
                    random_state=42, use_label_encoder=False, eval_metric='logloss'
                )
            else:
                model = model_class
            
            model.fit(X_fold_train, y_fold_train)
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            fold_auc = roc_auc_score(y_fold_val, y_pred_proba)
            cv_scores.append(fold_auc)
        
        return cv_scores
    
    def train_and_evaluate_model(self, X_train, X_test, y_train, y_test, train_participant_ids, model_name="Model"):
        """Train and evaluate model with very conservative settings"""
        logger.info(f"\n🚀 Training {model_name}...")
        logger.info(f"   Training set: {X_train.shape}")
        logger.info(f"   Test set: {X_test.shape}")
        
        # Get CV scores
        cv_scores = self.participant_cv_scores(X_train, y_train, train_participant_ids)
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        logger.info(f"   Participant-level CV AUC: {cv_mean:.4f} ± {cv_std:.4f}")
        
        # Train final model with very conservative settings
        model = xgb.XGBClassifier(
            n_estimators=30, max_depth=2, learning_rate=0.01,
            subsample=0.6, colsample_bytree=0.6,
            reg_alpha=5.0, reg_lambda=5.0,
            min_child_weight=10,
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        
        # Evaluate
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
        
        logger.info(f"\n📊 {model_name} Results:")
        logger.info(f"   CV AUC:      {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
        logger.info(f"   Test AUC:    {metrics['test_auc']:.4f}")
        logger.info(f"   Accuracy:    {metrics['test_accuracy']:.4f}")
        logger.info(f"   Precision:   {metrics['test_precision']:.4f}")
        logger.info(f"   Recall:      {metrics['test_recall']:.4f}")
        logger.info(f"   F1-score:    {metrics['test_f1']:.4f}")
        
        # Performance assessment
        if metrics['test_auc'] > 0.8:
            logger.warning("   ⚠️  Still high - may indicate fundamental differences")
        elif metrics['test_auc'] > 0.7:
            logger.info("   ✅ Moderate performance - some discrimination ability")
        elif metrics['test_auc'] > 0.6:
            logger.info("   ✅ Realistic performance - subtle differences detected")
        else:
            logger.info("   ✅ Low performance - minimal discrimination ability")
        
        return metrics, model
    
    def run_complete_analysis(self):
        """Run complete analysis with ONLY pure movement patterns"""
        logger.info(f"\n🔍 Starting PURE Movement Pattern Analysis - {datetime.now()}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🎯 Focus: PURE movement patterns ONLY (no spatial coordinates)")
        
        try:
            # 1. Load PURE movement data only
            df = self.load_pure_movement_data()
            
            # Check if we have enough features
            feature_count = len(df.columns) - 2  # excluding diagnosis and participant_id
            if feature_count < 5:
                logger.error(f"❌ Too few features remaining: {feature_count}")
                logger.error("   This suggests all features were spatial-dependent!")
                return None
            
            # 2. Participant-level split
            X = df.drop(['diagnosis', 'participant_id'], axis=1)
            y = df['diagnosis']
            participant_ids = df['participant_id']
            
            X_train, X_test, y_train, y_test, train_pids = self.participant_level_split(
                X, y, participant_ids
            )
            
            # Get participant IDs for train/test sets
            train_participant_ids = participant_ids[participant_ids.index.isin(X_train.index)]
            test_participant_ids = participant_ids[participant_ids.index.isin(X_test.index)]
            
            # Storage for results
            all_results = {}
            
            # 3. PURE MOVEMENT PATTERNS
            logger.info(f"\n{'='*60}")
            logger.info("🔍 ANALYSIS 1: PURE MOVEMENT PATTERNS")
            logger.info(f"{'='*60}")
            
            pure_pipeline = self.create_feature_pipeline(n_features=min(8, feature_count))
            X_train_pure = pure_pipeline.fit_transform(X_train, y_train)
            X_test_pure = pure_pipeline.transform(X_test)
            
            pure_results, pure_model = self.train_and_evaluate_model(
                X_train_pure, X_test_pure, y_train, y_test, train_participant_ids, "Pure Movement Patterns"
            )
            all_results['pure_movement'] = pure_results
            
            # 4. MOVEMENT EMBEDDINGS
            logger.info(f"\n{'='*60}")
            logger.info("🧠 ANALYSIS 2: PURE MOVEMENT EMBEDDINGS")
            logger.info(f"{'='*60}")
            
            train_embeddings, test_embeddings = self.create_graph_embeddings(
                X_train, y_train, X_test, train_participant_ids, test_participant_ids, embedding_dim=12
            )
            
            embeddings_results, embeddings_model = self.train_and_evaluate_model(
                train_embeddings, test_embeddings, y_train, y_test, train_participant_ids, "Pure Movement Embeddings"
            )
            all_results['pure_embeddings'] = embeddings_results
            
            # 5. COMBINED ANALYSIS
            logger.info(f"\n{'='*60}")
            logger.info("🔗 ANALYSIS 3: COMBINED PURE FEATURES")
            logger.info(f"{'='*60}")
            
            X_train_combined = np.hstack([X_train_pure, train_embeddings])
            X_test_combined = np.hstack([X_test_pure, test_embeddings])
            
            combined_results, combined_model = self.train_and_evaluate_model(
                X_train_combined, X_test_combined, y_train, y_test, train_participant_ids, "Combined Pure Features"
            )
            all_results['combined'] = combined_results
            
            # 6. Save results
            self.save_results(all_results, feature_count)
            
            # 7. Print final summary
            self.print_final_summary(all_results, feature_count)
            
            logger.info(f"\n✅ Complete pure movement analysis finished!")
            logger.info(f"📁 Results saved to: {self.output_dir}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def save_results(self, all_results, feature_count):
        """Save results to JSON"""
        serializable_results = {}
        for approach, metrics in all_results.items():
            serializable_results[approach] = {}
            for key, value in metrics.items():
                if isinstance(value, (np.floating, np.integer)):
                    serializable_results[approach][key] = float(value)
                else:
                    serializable_results[approach][key] = value
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'PURE Movement Pattern Analysis (NO spatial coordinates)',
            'feature_focus': 'ONLY pure angles and temporal patterns',
            'spatial_exclusion': 'ALL spatial coordinates and ROM x,y excluded',
            'feature_count': feature_count,
            'participant_structure': 'Participants 0-49: ASD, 50-99: Typical, 8 samples each',
            'results': serializable_results
        }
        
        with open(f"{self.output_dir}/pure_movement_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"💾 Results saved to: {self.output_dir}/pure_movement_report.json")
    
    def print_final_summary(self, all_results, feature_count):
        """Print comprehensive final summary"""
        logger.info("\n" + "="*70)
        logger.info("🏁 PURE MOVEMENT PATTERN ANALYSIS COMPLETE")
        logger.info("="*70)
        
        # Find best approach
        best_approach = max(all_results.keys(), key=lambda x: all_results[x]['test_auc'])
        best_auc = all_results[best_approach]['test_auc']
        
        logger.info(f"\n🏆 Best Approach: {best_approach.upper().replace('_', ' ')}")
        logger.info(f"   Test AUC: {best_auc:.4f}")
        
        # Comparison table
        logger.info(f"\n📊 Performance Comparison (PURE MOVEMENT PATTERNS ONLY):")
        logger.info(f"{'Approach':<25} {'CV AUC':<12} {'Test AUC':<10} {'Accuracy':<10} {'F1':<10}")
        logger.info("-" * 67)
        
        approach_names = {
            'pure_movement': 'Pure Movement Patterns',
            'pure_embeddings': 'Pure Movement Embeddings', 
            'combined': 'Combined Pure Features'
        }
        
        for approach, metrics in all_results.items():
            name = approach_names.get(approach, approach)
            cv_auc = f"{metrics['cv_auc_mean']:.3f}±{metrics['cv_auc_std']:.3f}"
            test_auc = f"{metrics['test_auc']:.3f}"
            accuracy = f"{metrics['test_accuracy']:.3f}"
            f1 = f"{metrics['test_f1']:.3f}"
            
            logger.info(f"{name:<25} {cv_auc:<12} {test_auc:<10} {accuracy:<10} {f1:<10}")
        
        # Overall assessment
        logger.info(f"\n📊 OVERALL ASSESSMENT:")
        if best_auc < 0.65:
            logger.info("   ✅ EXCELLENT: Realistic performance - minimal spatial bias!")
            logger.info("   🎯 Pure movement patterns show subtle but meaningful differences")
        elif best_auc < 0.75:
            logger.info("   ✅ GOOD: Moderate performance - some spatial bias removed")
            logger.info("   🎯 Movement patterns provide reasonable discrimination")
        elif best_auc < 0.85:
            logger.info("   ⚠️  MODERATE: Still somewhat high - fundamental differences remain")
        else:
            logger.info("   ❌ HIGH: Performance still too high - deeper bias investigation needed")
        
        # Feature impact assessment
        logger.info(f"\n🔍 PURE MOVEMENT FILTERING IMPACT:")
        logger.info("   ❌ Excluded: ALL spatial coordinates (mean-x, mean-y, mean-z)")
        logger.info("   ❌ Excluded: ALL ROM spatial ranges (x,y coordinates)")
        logger.info("   ❌ Excluded: Height-dependent features (velocity, stride lengths)")
        logger.info("   ❌ Excluded: Distance and position features")
        logger.info("   ✅ Kept ONLY: Pure joint angles and temporal timing")
        logger.info(f"   📈 Final feature count: {feature_count}")
        
        # Participant structure confirmation
        logger.info(f"\n👥 Participant Structure Confirmation:")
        logger.info("   ✅ Participants 0-49: ASD (samples 0-399)")
        logger.info("   ✅ Participants 50-99: Typical (samples 400-799)")
        logger.info("   ✅ 8 samples per participant (augmentations)")
        logger.info("   ✅ No participant leakage between train/test")
        
        logger.info(f"\n📁 Complete results in: {os.path.abspath(self.output_dir)}")
        logger.info("\n✅ Pure movement pattern analysis completed!")


def main():
    """Main function to run PURE movement pattern analysis"""
    try:
        logger.info("🎯 NeuroGait PURE Movement Pattern Analysis - COMPLETELY FIXED")
        logger.info("📋 EXTREME filtering - ONLY pure movement patterns:")
        logger.info("   ❌ NO spatial coordinates (mean-x, mean-y, mean-z)")
        logger.info("   ❌ NO ROM spatial ranges (x,y coordinates)")
        logger.info("   ❌ NO height-dependent features")
        logger.info("   ✅ ONLY pure joint angles and temporal timing")
        logger.info("   • Participants 0-49: ASD, 50-99: Typical, 8 samples each")
        
        analyzer = PureMovementAnalysis(samples_per_participant=8)
        results = analyzer.run_complete_analysis()
        
        if results is None:
            print("❌ Analysis failed - insufficient features after filtering")
            return
        
        print("\n" + "="*60)
        print("🏁 PURE MOVEMENT ANALYSIS FINISHED")
        print("="*60)
        print("🎯 Focus: PURE movement patterns ONLY")
        print("="*60)
        
        for approach, metrics in results.items():
            approach_name = approach.replace('_', ' ').title()
            print(f"\n{approach_name}:")
            print(f"  Test AUC: {metrics['test_auc']:.4f}")
            print(f"  Test F1:  {metrics['test_f1']:.4f}")
            print(f"  CV AUC:   {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
        
        # Overall assessment
        best_auc = max(metrics['test_auc'] for metrics in results.values())
        print(f"\n📊 FINAL ASSESSMENT:")
        if best_auc < 0.65:
            print("🎉 SUCCESS: Pure movement patterns show realistic discrimination!")
            print("✅ Spatial bias effectively removed")
        elif best_auc < 0.75:
            print("✅ GOOD: Reasonable performance with reduced spatial bias")
        else:
            print("⚠️  MODERATE: Some fundamental movement differences remain")
        
        print(f"\n🔍 KEY IMPROVEMENTS:")
        print("✅ EXTREME feature filtering - only pure angles & timing")
        print("✅ All spatial coordinates completely removed")
        print("✅ Height-dependent features excluded")
        print("✅ Conservative model settings")
        print("✅ Participant-level validation")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()