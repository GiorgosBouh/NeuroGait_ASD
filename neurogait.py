<<<<<<< HEAD
#!/usr/bin/env python3
"""
Complete NeuroGait Analysis - Temporal/Angular Features Only
Uses only movement patterns, not spatial positions to avoid height bias
Includes: Raw features, Graph embeddings, Combined analysis
All with proper participant-level splitting (no data leakage)
"""
=======
>>>>>>> de692d289280d9bf55a18121c2af96d558ab4021

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import warnings
import logging
from datetime import datetime

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

<<<<<<< HEAD
class TemporalAngularAnalysis:
    def __init__(self, samples_per_participant=8):
        self.samples_per_participant = samples_per_participant
        self.output_dir = f"temporal_angular_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Angle feature mappings from the documentation
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
        
    def load_temporal_angular_data(self):
        """Load data keeping ONLY temporal and angular features (no spatial positions)"""
        logger.info("📊 Loading data with TEMPORAL/ANGULAR features only...")
        
        # Load data
        df = pd.read_csv('Final dataset.csv', sep=';', decimal=',')
        logger.info(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
        
        # Convert target
        df['class'] = df['class'].map({'A': 1, 'T': 0})
        df = df.rename(columns={'class': 'diagnosis'})
        
        # CRITICAL: Keep ONLY temporal and angular features (NO spatial positions)
        logger.info("\n🎯 Filtering to temporal/angular features only (NO SPATIAL POSITIONS)...")
        
        cols_to_keep = ['diagnosis']
        spatial_excluded = 0
        
        for col in df.columns:
            col_clean = col.strip()
            
            # ❌ EXCLUDE: All spatial coordinate features (mean-x, mean-y, mean-z)
            if col_clean.startswith('mean-') and any(coord in col_clean for coord in ['-x-', '-y-', '-z-']):
                spatial_excluded += 1
                continue
            
            # ✅ INCLUDE: Temporal gait parameters
            elif col_clean in ['GaCT', 'StaT', 'SwiT', 'Velocity']:
                cols_to_keep.append(col)
                logger.info(f"   ✅ Temporal: {col_clean}")
            
            # ✅ INCLUDE: Spatial gait measures (stride length, width) - these are patterns, not positions
            elif col_clean in ['MaxStLe', 'MaxStWi', 'StrLe']:
                cols_to_keep.append(col)
                logger.info(f"   ✅ Spatial pattern: {col_clean}")
            
            # ✅ INCLUDE: Angular features (mean angles between joints)
            elif col_clean.startswith('mean ') and any(angle in col_clean for angle in self.angle_mappings.keys()):
                cols_to_keep.append(col)
                angle_type = next((angle for angle in self.angle_mappings.keys() if angle in col_clean), "unknown")
                logger.info(f"   ✅ Angular: {col_clean} ({angle_type})")
            
            # ✅ INCLUDE: Range of Motion features (movement patterns, not absolute positions)
            elif col_clean.startswith('Rom'):
                cols_to_keep.append(col)
                logger.info(f"   ✅ ROM: {col_clean}")
            
            # ✅ INCLUDE: Distance-based features (relative measurements)
            elif col_clean in ['MaxDBFE', 'MinDBFE', 'Threshold']:
                cols_to_keep.append(col)
                logger.info(f"   ✅ Distance: {col_clean}")
            
            # ✅ INCLUDE: Hand position relative features (binary, less height-dependent)
            elif col_clean in ['HaTiLPos', 'HaTiRPos']:
                cols_to_keep.append(col)
                logger.info(f"   ✅ Relative position: {col_clean}")
        
        # Filter dataset
        df_filtered = df[cols_to_keep]
        
        # Remove constant features
        for col in df_filtered.columns:
            if col != 'diagnosis' and df_filtered[col].nunique() <= 1:
                df_filtered = df_filtered.drop(columns=[col])
        
        logger.info(f"\n📊 FEATURE FILTERING SUMMARY:")
        logger.info(f"   ❌ Excluded spatial coordinates: {spatial_excluded} features")
        logger.info(f"   ✅ Kept temporal/angular/pattern features: {len(df_filtered.columns)-1}")
        logger.info(f"   📈 Data reduction: {spatial_excluded / (len(df.columns)-1) * 100:.1f}% of features removed")
        logger.info(f"   🎯 Focus: Movement patterns, NOT absolute positions")
        logger.info(f"   📊 Class distribution: {df_filtered['diagnosis'].value_counts().to_dict()}")
        
        return df_filtered
=======
def load_and_prepare_data():
    """Load and prepare data with participant structure"""
    logger.info("📊 Loading data...")
    
    # Load data
    df = pd.read_csv('Final dataset.csv', sep=';', decimal=',')
    logger.info(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
>>>>>>> de692d289280d9bf55a18121c2af96d558ab4021
    
<<<<<<< HEAD
    def participant_level_split(self, X, y, test_size=0.2):
        """Split at participant level to prevent leakage"""
        logger.info("\n🔧 Performing participant-level split...")
        
        n_samples = len(X)
        n_participants = n_samples // self.samples_per_participant
        
        # Create participant IDs
        participant_ids = np.repeat(range(n_participants), self.samples_per_participant)
        
        # Get one label per participant
        participant_labels = y[::self.samples_per_participant].values
        
        # Split participants
        train_pids, test_pids = train_test_split(
            range(n_participants), 
            test_size=test_size, 
            stratify=participant_labels, 
            random_state=42
        )
        
        # Get sample indices
        train_mask = np.isin(participant_ids, train_pids)
        test_mask = np.isin(participant_ids, test_pids)
        
        X_train = X[train_mask].reset_index(drop=True)
        X_test = X[test_mask].reset_index(drop=True)
        y_train = y[train_mask].reset_index(drop=True)
        y_test = y[test_mask].reset_index(drop=True)
        
        logger.info(f"✅ Split: {len(train_pids)} train participants ({len(X_train)} samples)")
        logger.info(f"         {len(test_pids)} test participants ({len(X_test)} samples)")
        logger.info(f"   Train class distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"   Test class distribution: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test, train_pids
=======
    # Convert target
    df['class'] = df['class'].map({'A': 1, 'T': 0})
    df = df.rename(columns={'class': 'diagnosis'})
>>>>>>> de692d289280d9bf55a18121c2af96d558ab4021
    
<<<<<<< HEAD
    def create_feature_pipeline(self, n_features=20):
        """Create feature processing pipeline"""
        
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
                          if any(upper_triangle[column] > 0.85)]  # Slightly more permissive
                
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
=======
    # Keep only mean features
    cols_to_keep = ['diagnosis']
    for col in df.columns:
        col_clean = col.strip()
        if (col_clean.startswith('mean-') and any(coord in col_clean for coord in ['-x-', '-y-', '-z-'])) or \
           (col_clean.startswith('mean ') and len(col_clean.split()) >= 2) or \
           col_clean.startswith('Rom') or \
           col_clean in ['MaxStLe', 'MaxStWi', 'StrLe', 'GaCT', 'StaT', 'SwiT', 'Velocity', 'HaTiLPos', 'HaTiRPos', 'MaxDBFE', 'MinDBFE', 'Threshold']:
            cols_to_keep.append(col)
>>>>>>> de692d289280d9bf55a18121c2af96d558ab4021
    
<<<<<<< HEAD
    def create_movement_pattern_graph(self, X_train, y_train, similarity_threshold=0.6):
        """Create graph based on movement pattern similarity (no spatial coords)"""
        logger.info(f"\n🧠 Creating movement pattern similarity graph...")
        
        G = nx.Graph()
        
        # Add participant nodes
        n_participants = len(X_train) // self.samples_per_participant
        
        for i in range(n_participants):
            participant_id = f"P_{i:04d}"
            
            # Get this participant's samples
            start_idx = i * self.samples_per_participant
            end_idx = start_idx + self.samples_per_participant
            
            participant_features = X_train.iloc[start_idx:end_idx].mean()  # Average across augmentations
            participant_label = int(y_train.iloc[start_idx])
            
            # Add node with movement pattern statistics
            feature_stats = {
                'label': participant_label,
                'mean_activity': float(participant_features.mean()),
                'feature_std': float(participant_features.std()),
                'temporal_pattern': float(participant_features.get('Velocity', 0)),
                'node_type': 'participant'
            }
            G.add_node(participant_id, **feature_stats)
=======
    df_filtered = df[cols_to_keep]
    
    # Remove constant features
    for col in df_filtered.columns:
        if col != 'diagnosis' and df_filtered[col].nunique() <= 1:
            df_filtered = df_filtered.drop(columns=[col])
    
    logger.info(f"✅ Kept {len(df_filtered.columns)-1} features")
    return df_filtered

def participant_level_split(X, y, test_size=0.2, samples_per_participant=8):
    """Split at participant level to prevent leakage"""
    logger.info("🔧 Performing participant-level split...")
    
    n_samples = len(X)
    n_participants = n_samples // samples_per_participant
    
    # Create participant IDs
    participant_ids = np.repeat(range(n_participants), samples_per_participant)
    
    # Get one label per participant (they're all the same for each participant)
    participant_labels = y[::samples_per_participant].values
    
    # Split participants
    train_pids, test_pids = train_test_split(
        range(n_participants), 
        test_size=test_size, 
        stratify=participant_labels, 
        random_state=42
    )
    
    # Get sample indices
    train_mask = np.isin(participant_ids, train_pids)
    test_mask = np.isin(participant_ids, test_pids)
    
    X_train = X[train_mask].reset_index(drop=True)
    X_test = X[test_mask].reset_index(drop=True)
    y_train = y[train_mask].reset_index(drop=True)
    y_test = y[test_mask].reset_index(drop=True)
    
    logger.info(f"✅ Split: {len(train_pids)} train participants ({len(X_train)} samples)")
    logger.info(f"         {len(test_pids)} test participants ({len(X_test)} samples)")
    
    return X_train, X_test, y_train, y_test, train_pids

def simple_cv_with_participants(X_train, y_train, train_pids, samples_per_participant=8):
    """Simple participant-level CV"""
    logger.info("🔄 Creating participant-level CV...")
    
    # Get participant labels
    n_train_participants = len(train_pids)
    participant_labels = y_train[::samples_per_participant].values
    
    # Create CV splits
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_p_idx, val_p_idx) in enumerate(skf.split(range(n_train_participants), participant_labels)):
        # Get sample indices for this fold
        train_sample_indices = []
        val_sample_indices = []
        
        for p_idx in train_p_idx:
            start_idx = p_idx * samples_per_participant
            end_idx = start_idx + samples_per_participant
            train_sample_indices.extend(range(start_idx, end_idx))
        
        for p_idx in val_p_idx:
            start_idx = p_idx * samples_per_participant
            end_idx = start_idx + samples_per_participant
            val_sample_indices.extend(range(start_idx, end_idx))
>>>>>>> de692d289280d9bf55a18121c2af96d558ab4021
        
<<<<<<< HEAD
        # Add similarity edges using movement patterns
        logger.info("   Computing movement pattern similarities...")
=======
        # Get fold data
        X_fold_train = X_train.iloc[train_sample_indices]
        X_fold_val = X_train.iloc[val_sample_indices]
        y_fold_train = y_train.iloc[train_sample_indices]
        y_fold_val = y_train.iloc[val_sample_indices]
>>>>>>> de692d289280d9bf55a18121c2af96d558ab4021
        
<<<<<<< HEAD
        # Create participant-level feature matrix (average of augmentations)
        participant_features = []
        for i in range(n_participants):
            start_idx = i * self.samples_per_participant
            end_idx = start_idx + self.samples_per_participant
            participant_avg = X_train.iloc[start_idx:end_idx].mean()
            participant_features.append(participant_avg.values)
        
        participant_features = np.array(participant_features)
        
        # Use k-NN to find similar movement patterns
        knn = NearestNeighbors(n_neighbors=min(8, n_participants//2), metric='cosine')
        knn.fit(participant_features)
        
        distances, indices = knn.kneighbors(participant_features)
        edge_count = 0
=======
        # Train fold model
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=2.0, reg_lambda=2.0,
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        )
        model.fit(X_fold_train, y_fold_train)
>>>>>>> de692d289280d9bf55a18121c2af96d558ab4021
        
<<<<<<< HEAD
        for i, (neighbors, dists) in enumerate(zip(indices, distances)):
            participant_i = f"P_{i:04d}"
            for j, dist in zip(neighbors, dists):
                if i != j and (1 - dist) > similarity_threshold:
                    participant_j = f"P_{j:04d}"
                    similarity = 1 - dist
                    G.add_edge(participant_i, participant_j, 
                             weight=similarity, 
                             connection_type='movement_similarity')
                    edge_count += 1
=======
        # Evaluate
        y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
        fold_auc = roc_auc_score(y_fold_val, y_pred_proba)
        cv_scores.append(fold_auc)
>>>>>>> de692d289280d9bf55a18121c2af96d558ab4021
        
<<<<<<< HEAD
        logger.info(f"   Added {edge_count} movement pattern similarity edges")
        logger.info(f"   Movement graph: {G.number_of_nodes()} participants, {G.number_of_edges()} edges")
        
        return G
=======
        logger.info(f"   Fold {fold+1}: {len(train_p_idx)} train participants, {len(val_p_idx)} val participants, AUC: {fold_auc:.4f}")
>>>>>>> de692d289280d9bf55a18121c2af96d558ab4021
    
<<<<<<< HEAD
    def create_graph_embeddings(self, X_train, y_train, X_test, embedding_dim=24):
        """Create graph embeddings based on movement patterns"""
        logger.info(f"\n🧠 Creating movement pattern embeddings (dim={embedding_dim})...")
        
        try:
            # Create movement pattern graph
            movement_graph = self.create_movement_pattern_graph(X_train, y_train)
            
            # Generate embeddings if graph has edges
            if movement_graph.number_of_edges() > 0:
                logger.info("   Running Node2Vec on movement patterns...")
                
                node2vec = Node2Vec(
                    movement_graph,
                    dimensions=embedding_dim,
                    walk_length=20,
                    num_walks=40,
                    p=1.0,
                    q=1.0,
                    workers=1,
                    quiet=True
                )
                
                model = node2vec.fit(window=5, min_count=1, batch_words=4, epochs=10)
                
                # Get embeddings for training participants
                n_train_participants = len(X_train) // self.samples_per_participant
                participant_embeddings = np.zeros((n_train_participants, embedding_dim))
                
                for i in range(n_train_participants):
                    participant_id = f"P_{i:04d}"
                    if participant_id in model.wv:
                        participant_embeddings[i] = model.wv[participant_id]
                    else:
                        participant_embeddings[i] = np.random.normal(0, 0.01, embedding_dim)
                
                # Replicate embeddings for all samples (8 per participant)
                train_embeddings = np.repeat(participant_embeddings, self.samples_per_participant, axis=0)
                
                # For test set: project using k-NN from training movement patterns
                logger.info("   Projecting test embeddings using movement pattern similarity...")
                
                n_test_participants = len(X_test) // self.samples_per_participant
                test_participant_features = []
                
                for i in range(n_test_participants):
                    start_idx = i * self.samples_per_participant
                    end_idx = start_idx + self.samples_per_participant
                    participant_avg = X_test.iloc[start_idx:end_idx].mean()
                    test_participant_features.append(participant_avg.values)
                
                test_participant_features = np.array(test_participant_features)
                
                # Find similar training participants
                train_participant_features = []
                for i in range(n_train_participants):
                    start_idx = i * self.samples_per_participant
                    end_idx = start_idx + self.samples_per_participant
                    participant_avg = X_train.iloc[start_idx:end_idx].mean()
                    train_participant_features.append(participant_avg.values)
                
                train_participant_features = np.array(train_participant_features)
                
                knn = NearestNeighbors(n_neighbors=min(5, n_train_participants), metric='cosine')
                knn.fit(train_participant_features)
                
                test_distances, test_indices = knn.kneighbors(test_participant_features)
                test_participant_embeddings = np.zeros((n_test_participants, embedding_dim))
                
                for i, (neighbors, dists) in enumerate(zip(test_indices, test_distances)):
                    weights = 1 / (dists + 1e-8)
                    weights = weights / weights.sum()
                    test_participant_embeddings[i] = np.average(participant_embeddings[neighbors], axis=0, weights=weights)
                
                # Replicate for all test samples
                test_embeddings = np.repeat(test_participant_embeddings, self.samples_per_participant, axis=0)
                
                logger.info(f"✅ Created movement embeddings: train {train_embeddings.shape}, test {test_embeddings.shape}")
                
            else:
                logger.warning("   No edges in movement graph, using random embeddings")
                train_embeddings = np.random.normal(0, 0.01, (len(X_train), embedding_dim))
                test_embeddings = np.random.normal(0, 0.01, (len(X_test), embedding_dim))
            
            return train_embeddings, test_embeddings
            
        except Exception as e:
            logger.error(f"❌ Graph embedding failed: {str(e)}")
            logger.info("   Using random embeddings as fallback")
            train_embeddings = np.random.normal(0, 0.01, (len(X_train), embedding_dim))
            test_embeddings = np.random.normal(0, 0.01, (len(X_test), embedding_dim))
            return train_embeddings, test_embeddings
=======
    return cv_scores

def train_final_model(X_train, X_test, y_train, y_test):
    """Train final model"""
    logger.info("🚀 Training final model...")
>>>>>>> de692d289280d9bf55a18121c2af96d558ab4021
    
<<<<<<< HEAD
    def participant_cv_scores(self, X_train, y_train, train_pids, model_class=None):
        """Get CV scores at participant level"""
        n_train_participants = len(train_pids)
        participant_labels = y_train[::self.samples_per_participant].values
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_p_idx, val_p_idx in skf.split(range(n_train_participants), participant_labels):
            # Get sample indices
            train_sample_indices = []
            val_sample_indices = []
            
            for p_idx in train_p_idx:
                start_idx = p_idx * self.samples_per_participant
                end_idx = start_idx + self.samples_per_participant
                train_sample_indices.extend(range(start_idx, end_idx))
            
            for p_idx in val_p_idx:
                start_idx = p_idx * self.samples_per_participant
                end_idx = start_idx + self.samples_per_participant
                val_sample_indices.extend(range(start_idx, end_idx))
            
            X_fold_train = X_train[train_sample_indices] if isinstance(X_train, np.ndarray) else X_train.iloc[train_sample_indices]
            X_fold_val = X_train[val_sample_indices] if isinstance(X_train, np.ndarray) else X_train.iloc[val_sample_indices]
            y_fold_train = y_train.iloc[train_sample_indices]
            y_fold_val = y_train.iloc[val_sample_indices]
            
            # Train model
            if model_class is None:
                model = xgb.XGBClassifier(
                    n_estimators=50, max_depth=3, learning_rate=0.03,
                    subsample=0.7, colsample_bytree=0.7,
                    reg_alpha=3.0, reg_lambda=3.0,
                    random_state=42, use_label_encoder=False, eval_metric='logloss'
                )
            else:
                model = model_class
            
            model.fit(X_fold_train, y_fold_train)
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            fold_auc = roc_auc_score(y_fold_val, y_pred_proba)
            cv_scores.append(fold_auc)
        
        return cv_scores
    
    def train_and_evaluate_model(self, X_train, X_test, y_train, y_test, train_pids, model_name="Model"):
        """Train and evaluate model with participant-level CV"""
        logger.info(f"\n🚀 Training {model_name}...")
        logger.info(f"   Training set: {X_train.shape}")
        logger.info(f"   Test set: {X_test.shape}")
=======
    # Feature selection and scaling
    # Remove highly correlated features
    corr_matrix = X_train.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]
    
    X_train_decorr = X_train.drop(columns=to_drop)
    X_test_decorr = X_test.drop(columns=to_drop)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_decorr)
    X_test_scaled = scaler.transform(X_test_decorr)
    
    # Select features
    selector = SelectKBest(f_classif, k=min(25, X_train_scaled.shape[1]))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    logger.info(f"   Selected {X_train_selected.shape[1]} features after processing")
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=2.0, reg_lambda=2.0,
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    )
    model.fit(X_train_selected, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_selected)
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
    
    metrics = {
        'test_auc': roc_auc_score(y_test, y_pred_proba),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred),
        'test_recall': recall_score(y_test, y_pred),
        'test_f1': f1_score(y_test, y_pred),
    }
    
    return metrics

def main():
    """Main analysis"""
    logger.info("🔍 Starting SIMPLE Fixed NeuroGait Analysis")
    
    try:
        # Load data
        df = load_and_prepare_data()
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']
>>>>>>> de692d289280d9bf55a18121c2af96d558ab4021
        
<<<<<<< HEAD
        # Get CV scores
        cv_scores = self.participant_cv_scores(X_train, y_train, train_pids)
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
=======
        # Participant-level split
        X_train, X_test, y_train, y_test, train_pids = participant_level_split(X, y)
>>>>>>> de692d289280d9bf55a18121c2af96d558ab4021
        
<<<<<<< HEAD
        logger.info(f"   Participant-level CV AUC: {cv_mean:.4f} ± {cv_std:.4f}")
=======
        # Participant-level CV
        cv_scores = simple_cv_with_participants(X_train, y_train, train_pids)
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
>>>>>>> de692d289280d9bf55a18121c2af96d558ab4021
        
<<<<<<< HEAD
=======
        logger.info(f"   CV AUC: {cv_mean:.4f} ± {cv_std:.4f}")
        
>>>>>>> de692d289280d9bf55a18121c2af96d558ab4021
        # Train final model
<<<<<<< HEAD
        model = xgb.XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=3.0, reg_lambda=3.0,
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        )
        model.fit(X_train, y_train)
=======
        metrics = train_final_model(X_train, X_test, y_train, y_test)
>>>>>>> de692d289280d9bf55a18121c2af96d558ab4021
        
<<<<<<< HEAD
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
=======
        # Results
        logger.info("\n📊 FINAL RESULTS (NO PARTICIPANT LEAKAGE):")
        logger.info(f"   CV AUC:      {cv_mean:.4f} ± {cv_std:.4f}")
>>>>>>> de692d289280d9bf55a18121c2af96d558ab4021
        logger.info(f"   Test AUC:    {metrics['test_auc']:.4f}")
        logger.info(f"   Accuracy:    {metrics['test_accuracy']:.4f}")
        logger.info(f"   Precision:   {metrics['test_precision']:.4f}")
        logger.info(f"   Recall:      {metrics['test_recall']:.4f}")
        logger.info(f"   F1-score:    {metrics['test_f1']:.4f}")
        
<<<<<<< HEAD
        # Performance assessment
        if metrics['test_auc'] > 0.85:
            logger.warning("   ⚠️  Still high performance")
        elif metrics['test_auc'] > 0.75:
            logger.info("   ✅ Good performance")
        elif metrics['test_auc'] > 0.65:
            logger.info("   ✅ Realistic performance")
=======
        if metrics['test_auc'] < 0.85:
            logger.info("✅ Realistic performance achieved!")
>>>>>>> de692d289280d9bf55a18121c2af96d558ab4021
        else:
<<<<<<< HEAD
            logger.info("   ℹ️  Lower performance")
=======
            logger.warning("⚠️  Still high - may have other issues")
>>>>>>> de692d289280d9bf55a18121c2af96d558ab4021
        
<<<<<<< HEAD
        return metrics, model
    
    def run_complete_analysis(self):
        """Run complete analysis: Raw + Embeddings + Combined"""
        logger.info(f"\n🔍 Starting Complete Temporal/Angular Analysis - {datetime.now()}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🎯 Focus: MOVEMENT PATTERNS only (no spatial positions)")
=======
        return metrics
>>>>>>> de692d289280d9bf55a18121c2af96d558ab4021
        
<<<<<<< HEAD
        try:
            # 1. Load temporal/angular data only
            df = self.load_temporal_angular_data()
            
            # 2. Participant-level split
            X = df.drop('diagnosis', axis=1)
            y = df['diagnosis']
            
            X_train, X_test, y_train, y_test, train_pids = self.participant_level_split(X, y)
            
            # Storage for results
            all_results = {}
            
            # 3. RAW TEMPORAL/ANGULAR FEATURES
            logger.info(f"\n{'='*60}")
            logger.info("🔍 ANALYSIS 1: RAW TEMPORAL/ANGULAR FEATURES")
            logger.info(f"{'='*60}")
            
            raw_pipeline = self.create_feature_pipeline(n_features=15)
            X_train_raw = raw_pipeline.fit_transform(X_train, y_train)
            X_test_raw = raw_pipeline.transform(X_test)
            
            raw_results, raw_model = self.train_and_evaluate_model(
                X_train_raw, X_test_raw, y_train, y_test, train_pids, "Raw Temporal/Angular"
            )
            all_results['raw_temporal_angular'] = raw_results
            
            # 4. MOVEMENT PATTERN EMBEDDINGS
            logger.info(f"\n{'='*60}")
            logger.info("🧠 ANALYSIS 2: MOVEMENT PATTERN EMBEDDINGS")
            logger.info(f"{'='*60}")
            
            train_embeddings, test_embeddings = self.create_graph_embeddings(
                X_train, y_train, X_test, embedding_dim=20
            )
            
            embeddings_results, embeddings_model = self.train_and_evaluate_model(
                train_embeddings, test_embeddings, y_train, y_test, train_pids, "Movement Pattern Embeddings"
            )
            all_results['movement_embeddings'] = embeddings_results
            
            # 5. COMBINED ANALYSIS
            logger.info(f"\n{'='*60}")
            logger.info("🔗 ANALYSIS 3: COMBINED (TEMPORAL/ANGULAR + EMBEDDINGS)")
            logger.info(f"{'='*60}")
            
            X_train_combined = np.hstack([X_train_raw, train_embeddings])
            X_test_combined = np.hstack([X_test_raw, test_embeddings])
            
            combined_results, combined_model = self.train_and_evaluate_model(
                X_train_combined, X_test_combined, y_train, y_test, train_pids, "Combined Features"
            )
            all_results['combined'] = combined_results
            
            # 6. Save results
            self.save_results(all_results)
            
            # 7. Print final summary
            self.print_final_summary(all_results)
            
            logger.info(f"\n✅ Complete temporal/angular analysis finished!")
            logger.info(f"📁 Results saved to: {self.output_dir}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def save_results(self, all_results):
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
            'analysis_type': 'Complete Temporal/Angular Analysis (No Spatial Positions)',
            'feature_focus': 'Movement patterns, angles, temporal features only',
            'spatial_exclusion': 'All mean-x, mean-y, mean-z coordinates excluded',
            'results': serializable_results
        }
        
        with open(f"{self.output_dir}/temporal_angular_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"💾 Results saved to: {self.output_dir}/temporal_angular_report.json")
    
    def print_final_summary(self, all_results):
        """Print comprehensive final summary"""
        logger.info("\n" + "="*70)
        logger.info("🏁 TEMPORAL/ANGULAR ANALYSIS COMPLETE")
        logger.info("="*70)
        
        # Find best approach
        best_approach = max(all_results.keys(), key=lambda x: all_results[x]['test_auc'])
        best_auc = all_results[best_approach]['test_auc']
        
        logger.info(f"\n🏆 Best Approach: {best_approach.upper().replace('_', ' ')}")
        logger.info(f"   Test AUC: {best_auc:.4f}")
        
        # Comparison table
        logger.info(f"\n📊 Performance Comparison (MOVEMENT PATTERNS ONLY):")
        logger.info(f"{'Approach':<25} {'CV AUC':<12} {'Test AUC':<10} {'Accuracy':<10} {'F1':<10}")
        logger.info("-" * 67)
        
        approach_names = {
            'raw_temporal_angular': 'Raw Temporal/Angular',
            'movement_embeddings': 'Movement Embeddings', 
            'combined': 'Combined'
        }
        
        for approach, metrics in all_results.items():
            name = approach_names.get(approach, approach)
            cv_auc = f"{metrics['cv_auc_mean']:.3f}±{metrics['cv_auc_std']:.3f}"
            test_auc = f"{metrics['test_auc']:.3f}"
            accuracy = f"{metrics['test_accuracy']:.3f}"
            f1 = f"{metrics['test_f1']:.3f}"
            
            logger.info(f"{name:<25} {cv_auc:<12} {test_auc:<10} {accuracy:<10} {f1:<10}")
        
        # Insights
        logger.info(f"\n💡 Key Insights:")
        
        raw_auc = all_results['raw_temporal_angular']['test_auc']
        embeddings_auc = all_results['movement_embeddings']['test_auc']
        combined_auc = all_results['combined']['test_auc']
        
        if combined_auc > max(raw_auc, embeddings_auc):
            logger.info("   ✅ Combined approach shows best performance")
        elif embeddings_auc > raw_auc:
            logger.info("   🧠 Movement embeddings outperform raw features")
        else:
            logger.info("   📊 Raw temporal/angular features perform best")
        
        if max(raw_auc, embeddings_auc, combined_auc) < 0.85:
            logger.info("   ✅ More realistic performance levels achieved!")
            logger.info("   🎯 Movement patterns provide meaningful but not perfect discrimination")
        elif max(raw_auc, embeddings_auc, combined_auc) < 0.9:
            logger.info("   ⚠️  Still high but more reasonable than spatial features")
        else:
            logger.info("   ⚠️  High performance persists - may indicate fundamental group differences")
        
        # Feature impact assessment
        logger.info(f"\n🔍 Spatial Position Impact Assessment:")
        logger.info("   ❌ Excluded: All mean-x, mean-y, mean-z coordinates")
        logger.info("   ✅ Focused on: Movement patterns, timing, angles")
        logger.info("   📈 Expected: More realistic performance vs spatial features")
        
        logger.info(f"\n📁 Complete results in: {os.path.abspath(self.output_dir)}")
        logger.info("\n✅ Temporal/Angular analysis completed!")


def main():
    """Main function to run complete temporal/angular analysis"""
    try:
        analyzer = TemporalAngularAnalysis(samples_per_participant=8)
        results = analyzer.run_complete_analysis()
        
        print("\n" + "="*60)
        print("🏁 TEMPORAL/ANGULAR ANALYSIS FINISHED")
        print("="*60)
        print("🎯 Focus: MOVEMENT PATTERNS ONLY (no spatial positions)")
        print("="*60)
        
        for approach, metrics in results.items():
            approach_name = approach.replace('_', ' ').title()
            print(f"\n{approach_name}:")
            print(f"  Test AUC: {metrics['test_auc']:.4f}")
            print(f"  Test F1:  {metrics['test_f1']:.4f}")
            print(f"  CV AUC:   {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
        
        # Overall assessment
        best_auc = max(metrics['test_auc'] for metrics in results.values())
        print(f"\n📊 OVERALL ASSESSMENT:")
        if best_auc < 0.8:
            print("✅ REALISTIC: Movement patterns show meaningful but realistic discrimination")
        elif best_auc < 0.9:
            print("⚠️  MODERATE: Still somewhat high, but better than spatial features")
        else:
            print("❌ HIGH: Performance still too high - fundamental group differences remain")
        
        return results
        
=======
>>>>>>> de692d289280d9bf55a18121c2af96d558ab4021
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
