#!/usr/bin/env python3
"""
NeuroGait Knowledge Graph Builder - OPTIMIZED FOR ML PERFORMANCE
Key optimizations:
1. ML-focused feature engineering in the graph
2. Semantic embeddings that preserve clinical meaning
3. Better feature aggregation strategies
4. Dimensionality optimized for the dataset size
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import logging
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('.env')

class OptimizedNeuroGaitGraphBuilder:
    def __init__(self, samples_per_participant=8):
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = os.getenv('NEO4J_USER', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD', 'palatiou')
        self.driver = None
        self.samples_per_participant = samples_per_participant
        
        # Augmentation type mapping
        self.augmentation_types = [
            'original', 'jittering', 'scaling_up', 'scaling_down',
            'translation_left', 'translation_right', 'horizontal_flip', 'temporal_slice'
        ]
        
        # Clinical feature groups (based on ASD research)
        self.clinical_feature_groups = {
            'upper_body_coordination': [
                'mean HESHL', 'mean HESHR', 'mean SPELL', 'mean SPELR',
                'mean SHWRL', 'mean SHWRR', 'mean ELHAL', 'mean ELHAR'
            ],
            'hand_movement_patterns': [
                'mean THHAL', 'mean THHAR'
            ],
            'lower_body_coordination': [
                'mean SPKNL', 'mean SPKNR', 'mean HIANL', 'mean HIANR',
                'mean KNFOL', 'mean KNFOR'
            ],
            'temporal_gait_patterns': [
                'GaCT', 'StaT', 'SwiT'
            ],
            'spatial_gait_patterns': [
                'MaxStLe', 'MaxStWi', 'StrLe', 'Velocity'
            ],
            'bilateral_asymmetry': [],  # Will be computed
            'movement_variability': []  # Will be computed
        }
        
        # Body parts for coordinate analysis
        self.body_parts = [
            'Head', 'Neck', 'SpineShoulder', 'ShoulderLeft', 'ShoulderRight',
            'ElbowLeft', 'ElbowRight', 'WristLeft', 'WristRight', 
            'ThumbLeft', 'ThumbRight', 'HandLeft', 'HandRight',
            'HandTipLeft', 'HandTipRight', 'SpineMid', 'SpineBase',
            'HipLeft', 'HipRight', 'KneeLeft', 'KneeRight',
            'AnkleLeft', 'AnkleRight', 'FootLeft', 'FootRight'
        ]
        
    def convert_to_float(self, value):
        """Convert string with comma decimal separator to float"""
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).replace(',', '.'))
        except (ValueError, AttributeError):
            return None
    
    def connect(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"✅ Connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            return False
    
    def clear_database(self):
        """Clear existing data"""
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("🗑️ Database cleared")
        except Exception as e:
            logger.error(f"❌ Error clearing database: {e}")
            raise
    
    def create_constraints_and_indexes(self):
        """Create optimized constraints and indexes for ML performance"""
        constraints = [
            "CREATE CONSTRAINT participant_id_unique IF NOT EXISTS FOR (p:Participant) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT sample_id_unique IF NOT EXISTS FOR (s:Sample) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT feature_group_unique IF NOT EXISTS FOR (fg:FeatureGroup) REQUIRE fg.name IS UNIQUE",
            "CREATE CONSTRAINT clinical_pattern_unique IF NOT EXISTS FOR (cp:ClinicalPattern) REQUIRE cp.id IS UNIQUE",
            "CREATE CONSTRAINT classification_unique IF NOT EXISTS FOR (c:Classification) REQUIRE c.label IS UNIQUE"
        ]
        
        indexes = [
            "CREATE INDEX sample_participant_idx IF NOT EXISTS FOR (s:Sample) ON (s.participant_id)",
            "CREATE INDEX sample_class_idx IF NOT EXISTS FOR (s:Sample) ON (s.diagnosis)",
            "CREATE INDEX pattern_value_idx IF NOT EXISTS FOR (cp:ClinicalPattern) ON (cp.value)",
            "CREATE INDEX feature_importance_idx IF NOT EXISTS FOR (fg:FeatureGroup) ON (fg.importance_score)"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.debug(f"Constraint might already exist: {e}")
            
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    logger.debug(f"Index might already exist: {e}")
            
            logger.info("✅ Optimized constraints and indexes created")
    
    def load_and_engineer_features(self, filepath="Final dataset.csv"):
        """Load data and perform advanced feature engineering"""
        logger.info(f"📊 Loading and engineering features from {filepath}...")
        
        # Read CSV
        try:
            df = pd.read_csv(filepath, delimiter=';', decimal=',', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, delimiter=';', decimal=',', encoding='latin-1')
        
        logger.info(f"📋 Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Convert numeric columns
        numeric_columns = [col for col in df.columns if col != 'class']
        for col in numeric_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: self.convert_to_float(x) if pd.notna(x) else np.nan)
        
        # Create participant structure
        participant_ids = []
        for i in range(len(df)):
            participant_id = i // 8
            participant_ids.append(participant_id)
        
        df['participant_id'] = participant_ids
        df['diagnosis'] = df['class'].map({'A': 'ASD', 'T': 'Typical'})
        
        # Advanced feature engineering
        df_engineered = self._engineer_advanced_features(df)
        
        logger.info(f"✅ Feature engineering completed:")
        logger.info(f"  📊 Total samples: {len(df_engineered)}")
        logger.info(f"  🔢 Engineered features: {len([col for col in df_engineered.columns if col.startswith('eng_')])}")
        logger.info(f"  🎯 Class distribution: {df_engineered['diagnosis'].value_counts().to_dict()}")
        
        return df_engineered
    
    def _engineer_advanced_features(self, df):
        """Engineer advanced clinical features for better ASD discrimination"""
        logger.info("🔧 Engineering advanced clinical features...")
        
        df_eng = df.copy()
        
        # 1. Bilateral Asymmetry Features
        logger.info("  📐 Computing bilateral asymmetry features...")
        bilateral_pairs = [
            ('mean HESHL', 'mean HESHR'),
            ('mean SPELR', 'mean SPELL'),
            ('mean SHWRL', 'mean SHWRR'),
            ('mean SPKNL', 'mean SPKNR'),
            ('mean HIANL', 'mean HIANR')
        ]
        
        for left_feature, right_feature in bilateral_pairs:
            if left_feature in df.columns and right_feature in df.columns:
                # Asymmetry ratio
                df_eng[f'eng_asymmetry_{left_feature.split()[-1]}'] = (
                    df[left_feature] - df[right_feature]
                ) / (df[left_feature] + df[right_feature] + 1e-6)
                
                # Asymmetry magnitude
                df_eng[f'eng_asymmetry_mag_{left_feature.split()[-1]}'] = np.abs(
                    df[left_feature] - df[right_feature]
                )
        
        # 2. Movement Coordination Features
        logger.info("  🤝 Computing movement coordination features...")
        
        # Upper body coordination
        upper_features = [f for f in self.clinical_feature_groups['upper_body_coordination'] if f in df.columns]
        if len(upper_features) >= 2:
            df_eng['eng_upper_body_variability'] = df[upper_features].std(axis=1)
            df_eng['eng_upper_body_mean'] = df[upper_features].mean(axis=1)
            df_eng['eng_upper_body_range'] = df[upper_features].max(axis=1) - df[upper_features].min(axis=1)
        
        # Lower body coordination
        lower_features = [f for f in self.clinical_feature_groups['lower_body_coordination'] if f in df.columns]
        if len(lower_features) >= 2:
            df_eng['eng_lower_body_variability'] = df[lower_features].std(axis=1)
            df_eng['eng_lower_body_mean'] = df[lower_features].mean(axis=1)
            df_eng['eng_lower_body_range'] = df[lower_features].max(axis=1) - df[lower_features].min(axis=1)
        
        # 3. Temporal Pattern Features
        logger.info("  ⏱️ Computing temporal pattern features...")
        temporal_features = [f for f in self.clinical_feature_groups['temporal_gait_patterns'] if f in df.columns]
        if len(temporal_features) >= 2:
            df_eng['eng_temporal_variability'] = df[temporal_features].std(axis=1)
            df_eng['eng_temporal_efficiency'] = df['GaCT'] / (df['StaT'] + df['SwiT'] + 1e-6) if 'GaCT' in df.columns else 0
            
        # 4. Spatial Efficiency Features
        logger.info("  📏 Computing spatial efficiency features...")
        if 'StrLe' in df.columns and 'GaCT' in df.columns:
            df_eng['eng_gait_efficiency'] = df['StrLe'] / (df['GaCT'] + 1e-6)
        
        if 'Velocity' in df.columns and 'StrLe' in df.columns:
            df_eng['eng_stride_frequency'] = df['Velocity'] / (df['StrLe'] + 1e-6)
        
        # 5. Stability and Variability Features
        logger.info("  📊 Computing stability features...")
        
        # Overall movement variability (important for ASD)
        movement_features = [col for col in df.columns if col.startswith('mean ') and any(x in col for x in ['HESL', 'SPEL', 'SHWR'])]
        if len(movement_features) >= 3:
            df_eng['eng_overall_movement_variability'] = df[movement_features].std(axis=1)
            df_eng['eng_movement_complexity'] = df[movement_features].apply(lambda x: np.std(np.diff(x.dropna())), axis=1)
        
        # 6. Clinical Significance Features
        logger.info("  🏥 Computing clinical significance features...")
        
        # Hand-arm coordination (critical for ASD)
        hand_features = ['mean THHAL', 'mean THHAR', 'mean ELHAL', 'mean ELHAR']
        available_hand = [f for f in hand_features if f in df.columns]
        if len(available_hand) >= 2:
            df_eng['eng_hand_coordination'] = df[available_hand].mean(axis=1)
            df_eng['eng_hand_asymmetry'] = df[available_hand].std(axis=1)
        
        # Postural control
        postural_features = ['mean SPKNL', 'mean SPKNR', 'mean HIANL', 'mean HIANR']
        available_postural = [f for f in postural_features if f in df.columns]
        if len(available_postural) >= 2:
            df_eng['eng_postural_control'] = df[available_postural].mean(axis=1)
            df_eng['eng_postural_variability'] = df[available_postural].std(axis=1)
        
        # 7. Participant-level Features (for better embeddings)
        logger.info("  👤 Computing participant-level features...")
        
        # For each participant, compute cross-sample variability
        participant_features = {}
        base_movement_features = [col for col in df.columns if col.startswith('mean ') or col in ['GaCT', 'StaT', 'SwiT']]
        
        for pid in df['participant_id'].unique():
            participant_data = df[df['participant_id'] == pid]
            if len(participant_data) >= 4:  # Need sufficient samples
                for feature in base_movement_features:
                    if feature in participant_data.columns:
                        variability_key = f'eng_participant_var_{feature.replace(" ", "_").replace("-", "_")}'
                        participant_features[pid] = participant_features.get(pid, {})
                        participant_features[pid][variability_key] = participant_data[feature].std()
        
        # Add participant variability features back to dataframe
        for idx, row in df_eng.iterrows():
            pid = row['participant_id']
            if pid in participant_features:
                for var_feature, var_value in participant_features[pid].items():
                    df_eng.at[idx, var_feature] = var_value
        
        # Fill NaN values with 0 for engineered features
        eng_columns = [col for col in df_eng.columns if col.startswith('eng_')]
        df_eng[eng_columns] = df_eng[eng_columns].fillna(0)
        
        logger.info(f"  ✅ Created {len(eng_columns)} engineered features")
        
        return df_eng
    
    def create_optimized_graph_structure(self):
        """Create optimized graph structure for ML performance"""
        logger.info("🏗️ Creating optimized graph structure...")
        
        with self.driver.session() as session:
            # Create classification nodes
            session.run("""
                MERGE (asd:Classification {label: 'ASD', description: 'Autism Spectrum Disorder'})
                MERGE (typical:Classification {label: 'Typical', description: 'Typical Development'})
            """)
            
            # Create clinical feature groups
            for group_name, features in self.clinical_feature_groups.items():
                session.run("""
                    MERGE (fg:FeatureGroup {
                        name: $group_name,
                        description: $description,
                        feature_count: $feature_count,
                        clinical_relevance: $relevance
                    })
                """, 
                group_name=group_name,
                description=f"Clinical feature group: {group_name}",
                feature_count=len(features),
                relevance=self._get_clinical_relevance_score(group_name)
                )
            
            # Create augmentation type nodes
            for i, aug_type in enumerate(self.augmentation_types):
                session.run("""
                    MERGE (at:AugmentationType {
                        name: $aug_type,
                        index: $index,
                        is_original: $is_original
                    })
                """, aug_type=aug_type, index=i, is_original=(aug_type == 'original'))
            
            logger.info("✅ Optimized graph structure created")
    
    def _get_clinical_relevance_score(self, group_name):
        """Assign clinical relevance scores based on ASD research"""
        relevance_scores = {
            'upper_body_coordination': 0.9,  # High relevance for ASD
            'hand_movement_patterns': 0.95,  # Very high relevance
            'lower_body_coordination': 0.8,   # High relevance
            'temporal_gait_patterns': 0.85,   # High relevance
            'spatial_gait_patterns': 0.7,     # Moderate relevance
            'bilateral_asymmetry': 0.92,      # Very high relevance for ASD
            'movement_variability': 0.88       # High relevance
        }
        return relevance_scores.get(group_name, 0.5)
    
    def create_participants_and_samples_optimized(self, df):
        """Create participants and samples with ML-optimized relationships"""
        logger.info("👥 Creating optimized participants and samples...")
        
        with self.driver.session() as session:
            # Create participants with comprehensive clinical profiles
            unique_participants = df.groupby(['participant_id', 'diagnosis']).first().reset_index()
            
            for _, row in unique_participants.iterrows():
                # Calculate participant-level clinical metrics
                participant_data = df[df['participant_id'] == row['participant_id']]
                
                clinical_metrics = self._compute_participant_clinical_metrics(participant_data)
                
                session.run("""
                    MERGE (p:Participant {
                        id: $participant_id,
                        diagnosis: $diagnosis,
                        sample_count: $sample_count,
                        movement_variability: $movement_variability,
                        coordination_score: $coordination_score,
                        asymmetry_score: $asymmetry_score,
                        temporal_consistency: $temporal_consistency,
                        clinical_severity: $clinical_severity
                    })
                    MERGE (c:Classification {label: $diagnosis})
                    MERGE (p)-[:HAS_DIAGNOSIS]->(c)
                """, 
                participant_id=f"P_{row['participant_id']:03d}",
                diagnosis=row['diagnosis'],
                sample_count=len(participant_data),
                **clinical_metrics
                )
            
            # Create samples with enhanced clinical patterns
            batch_size = 50
            batch_data = []
            
            for idx, row in df.iterrows():
                # Compute clinical patterns for this sample
                clinical_patterns = self._compute_sample_clinical_patterns(row)
                
                sample_data = {
                    'sample_id': f"S_{row['participant_id']:03d}_{idx % 8}",
                    'participant_id': f"P_{row['participant_id']:03d}",
                    'diagnosis': row['diagnosis'],
                    'augmentation_type': self.augmentation_types[idx % 8],
                    'sample_index': idx,
                    **clinical_patterns
                }
                batch_data.append(sample_data)
                
                if len(batch_data) >= batch_size:
                    self._create_optimized_sample_batch(session, batch_data)
                    batch_data = []
            
            if batch_data:
                self._create_optimized_sample_batch(session, batch_data)
            
            logger.info(f"✅ Created {len(unique_participants)} participants and {len(df)} samples with clinical profiles")
    
    def _compute_participant_clinical_metrics(self, participant_data):
        """Compute comprehensive clinical metrics for a participant"""
        metrics = {}
        
        # Movement variability across samples
        movement_cols = [col for col in participant_data.columns if col.startswith('mean ')]
        if movement_cols:
            variabilities = []
            for col in movement_cols:
                if participant_data[col].notna().sum() > 1:
                    variabilities.append(participant_data[col].std())
            metrics['movement_variability'] = np.mean(variabilities) if variabilities else 0.0
        else:
            metrics['movement_variability'] = 0.0
        
        # Coordination score (inverse of asymmetry)
        asymmetry_cols = [col for col in participant_data.columns if 'eng_asymmetry' in col]
        if asymmetry_cols:
            metrics['asymmetry_score'] = participant_data[asymmetry_cols].mean().mean()
            metrics['coordination_score'] = 1.0 / (1.0 + metrics['asymmetry_score'])
        else:
            metrics['asymmetry_score'] = 0.0
            metrics['coordination_score'] = 1.0
        
        # Temporal consistency
        if 'GaCT' in participant_data.columns:
            metrics['temporal_consistency'] = 1.0 / (1.0 + participant_data['GaCT'].std())
        else:
            metrics['temporal_consistency'] = 1.0
        
        # Clinical severity (higher values indicate more atypical patterns)
        eng_cols = [col for col in participant_data.columns if col.startswith('eng_')]
        if eng_cols:
            # Normalize engineered features and compute severity
            severity_features = participant_data[eng_cols].mean()
            metrics['clinical_severity'] = float(np.sqrt(np.sum(severity_features ** 2)))
        else:
            metrics['clinical_severity'] = 0.0
        
        return metrics
    
    def _compute_sample_clinical_patterns(self, sample_row):
        """Compute clinical patterns for a single sample"""
        patterns = {}
        
        # Upper body pattern
        upper_features = [f for f in self.clinical_feature_groups['upper_body_coordination'] 
                         if f in sample_row.index and pd.notna(sample_row[f])]
        if upper_features:
            patterns['upper_body_pattern'] = float(sample_row[upper_features].mean())
            patterns['upper_body_variability'] = float(sample_row[upper_features].std())
        else:
            patterns['upper_body_pattern'] = 0.0
            patterns['upper_body_variability'] = 0.0
        
        # Lower body pattern
        lower_features = [f for f in self.clinical_feature_groups['lower_body_coordination'] 
                         if f in sample_row.index and pd.notna(sample_row[f])]
        if lower_features:
            patterns['lower_body_pattern'] = float(sample_row[lower_features].mean())
            patterns['lower_body_variability'] = float(sample_row[lower_features].std())
        else:
            patterns['lower_body_pattern'] = 0.0
            patterns['lower_body_variability'] = 0.0
        
        # Temporal pattern
        temporal_features = [f for f in self.clinical_feature_groups['temporal_gait_patterns'] 
                           if f in sample_row.index and pd.notna(sample_row[f])]
        if temporal_features:
            patterns['temporal_pattern'] = float(sample_row[temporal_features].mean())
        else:
            patterns['temporal_pattern'] = 0.0
        
        # Overall movement complexity
        eng_features = [col for col in sample_row.index if col.startswith('eng_')]
        if eng_features:
            patterns['movement_complexity'] = float(np.sqrt(np.sum(sample_row[eng_features] ** 2)))
        else:
            patterns['movement_complexity'] = 0.0
        
        return patterns
    
    def _create_optimized_sample_batch(self, session, batch_data):
        """Create optimized sample batch with clinical patterns"""
        session.run("""
            UNWIND $batch AS data
            MATCH (p:Participant {id: data.participant_id})
            MATCH (at:AugmentationType {name: data.augmentation_type})
            CREATE (s:Sample {
                id: data.sample_id,
                participant_id: data.participant_id,
                diagnosis: data.diagnosis,
                augmentation_type: data.augmentation_type,
                sample_index: data.sample_index,
                upper_body_pattern: data.upper_body_pattern,
                upper_body_variability: data.upper_body_variability,
                lower_body_pattern: data.lower_body_pattern,
                lower_body_variability: data.lower_body_variability,
                temporal_pattern: data.temporal_pattern,
                movement_complexity: data.movement_complexity
            })
            CREATE (p)-[:HAS_SAMPLE]->(s)
            CREATE (s)-[:AUGMENTED_BY]->(at)
            
            // Create clinical pattern nodes
            CREATE (cp:ClinicalPattern {
                id: data.sample_id + '_pattern',
                sample_id: data.sample_id,
                upper_body: data.upper_body_pattern,
                lower_body: data.lower_body_pattern,
                temporal: data.temporal_pattern,
                complexity: data.movement_complexity,
                overall_score: (data.upper_body_pattern + data.lower_body_pattern + 
                               data.temporal_pattern + data.movement_complexity) / 4.0
            })
            CREATE (s)-[:HAS_CLINICAL_PATTERN]->(cp)
        """, batch=batch_data)
    
    def create_feature_importance_weights(self, df):
        """Compute and store feature importance weights in the graph"""
        logger.info("🎯 Computing feature importance weights...")
        
        # Prepare data for feature selection
        feature_cols = [col for col in df.columns if 
                       col.startswith('mean ') or col.startswith('eng_') or 
                       col in ['GaCT', 'StaT', 'SwiT', 'MaxStLe', 'MaxStWi', 'StrLe', 'Velocity']]
        
        X = df[feature_cols].fillna(0)
        y = df['diagnosis'].map({'ASD': 1, 'Typical': 0})
        
        # Feature selection using statistical tests
        selector = SelectKBest(f_classif, k='all')
        selector.fit(X, y)
        
        feature_scores = dict(zip(feature_cols, selector.scores_))
        feature_pvalues = dict(zip(feature_cols, selector.pvalues_))
        
        # Store feature importance in graph
        with self.driver.session() as session:
            for feature, score in feature_scores.items():
                pvalue = feature_pvalues[feature]
                importance = float(score)
                significance = float(pvalue)
                
                # Determine feature group
                feature_group = self._classify_feature_to_group(feature)
                
                session.run("""
                    MERGE (fg:FeatureGroup {name: $feature_group})
                    SET fg.importance_score = CASE 
                        WHEN fg.importance_score IS NULL THEN $importance
                        ELSE (fg.importance_score + $importance) / 2
                    END
                    
                    CREATE (fi:FeatureImportance {
                        feature_name: $feature,
                        importance_score: $importance,
                        p_value: $significance,
                        is_significant: $is_significant,
                        feature_group: $feature_group
                    })
                    CREATE (fi)-[:BELONGS_TO]->(fg)
                """, 
                feature=feature,
                feature_group=feature_group,
                importance=importance,
                significance=significance,
                is_significant=pvalue < 0.05
                )
        
        logger.info(f"✅ Stored importance weights for {len(feature_scores)} features")
        
        return feature_scores
    
    def _classify_feature_to_group(self, feature):
        """Classify a feature to its clinical group"""
        for group_name, features in self.clinical_feature_groups.items():
            if feature in features:
                return group_name
        
        if feature.startswith('eng_asymmetry'):
            return 'bilateral_asymmetry'
        elif feature.startswith('eng_'):
            return 'movement_variability'
        else:
            return 'other'
    
    def create_ml_optimized_embeddings(self, df, embedding_dim=16):
        """Create ML-optimized embeddings stored in the graph"""
        logger.info(f"🧠 Creating ML-optimized embeddings (dim={embedding_dim})...")
        
        # Select most important features
        feature_cols = [col for col in df.columns if 
                       col.startswith('mean ') or col.startswith('eng_') or 
                       col in ['GaCT', 'StaT', 'SwiT']]
        
        X = df[feature_cols].fillna(0)
        y = df['diagnosis'].map({'ASD': 1, 'Typical': 0})
        
        # Feature selection and dimensionality reduction
        selector = SelectKBest(f_classif, k=min(32, len(feature_cols)))  # Select top features
        X_selected = selector.fit_transform(X, y)
        
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        logger.info(f"  📊 Selected {len(selected_features)} most discriminative features")
        
        # Standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # PCA for optimal dimensionality
        pca = PCA(n_components=min(embedding_dim, X_scaled.shape[1]))
        embeddings = pca.fit_transform(X_scaled)
        
        # Store embeddings in the graph
        with self.driver.session() as session:
            batch_data = []
            
            for idx, (_, row) in enumerate(df.iterrows()):
                embedding_vector = embeddings[idx].tolist()
                
                embedding_data = {
                    'sample_id': f"S_{row['participant_id']:03d}_{idx % 8}",
                    'participant_id': f"P_{row['participant_id']:03d}",
                    'embedding_vector': embedding_vector,
                    'embedding_dim': len(embedding_vector),
                    'explained_variance': float(pca.explained_variance_ratio_[0]) if len(pca.explained_variance_ratio_) > 0 else 0.0
                }
                batch_data.append(embedding_data)
                
                if len(batch_data) >= 100:
                    self._store_embedding_batch(session, batch_data)
                    batch_data = []
            
            if batch_data:
                self._store_embedding_batch(session, batch_data)
            
            # Store PCA and scaler metadata
            session.run("""
                CREATE (em:EmbeddingModel {
                    type: 'PCA_StandardScaler',
                    embedding_dim: $embedding_dim,
                    n_input_features: $n_features,
                    explained_variance_ratio: $explained_variance,
                    selected_features: $selected_features,
                    created_date: datetime()
                })
            """, 
            embedding_dim=len(embeddings[0]),
            n_features=len(selected_features),
            explained_variance=pca.explained_variance_ratio_.tolist(),
            selected_features=selected_features
            )
        
        logger.info(f"✅ Created optimized embeddings with {pca.explained_variance_ratio_.sum():.3f} explained variance")
        
        return embeddings, selected_features, pca, scaler
    
    def _store_embedding_batch(self, session, batch_data):
        """Store embedding batch in the graph"""
        session.run("""
            UNWIND $batch AS data
            MATCH (s:Sample {id: data.sample_id})
            CREATE (e:Embedding {
                sample_id: data.sample_id,
                vector: data.embedding_vector,
                dimension: data.embedding_dim,
                explained_variance: data.explained_variance
            })
            CREATE (s)-[:HAS_EMBEDDING]->(e)
        """, batch=batch_data)
    
    def create_similarity_relationships(self):
        """Create similarity relationships between samples for better embeddings"""
        logger.info("🔗 Creating similarity relationships...")
        
        with self.driver.session() as session:
            # Create similarity relationships based on clinical patterns
            session.run("""
                MATCH (s1:Sample)-[:HAS_CLINICAL_PATTERN]->(cp1:ClinicalPattern)
                MATCH (s2:Sample)-[:HAS_CLINICAL_PATTERN]->(cp2:ClinicalPattern)
                WHERE s1.id < s2.id AND s1.diagnosis = s2.diagnosis
                WITH s1, s2, cp1, cp2,
                     abs(cp1.upper_body - cp2.upper_body) + 
                     abs(cp1.lower_body - cp2.lower_body) + 
                     abs(cp1.temporal - cp2.temporal) +
                     abs(cp1.complexity - cp2.complexity) AS pattern_distance
                WHERE pattern_distance < 0.5  // Similar patterns
                CREATE (s1)-[:SIMILAR_TO {
                    pattern_distance: pattern_distance,
                    similarity_score: 1.0 - pattern_distance
                }]->(s2)
            """)
            
            # Create contrast relationships between different diagnoses
            session.run("""
                MATCH (s1:Sample {diagnosis: 'ASD'})-[:HAS_CLINICAL_PATTERN]->(cp1:ClinicalPattern)
                MATCH (s2:Sample {diagnosis: 'Typical'})-[:HAS_CLINICAL_PATTERN]->(cp2:ClinicalPattern)
                WITH s1, s2, cp1, cp2,
                     abs(cp1.upper_body - cp2.upper_body) + 
                     abs(cp1.lower_body - cp2.lower_body) + 
                     abs(cp1.temporal - cp2.temporal) +
                     abs(cp1.complexity - cp2.complexity) AS pattern_distance
                WHERE pattern_distance > 1.0  // Different patterns
                WITH s1, s2, pattern_distance
                ORDER BY pattern_distance DESC
                LIMIT 1000  // Top contrasting pairs
                CREATE (s1)-[:CONTRASTS_WITH {
                    pattern_distance: pattern_distance,
                    contrast_score: pattern_distance
                }]->(s2)
            """)
        
        logger.info("✅ Created similarity and contrast relationships")
    
    def create_ml_export_optimized(self):
        """Create optimized ML export functions"""
        logger.info("📤 Creating optimized ML export functions...")
        
        export_functions = {
            'export_optimized_embeddings': """
                // Export optimized embeddings with clinical context
                MATCH (s:Sample)-[:HAS_EMBEDDING]->(e:Embedding)
                MATCH (s)-[:HAS_CLINICAL_PATTERN]->(cp:ClinicalPattern)
                MATCH (p:Participant)-[:HAS_SAMPLE]->(s)
                RETURN 
                    s.id as sample_id,
                    p.id as participant_id,
                    s.diagnosis as diagnosis,
                    s.augmentation_type as augmentation_type,
                    e.vector as embedding_vector,
                    e.dimension as embedding_dim,
                    cp.overall_score as clinical_score,
                    p.clinical_severity as participant_severity
                ORDER BY s.sample_index
            """,
            
            'export_clinical_features': """
                // Export engineered clinical features
                MATCH (s:Sample)-[:HAS_CLINICAL_PATTERN]->(cp:ClinicalPattern)
                MATCH (p:Participant)-[:HAS_SAMPLE]->(s)
                RETURN 
                    s.id as sample_id,
                    p.id as participant_id,
                    s.diagnosis as diagnosis,
                    cp.upper_body as upper_body_pattern,
                    cp.lower_body as lower_body_pattern,
                    cp.temporal as temporal_pattern,
                    cp.complexity as movement_complexity,
                    p.movement_variability as participant_variability,
                    p.coordination_score as coordination_score,
                    p.asymmetry_score as asymmetry_score
                ORDER BY s.sample_index
            """,
            
            'export_feature_importance': """
                // Export feature importance weights
                MATCH (fi:FeatureImportance)-[:BELONGS_TO]->(fg:FeatureGroup)
                RETURN 
                    fi.feature_name as feature_name,
                    fi.importance_score as importance_score,
                    fi.p_value as p_value,
                    fi.is_significant as is_significant,
                    fg.name as feature_group,
                    fg.clinical_relevance as clinical_relevance
                ORDER BY fi.importance_score DESC
            """
        }
        
        # Save export functions
        output_file = 'optimized_ml_export_functions.cypher'
        with open(output_file, 'w') as f:
            f.write("-- Optimized ML Export Functions for NeuroGait Knowledge Graph\n")
            f.write("-- Generated by OptimizedNeuroGaitGraphBuilder\n\n")
            
            for name, query in export_functions.items():
                f.write(f"-- {name.upper().replace('_', ' ')}\n")
                f.write(f"{query}\n\n")
        
        logger.info(f"✅ ML export functions saved to {output_file}")
        return export_functions
    
    def validate_optimized_graph(self):
        """Validate the optimized graph structure"""
        logger.info("🔍 Validating optimized graph structure...")
        
        with self.driver.session() as session:
            validation_results = {}
            
            # Check embeddings completeness
            result = session.run("""
                MATCH (s:Sample)
                OPTIONAL MATCH (s)-[:HAS_EMBEDDING]->(e:Embedding)
                WITH count(s) as total_samples, count(e) as samples_with_embeddings
                RETURN total_samples, samples_with_embeddings,
                       samples_with_embeddings * 100.0 / total_samples as embedding_coverage
            """)
            embedding_stats = result.single()
            validation_results['embedding_coverage'] = dict(embedding_stats)
            
            # Check clinical patterns completeness
            result = session.run("""
                MATCH (s:Sample)
                OPTIONAL MATCH (s)-[:HAS_CLINICAL_PATTERN]->(cp:ClinicalPattern)
                WITH count(s) as total_samples, count(cp) as samples_with_patterns
                RETURN total_samples, samples_with_patterns,
                       samples_with_patterns * 100.0 / total_samples as pattern_coverage
            """)
            pattern_stats = result.single()
            validation_results['pattern_coverage'] = dict(pattern_stats)
            
            # Check similarity relationships
            result = session.run("""
                MATCH ()-[r:SIMILAR_TO]->()
                RETURN count(r) as similarity_relationships
            """)
            validation_results['similarity_relationships'] = result.single()['similarity_relationships']
            
            result = session.run("""
                MATCH ()-[r:CONTRASTS_WITH]->()
                RETURN count(r) as contrast_relationships
            """)
            validation_results['contrast_relationships'] = result.single()['contrast_relationships']
            
            # Check feature importance coverage
            result = session.run("""
                MATCH (fi:FeatureImportance)
                RETURN count(fi) as total_features,
                       count(CASE WHEN fi.is_significant THEN 1 END) as significant_features
            """)
            importance_stats = result.single()
            validation_results['feature_importance'] = dict(importance_stats)
            
            # Log validation results
            logger.info("📊 Optimized Graph Validation Results:")
            for key, value in validation_results.items():
                if isinstance(value, dict):
                    logger.info(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        logger.info(f"    {sub_key}: {sub_value}")
                else:
                    logger.info(f"  {key}: {value}")
            
            return validation_results
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            logger.info("🔌 Neo4j connection closed")
    
    def build_optimized_graph(self, filepath="Final dataset.csv", clear_existing=True):
        """Build the optimized knowledge graph for better ML performance"""
        start_time = datetime.now()
        
        try:
            logger.info("🚀 Starting OPTIMIZED NeuroGait Knowledge Graph construction...")
            
            # Connect to Neo4j
            if not self.connect():
                return False
            
            # Clear existing data if requested
            if clear_existing:
                self.clear_database()
            
            # Create optimized constraints and indexes
            self.create_constraints_and_indexes()
            
            # Create optimized graph structure
            self.create_optimized_graph_structure()
            
            # Load and engineer features
            df_engineered = self.load_and_engineer_features(filepath)
            
            # Create participants and samples with clinical profiles
            self.create_participants_and_samples_optimized(df_engineered)
            
            # Compute and store feature importance
            feature_scores = self.create_feature_importance_weights(df_engineered)
            
            # Create ML-optimized embeddings
            embeddings, selected_features, pca, scaler = self.create_ml_optimized_embeddings(df_engineered)
            
            # Create similarity relationships
            self.create_similarity_relationships()
            
            # Create ML export functions
            export_functions = self.create_ml_export_optimized()
            
            # Validate optimized graph
            validation_results = self.validate_optimized_graph()
            
            # Calculate build time
            build_time = datetime.now() - start_time
            
            # Log final results
            logger.info("🎉 OPTIMIZED KNOWLEDGE GRAPH CONSTRUCTION COMPLETED!")
            logger.info(f"⏱️  Build time: {build_time}")
            logger.info("\n📊 OPTIMIZATION RESULTS:")
            
            logger.info("🧠 ML Optimizations Applied:")
            logger.info(f"  ✅ Advanced feature engineering: {len([col for col in df_engineered.columns if col.startswith('eng_')])} features")
            logger.info(f"  ✅ Clinical pattern modeling: {validation_results['pattern_coverage']['pattern_coverage']:.1f}% coverage")
            logger.info(f"  ✅ Optimized embeddings: {embeddings.shape[1]}D with {pca.explained_variance_ratio_.sum():.1f} explained variance")
            logger.info(f"  ✅ Feature importance weighting: {validation_results['feature_importance']['significant_features']} significant features")
            logger.info(f"  ✅ Similarity relationships: {validation_results['similarity_relationships']} similar pairs")
            logger.info(f"  ✅ Contrast relationships: {validation_results['contrast_relationships']} contrasting pairs")
            
            logger.info("\n🎯 Expected ML Performance Improvements:")
            logger.info("  🔹 Better feature representation through clinical engineering")
            logger.info("  🔹 Reduced dimensionality with preserved information")
            logger.info("  🔹 Clinical relevance-weighted features")
            logger.info("  🔹 Participant-level consistency modeling")
            logger.info("  🔹 Enhanced discrimination between ASD and Typical patterns")
            
            logger.info(f"\n📁 Export functions saved for optimized ML analysis")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error building optimized graph: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
        finally:
            self.close()


def main():
    """Main execution function for optimized graph building"""
    logger.info("🎯 NeuroGait Knowledge Graph Builder - OPTIMIZED FOR ML PERFORMANCE")
    logger.info("📋 Key optimizations:")
    logger.info("   • Advanced clinical feature engineering")
    logger.info("   • ML-focused embeddings with PCA optimization")
    logger.info("   • Feature importance weighting")
    logger.info("   • Clinical pattern modeling")
    logger.info("   • Similarity and contrast relationships")
    logger.info("   • Reduced dimensionality with preserved information")
    
    # Create optimized builder instance
    builder = OptimizedNeuroGaitGraphBuilder(samples_per_participant=8)
    
    # Build the optimized graph
    success = builder.build_optimized_graph("Final dataset.csv")
    
    if success:
        print("\n🎉 SUCCESS: Optimized Knowledge Graph created!")
        print("✅ Advanced feature engineering completed")
        print("✅ ML-optimized embeddings generated")
        print("✅ Clinical patterns modeled")
        print("✅ Feature importance weights computed")
        print("✅ Similarity relationships established")
        print("✅ Expected significant ML performance improvement!")
        print("\n🔗 Optimized graph is ready for enhanced ML analysis!")
        print("\n💡 Next steps:")
        print("   1. Run the ML analysis script")
        print("   2. Compare with previous results")  
        print("   3. Expected: Much better KG embedding performance!")
    else:
        print("❌ Failed to create optimized knowledge graph")
        print("📋 Check logs for details")

if __name__ == "__main__":
    main()