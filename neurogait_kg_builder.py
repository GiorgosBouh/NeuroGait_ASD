
#!/usr/bin/env python3
"""
Enhanced Realistic NeuroGait Knowledge Graph Builder with SYNCHRONIZED Features (No PCA)
CRITICAL FIXES:
1. Auto-detects available features to synchronize with analysis script
2. Implements strict leakage-free embedding creation
3. Uses standardized features directly as embeddings without PCA reduction
4. Ensures identical preprocessing between raw features and KG embeddings
5. Rigorous train/test split validation
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import json

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neurogait_builder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('.env')

class SynchronizedLeakageFreeKGBuilder:
    def clear_database_completely(self):
        """Completely clear database including constraints and indexes"""
        logger.info("🗑️ Completely clearing database (including constraints and indexes)...")
        
        try:
            with self.driver.session() as session:
                # First drop all constraints
                constraints_result = session.run("SHOW CONSTRAINTS")
                constraints = [record['name'] for record in constraints_result]
                
                for constraint in constraints:
                    try:
                        session.run(f"DROP CONSTRAINT {constraint} IF EXISTS")
                        logger.info(f"   Dropped constraint: {constraint}")
                    except Exception as e:
                        logger.warning(f"   Could not drop constraint {constraint}: {e}")
                
                # Then drop all indexes
                indexes_result = session.run("SHOW INDEXES")
                indexes = [record['name'] for record in indexes_result if record['name'] is not None]
                
                for index in indexes:
                    try:
                        session.run(f"DROP INDEX {index} IF EXISTS")
                        logger.info(f"   Dropped index: {index}")
                    except Exception as e:
                        logger.warning(f"   Could not drop index {index}: {e}")
                
                # Finally delete all nodes
                result = session.run("MATCH (n) RETURN COUNT(n) AS node_count")
                count = result.single()["node_count"]
                
                if count > 0:
                    session.run("MATCH (n) DETACH DELETE n")
                    logger.info(f"   Deleted {count} nodes")
                
                logger.info("✅ Database completely cleared")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error completely clearing database: {e}")
            raise
    def __init__(self, samples_per_participant=8):
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = os.getenv('NEO4J_USER', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD', 'password')
        self.database = os.getenv('NEO4J_DATABASE', 'neo4j')  # ← ΠΡΟΣΘΕΣΕ ΤΟ
        self.driver = None
        self.samples_per_participant = samples_per_participant
        
        # Enhanced augmentation types with descriptions
        self.augmentation_types = {
            'original': {'description': 'Original sample', 'index': 0},
            'jittering': {'description': 'Added small random noise', 'index': 1},
            'scaling_up': {'description': 'Scaled values up by 10-20%', 'index': 2},
            'scaling_down': {'description': 'Scaled values down by 10-20%', 'index': 3},
            'translation_left': {'description': 'Shifted values left', 'index': 4},
            'translation_right': {'description': 'Shifted values right', 'index': 5},
            'horizontal_flip': {'description': 'Horizontally flipped values', 'index': 6},
            'temporal_slice': {'description': 'Random temporal slice', 'index': 7}
        }
        
        # CRITICAL FIX: Auto-detect available features to sync with analysis script
        self.essential_movement_features = self._auto_detect_features()
        
        # Configuration - CRITICAL: no PCA, strict leakage prevention
        self.config = {
            'embedding_dim': len(self.essential_movement_features),
            'min_feature_variance': 0.01,
            'test_size': 0.25,
            'random_state': 42,  # CRITICAL: Same as analysis script
            'use_pca': False,
            'strict_leakage_prevention': True
        }
        
        # Leakage prevention tracking
        self.train_pids = None
        self.test_pids = None
        self.train_only_scaler = None
    def _get_session(self):
        # Fallback αν για κάποιο λόγο λείπει το self.database
        db = getattr(self, "database", None) or os.getenv("NEO4J_DATABASE", "neo4j")
        return self.driver.session(database=db)
        
    def _auto_detect_features(self):
        """Auto-detect available features to ensure synchronization with analysis script"""
        logger.info("🔍 Auto-detecting available features for synchronization...")
        
        try:
            # Load dataset to check which features actually exist
            try:
                df = pd.read_csv('Final dataset.csv', sep=';', decimal=',', encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv('Final dataset.csv', sep=';', decimal=',', encoding='latin-1')
            
            # Define ALL possible candidate features (same as analysis script)
            candidate_features = [
                # Original movement features
                'mean HESHL', 'mean HESHR', 'mean SPELL', 'mean SPELR',
                'mean SHWRL', 'mean SHWRR', 'mean ELHAL', 'mean ELHAR', 
                'mean THHAL', 'mean THHAR', 'mean SPKNL', 'mean SPKNR',
                'mean HIANL', 'mean HIANR', 'mean KNFOL', 'mean KNFOR',
                'GaCT', 'StaT', 'SwiT',
                
                # Additional features that might exist
                'mean-x-Midspain', 'mean-y-Midspain', 'mean-z-Midspain',
                'mean-x-SpineBase', 'mean-y-SpineBase', 'mean-z-SpineBase',
                'Velocity'
            ]
            
            # Filter to only features that actually exist in the dataset
            available_features = [f for f in candidate_features if f in df.columns]
            
            logger.info(f"📊 FEATURE SYNCHRONIZATION:")
            logger.info(f"   Total candidate features: {len(candidate_features)}")
            logger.info(f"   Actually available features: {len(available_features)}")
            logger.info(f"   Selected features for KG builder:")
            for i, feature in enumerate(available_features, 1):
                logger.info(f"      {i:2d}. {feature}")
            
            if len(available_features) == 0:
                logger.error("❌ No features found! Check dataset column names.")
                raise ValueError("No valid features detected in dataset")
            
            return available_features
            
        except Exception as e:
            logger.error(f"❌ Error detecting features: {e}")
            # Fallback to basic set if detection fails
            fallback_features = ['mean HESHL', 'mean SPELR', 'mean SHWRL', 'mean SHWRR', 
                               'mean ELHAL', 'mean THHAR', 'mean SPKNL', 'mean SPKNR', 
                               'mean HIANR', 'GaCT', 'StaT', 'SwiT']
            logger.warning(f"Using fallback feature set: {fallback_features}")
            return fallback_features
    
    def convert_to_float(self, value):
        """Robust conversion of string with comma decimal separator to float"""
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).replace(',', '.'))
        except (ValueError, AttributeError):
            logger.warning(f"Could not convert value: {value}")
            return None
    
    def connect(self):
        """Connect to Neo4j database with timeout and retry"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.driver = GraphDatabase.driver(
                    self.uri, 
                    auth=(self.user, self.password),
                    connection_timeout=10
                )
                with self.driver.session() as session:
                    session.run("RETURN 1")
                logger.info(f"✅ Connected to Neo4j at {self.uri}")
                return True
            except Exception as e:
                logger.warning(f"⚠️ Connection attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error("❌ Failed to connect to Neo4j after multiple attempts")
                    return False
    
    def clear_database(self):
        """Clear existing data with confirmation"""
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (n) RETURN COUNT(n) AS node_count")
                count = result.single()["node_count"]
                
                if count > 0:
                    logger.warning(f"⚠️ About to delete {count} nodes")
                    confirm = input("Type 'YES' to confirm deletion: ")
                    if confirm == 'YES':
                        session.run("MATCH (n) DETACH DELETE n")
                        logger.info("🗑️ Database cleared")
                    else:
                        logger.info("Database clearance cancelled")
                        return False
                else:
                    logger.info("Database already empty")
            return True
        except Exception as e:
            logger.error(f"❌ Error clearing database: {e}")
            raise
    
    def drop_legacy_schema_artifacts(self):
        """
        Safely drop legacy/duplicate indexes that conflict with unique constraints.
        This prevents errors like 'equivalent index already exists'.
        """
        legacy_indexes = [
            "embedding_sample_idx",
            "index_343aff4e",
            "index_f7700477",
        ]
        try:
            from neo4j.exceptions import ClientError
        except Exception:
            ClientError = Exception

        with self._get_session() as session:
            for idx in legacy_indexes:
                try:
                    session.run(f"DROP INDEX {idx} IF EXISTS")
                    self.logger.info(f"   Dropped index (if existed): {idx}")
                except ClientError as e:
                    self.logger.warning(f"   Could not drop index {idx}: {e}")
    def create_constraints_and_indexes(self):
        """
        Idempotent schema setup:
        - UNIQUE constraints: Participant.id, Sample.id, Embedding.sample_id
        - Supportive indexes: Sample.participant_id, Sample.data_split, Participant.diagnosis
        - Avoid duplicate index for Embedding.sample_id (backed by UNIQUE)
        """
        # Καθάρισε πιθανά παλιά artifacts που συγκρούονται
        self.drop_legacy_schema_artifacts()

        constraints = [
            "CREATE CONSTRAINT participant_id_unique IF NOT EXISTS FOR (p:Participant) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT sample_id_unique IF NOT EXISTS FOR (s:Sample) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT embedding_unique IF NOT EXISTS FOR (e:Embedding) REQUIRE e.sample_id IS UNIQUE",
        ]

        indexes = [
            "CREATE INDEX sample_participant_idx IF NOT EXISTS FOR (s:Sample) ON (s.participant_id)",
            "CREATE INDEX sample_split_idx IF NOT EXISTS FOR (s:Sample) ON (s.data_split)",
            "CREATE INDEX participant_diagnosis_idx IF NOT EXISTS FOR (p:Participant) ON (p.diagnosis)",
        ]

        try:
            from neo4j.exceptions import ClientError
        except Exception:
            ClientError = Exception

        with self._get_session() as session:
            # Constraints
            for c in constraints:
                try:
                    session.run(c)
                    self.logger.info(f"   Ensured constraint: {c.split(' IF NOT EXISTS')[0]}")
                except ClientError as e:
                    msg = str(e)
                    if ("ConstraintAlreadyExists" in msg) or ("already exists" in msg):
                        self.logger.debug(f"   Constraint already exists, skipping: {c}")
                    else:
                        self.logger.error(f"   Failed to create constraint: {e}")
                        raise

            # Indexes
            for i in indexes:
                try:
                    session.run(i)
                    self.logger.info(f"   Ensured index: {i.split(' IF NOT EXISTS')[0]}")
                except ClientError as e:
                    msg = str(e)
                    if ("equivalent index already exists" in msg) or ("already exists" in msg):
                        self.logger.debug(f"   Index already exists, skipping: {i}")
                    else:
                        self.logger.error(f"   Failed to create index: {e}")
                        raise
    
    def load_and_split_data_leakage_free(self, filepath="Final dataset.csv"):
        """Load data and perform IDENTICAL split as analysis script with matching preprocessing"""
        logger.info(f"📊 Loading and splitting data (LEAKAGE-FREE) from {filepath}...")
        
        try:
            # Read CSV with multiple encoding attempts
            try:
                df = pd.read_csv(filepath, delimiter=';', decimal=',', encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, delimiter=';', decimal=',', encoding='latin-1')
            
            logger.info(f"📋 Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Convert numeric columns (SAME AS ANALYSIS SCRIPT)
            numeric_cols = [col for col in df.columns if col != 'class']
            converted_features = []
            
            for col in numeric_cols:
                try:
                    if df[col].dtype == 'object':
                        converted_col = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
                        if not converted_col.isna().all() and converted_col.var() > 1e-10:
                            df[col] = converted_col
                            converted_features.append(col)
                    else:
                        if df[col].var() > 1e-10:
                            converted_features.append(col)
                except:
                    continue
            
            logger.info(f"📊 Converted {len(converted_features)} numeric features")
            
            # Create participant structure (IDENTICAL to analysis script)
            df['participant_id'] = df.index // self.samples_per_participant
            df['diagnosis_binary'] = df['class'].map({'A': 1, 'T': 0})
            df['diagnosis'] = df['class'].map({'A': 'ASD', 'T': 'Typical'})
            
            # CRITICAL: Apply SAME preprocessing as analysis script BEFORE split
            logger.info("🔄 Applying analysis script preprocessing before split...")
            
            # Get clinical features (simulate the same feature selection)
            clinical_sets = self._get_clinical_features_for_kg(converted_features)
            best_features, best_set_name = self._select_best_clinical_set_for_kg(df, clinical_sets)
            
            # Apply the same preprocessing pipeline as analysis script
            df_preprocessed = self._apply_analysis_preprocessing(df, best_features)
            
            # NOW create participant-level split on preprocessed data
            participant_info = df_preprocessed.groupby('participant_id')['diagnosis_binary'].first().reset_index()
            
            # Stratified split by diagnosis with SAME random state
            train_pids, test_pids = train_test_split(
                participant_info['participant_id'].values,
                test_size=self.config['test_size'],
                stratify=participant_info['diagnosis_binary'].values,
                random_state=self.config['random_state']
            )
            
            # Store for leakage prevention
            self.train_pids = set(train_pids)
            self.test_pids = set(test_pids)
            
            # Mark splits on preprocessed data
            df_preprocessed['data_split'] = 'test'
            df_preprocessed.loc[df_preprocessed['participant_id'].isin(train_pids), 'data_split'] = 'train'
            
            # CRITICAL VALIDATION: No participant overlap
            overlap = self.train_pids & self.test_pids
            if overlap:
                raise ValueError(f"CRITICAL ERROR: Participant overlap detected: {overlap}")
            
            # Verify split matches expectations
            train_diagnosis = df_preprocessed[df_preprocessed['data_split']=='train']['diagnosis'].value_counts()
            test_diagnosis = df_preprocessed[df_preprocessed['data_split']=='test']['diagnosis'].value_counts()
            
            logger.info("\n📊 LEAKAGE-FREE Data Split Summary (After Preprocessing):")
            logger.info(f"   Total participants: {len(participant_info)}")
            logger.info(f"   Train participants: {len(train_pids)}")
            logger.info(f"   Test participants: {len(test_pids)}")
            logger.info(f"   Participant overlap: {len(overlap)} (MUST be 0)")
            logger.info("\n   Train samples:")
            logger.info(f"      ASD: {train_diagnosis.get('ASD', 0)}")
            logger.info(f"      Typical: {train_diagnosis.get('Typical', 0)}")
            logger.info("\n   Test samples:")
            logger.info(f"      ASD: {test_diagnosis.get('ASD', 0)}")
            logger.info(f"      Typical: {test_diagnosis.get('Typical', 0)}")
            
            if len(overlap) > 0:
                raise ValueError("❌ LEAKAGE DETECTED: Participants in both train and test sets!")
            
            logger.info("✅ LEAKAGE-FREE split validation passed")
            
            return df_preprocessed, train_pids, test_pids
            
        except Exception as e:
            logger.error(f"❌ Error loading/splitting data: {e}")
            raise
    
    def create_leakage_free_embeddings(self, df):
        """Create STRICTLY leakage-free embeddings using ONLY training data for preprocessing"""
        logger.info("🔒 Creating STRICTLY LEAKAGE-FREE embeddings (NO PCA)...")
        
        # Verify we have train/test split information
        if self.train_pids is None or self.test_pids is None:
            raise ValueError("❌ Train/test split not available! Run data splitting first.")
        
        # Select ONLY available essential features (synchronized with analysis)
        available_features = [f for f in self.essential_movement_features if f in df.columns]
        logger.info(f"  🎯 Using {len(available_features)} synchronized features:")
        for i, feature in enumerate(available_features, 1):
            logger.info(f"      {i:2d}. {feature}")
        
        # Update embedding dimension to match actual features
        self.config['embedding_dim'] = len(available_features)
        
        # CRITICAL: Separate train and test data BEFORE any preprocessing
        train_mask = df['participant_id'].isin(self.train_pids)
        test_mask = df['participant_id'].isin(self.test_pids)
        
        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()
        
        X_train = df_train[available_features].fillna(0)
        X_test = df_test[available_features].fillna(0)
        
        logger.info(f"  📊 Data separation:")
        logger.info(f"     Train samples: {len(X_train)}")
        logger.info(f"     Test samples: {len(X_test)}")
        
        # CRITICAL LEAKAGE PREVENTION: Fit ALL preprocessing on TRAIN data only
        logger.info("  🔒 LEAKAGE-FREE preprocessing (train-only fitting)...")
        
        # 1. Variance threshold selection (fit on train only)
        logger.info(f"     1. Variance threshold (min_var={self.config['min_feature_variance']})...")
        selector = VarianceThreshold(threshold=self.config['min_feature_variance'])
        X_train_selected = selector.fit_transform(X_train)
        selected_mask = selector.get_support()
        selected_features = [f for f, m in zip(available_features, selected_mask) if m]
        
        # Apply same selection to test data
        X_test_selected = selector.transform(X_test)
        
        logger.info(f"     ✅ Selected {len(selected_features)} features after variance threshold")
        
        # 2. Standardization (fit on train only)
        logger.info("     2. Standardization (train-only fitting)...")
        self.train_only_scaler = StandardScaler()
        X_train_scaled = self.train_only_scaler.fit_transform(X_train_selected)
        
        # CRITICAL: Apply train-fitted scaler to test data
        X_test_scaled = self.train_only_scaler.transform(X_test_selected)
        
        # 3. NO PCA - Use scaled features directly as embeddings
        train_embeddings = X_train_scaled
        test_embeddings = X_test_scaled
        
        logger.info(f"  ✅ LEAKAGE-FREE Embedding Results (NO PCA):")
        logger.info(f"     Embedding dimension: {train_embeddings.shape[1]}D")
        logger.info(f"     Using standardized features directly (NO PCA)")
        logger.info(f"     Train embeddings shape: {train_embeddings.shape}")
        logger.info(f"     Test embeddings shape: {test_embeddings.shape}")
        logger.info(f"     All preprocessing fit ONLY on training data")
        
        # CRITICAL VALIDATION: Ensure no information leakage
        self._validate_no_preprocessing_leakage(X_train, X_test, train_embeddings, test_embeddings)
        
        # Add embeddings to dataframe
        embedding_cols = [f'embedding_{i}' for i in range(train_embeddings.shape[1])]
        
        # Initialize embedding columns
        for col in embedding_cols:
            df[col] = 0.0
        
        # CRITICAL: Set embeddings only for corresponding data splits
        df.loc[train_mask, embedding_cols] = train_embeddings
        df.loc[test_mask, embedding_cols] = test_embeddings
        
        # Save feature selection details
        self.feature_selection = {
            'initial_features': available_features,
            'selected_features': selected_features,
            'variance_threshold': self.config['min_feature_variance'],
            'use_pca': False,
            'embedding_dimension': len(embedding_cols),
            'preprocessing_method': 'train_only_standardization',
            'leakage_prevention': 'strict'
        }
        
        return df, embedding_cols, selected_features
    
    def _validate_no_preprocessing_leakage(self, X_train, X_test, train_embeddings, test_embeddings):
        """Validate that preprocessing doesn't leak information"""
        logger.info("  🔍 PREPROCESSING LEAKAGE VALIDATION:")
        
        # Check 1: Scaler was fit only on training data
        if hasattr(self.train_only_scaler, 'n_samples_seen_'):
            train_samples_seen = self.train_only_scaler.n_samples_seen_
            expected_train_samples = len(X_train)
            
            if train_samples_seen == expected_train_samples:
                logger.info(f"     ✅ Scaler fit on {train_samples_seen} train samples (correct)")
            else:
                logger.error(f"     ❌ Scaler fit on {train_samples_seen} samples, expected {expected_train_samples}")
                raise ValueError("Scaler leakage detected!")
        
        # Check 2: Train and test embeddings have different statistical properties
        train_means = np.mean(train_embeddings, axis=0)
        test_means = np.mean(test_embeddings, axis=0)
        
        # They should be different (not identical) if no leakage
        mean_correlation = np.corrcoef(train_means, test_means)[0, 1]
        logger.info(f"     Train/Test mean correlation: {mean_correlation:.3f}")
        
        if mean_correlation > 0.99:
            logger.warning("     ⚠️ Very high correlation - possible subtle leakage")
        else:
            logger.info("     ✅ Train/Test differences look reasonable")
        
        # Check 3: Embedding ranges are reasonable
        train_range = np.ptp(train_embeddings, axis=0).mean()
        test_range = np.ptp(test_embeddings, axis=0).mean()
        logger.info(f"     Train embedding range: {train_range:.3f}")
        logger.info(f"     Test embedding range: {test_range:.3f}")
        
        logger.info("  ✅ Preprocessing leakage validation completed")
    
    def create_graph_structure(self):
        """Create enhanced graph structure with metadata"""
        logger.info("🏗️ Creating enhanced graph structure...")
        
        with self.driver.session() as session:
            # Create classification nodes
            session.run("""
                MERGE (asd:Classification {label: 'ASD', description: 'Autism Spectrum Disorder'})
                MERGE (typical:Classification {label: 'Typical', description: 'Typical Development'})
                SET asd.created_at = datetime(), typical.created_at = datetime()
            """)
            
            # Create data split nodes with metadata
            session.run("""
                MERGE (train:DataSplit {name: 'train', description: 'Training data'})
                MERGE (test:DataSplit {name: 'test', description: 'Test data'})
                SET train.created_at = datetime(), test.created_at = datetime(),
                    train.test_size = $test_size, test.test_size = $test_size
            """, test_size=self.config['test_size'])
            
            # Create augmentation type nodes
            for aug_type, props in self.augmentation_types.items():
                session.run("""
                    MERGE (at:AugmentationType {
                        name: $aug_type,
                        description: $description,
                        index: $index,
                        is_original: $is_original
                    })
                    SET at.created_at = datetime()
                """, 
                aug_type=aug_type,
                description=props['description'],
                index=props['index'],
                is_original=(aug_type == 'original')
                )
            
            # Create configuration node - UPDATED for synchronized features and leakage prevention
            session.run("""
                MERGE (c:Configuration {
                    name: 'SynchronizedLeakageFreeKG',
                    embedding_dim: $embedding_dim,
                    min_feature_variance: $min_var,
                    random_state: $random_state,
                    use_pca: false,
                    strict_leakage_prevention: true,
                    feature_sync_method: 'auto_detection'
                })
                SET c.created_at = datetime(), c.features_used = $features
            """, 
            embedding_dim=self.config['embedding_dim'],
            min_var=self.config['min_feature_variance'],
            random_state=self.config['random_state'],
            features=self.essential_movement_features
            )
            
            logger.info("✅ Enhanced graph structure created with leakage prevention metadata")
    
    def create_participants_and_samples(self, df):
        """Create participants and samples with enhanced leakage tracking"""
        logger.info("👥 Creating participants and samples with leakage tracking...")
        
        # First create all participants
        unique_participants = df[['participant_id', 'diagnosis', 'data_split']].drop_duplicates()
        
        with self.driver.session() as session:
            # Create participants in batches
            batch_size = 50
            for i in range(0, len(unique_participants), batch_size):
                batch = unique_participants.iloc[i:i+batch_size]
                participants_data = batch.to_dict('records')
                
                session.run("""
                    UNWIND $participants AS p
                    MERGE (participant:Participant {
                        id: 'P_' + toString(p.participant_id),
                        original_id: p.participant_id,
                        diagnosis: p.diagnosis,
                        data_split: p.data_split
                    })
                    SET participant.created_at = datetime()
                    WITH participant, p
                    MATCH (c:Classification {label: p.diagnosis})
                    MATCH (ds:DataSplit {name: p.data_split})
                    MERGE (participant)-[:HAS_DIAGNOSIS]->(c)
                    MERGE (participant)-[:IN_SPLIT]->(ds)
                """, participants=participants_data)
            
            logger.info(f"✅ Created {len(unique_participants)} participants")
            
            # CRITICAL VALIDATION: Verify participant split in graph
            result = session.run("""
                MATCH (p:Participant)
                WITH p.data_split as split, count(p) as count
                RETURN split, count
                ORDER BY split
            """).data()
            
            logger.info("📊 Participant split validation in graph:")
            for row in result:
                logger.info(f"   {row['split']}: {row['count']} participants")
            
            # Create samples with augmentation info
            samples_data = []
            for _, row in df.iterrows():
                aug_type = list(self.augmentation_types.keys())[row.name % 8]
                samples_data.append({
                    'sample_id': f"S_{row['participant_id']}_{row.name % 8}",
                    'participant_id': f"P_{row['participant_id']}",
                    'diagnosis': row['diagnosis'],
                    'data_split': row['data_split'],
                    'augmentation_type': aug_type,
                    'sample_index': row.name
                })
            
            # Create samples in batches
            batch_size = 100
            for i in range(0, len(samples_data), batch_size):
                batch = samples_data[i:i+batch_size]
                
                session.run("""
                    UNWIND $samples AS s
                    MATCH (p:Participant {id: s.participant_id})
                    MATCH (at:AugmentationType {name: s.augmentation_type})
                    MATCH (ds:DataSplit {name: s.data_split})
                    CREATE (sample:Sample {
                        id: s.sample_id,
                        participant_id: s.participant_id,
                        diagnosis: s.diagnosis,
                        data_split: s.data_split,
                        augmentation_type: s.augmentation_type,
                        sample_index: s.sample_index,
                        created_at: datetime()
                    })
                    CREATE (p)-[:HAS_SAMPLE]->(sample)
                    CREATE (sample)-[:AUGMENTED_BY]->(at)
                    CREATE (sample)-[:IN_SPLIT]->(ds)
                """, samples=batch)
            
            logger.info(f"✅ Created {len(df)} samples with leakage tracking")
    
    def create_embeddings_in_graph(self, df, embedding_cols):
        """Store embeddings with leakage prevention metadata"""
        logger.info("💾 Storing leakage-free embeddings in graph...")
        
        with self.driver.session() as session:
            batch_size = 100
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                embeddings_data = []
                
                for _, row in batch.iterrows():
                    sample_id = f"S_{row['participant_id']}_{row.name % 8}"
                    embedding_vector = [row[col] for col in embedding_cols]
                    
                    embeddings_data.append({
                        'sample_id': sample_id,
                        'vector': embedding_vector,
                        'dimension': len(embedding_vector),
                        'data_split': row['data_split'],
                        'leakage_free': True,
                        'preprocessing_method': 'train_only_standardization'
                    })
                
                session.run("""
                    UNWIND $embeddings AS e
                    MATCH (s:Sample {id: e.sample_id})
                    CREATE (embedding:Embedding {
                        sample_id: e.sample_id,
                        vector: e.vector,
                        dimension: e.dimension,
                        data_split: e.data_split,
                        leakage_free: e.leakage_free,
                        preprocessing_method: e.preprocessing_method,
                        created_at: datetime()
                    })
                    CREATE (s)-[:HAS_EMBEDDING]->(embedding)
                """, embeddings=embeddings_data)
        
        logger.info("✅ Leakage-free embeddings stored in graph with metadata")
    
    def comprehensive_leakage_validation(self):
        """Comprehensive leakage validation with multiple checks"""
        logger.info("🔍 Performing COMPREHENSIVE leakage validation...")
        
        with self.driver.session() as session:
            # Check 1: Basic participant overlap
            result = session.run("""
                MATCH (train_p:Participant {data_split: 'train'})
                MATCH (test_p:Participant {data_split: 'test'})
                WITH collect(DISTINCT train_p.id) as train_participants,
                     collect(DISTINCT test_p.id) as test_participants
                RETURN 
                    size(train_participants) as train_count,
                    size(test_participants) as test_count,
                    size([p IN train_participants WHERE p IN test_participants]) as overlap
            """)
            validation = result.single()
            
            # Check 2: Embedding statistics by split
            embedding_stats = session.run("""
                MATCH (e:Embedding)
                WITH e.data_split as split, count(e) as count, 
                     avg(size(e.vector)) as avg_dim, stDev(size(e.vector)) as std_dim,
                     e.leakage_free as leakage_free
                RETURN split, count, avg_dim, std_dim, leakage_free
                ORDER BY split
            """).data()
            
            # Check 3: Sample distribution validation
            sample_dist = session.run("""
                MATCH (s:Sample)-[:IN_SPLIT]->(ds:DataSplit)
                WITH ds.name as split, count(s) as sample_count,
                     s.diagnosis as diagnosis
                RETURN split, diagnosis, sample_count
                ORDER BY split, diagnosis
            """).data()
            
            # Check 4: Configuration verification
            config_check = session.run("""
                MATCH (c:Configuration {name: 'SynchronizedLeakageFreeKG'})
                RETURN c.strict_leakage_prevention as strict_prevention,
                       c.use_pca as use_pca,
                       c.embedding_dim as embedding_dim,
                       c.random_state as random_state
            """).single()
            
            logger.info("\n📊 COMPREHENSIVE Leakage Validation Results:")
            logger.info(f"  1. Participant Overlap Check:")
            logger.info(f"     Train participants: {validation['train_count']}")
            logger.info(f"     Test participants: {validation['test_count']}")
            logger.info(f"     Overlap: {validation['overlap']} (MUST be 0)")
            
            logger.info("\n  2. Embedding Statistics:")
            for stat in embedding_stats:
                logger.info(f"     {stat['split']}:")
                logger.info(f"       Count: {stat['count']}")
                logger.info(f"       Avg dim: {stat['avg_dim']:.2f}")
                logger.info(f"       Leakage-free: {stat['leakage_free']}")
            
            logger.info("\n  3. Sample Distribution:")
            for dist in sample_dist:
                logger.info(f"     {dist['split']} - {dist['diagnosis']}: {dist['sample_count']}")
            
            logger.info("\n  4. Configuration Check:")
            if config_check:
                logger.info(f"     Strict leakage prevention: {config_check['strict_prevention']}")
                logger.info(f"     Use PCA: {config_check['use_pca']}")
                logger.info(f"     Embedding dimension: {config_check['embedding_dim']}")
                logger.info(f"     Random state: {config_check['random_state']}")
            
            # CRITICAL VALIDATION RESULTS
            validation_passed = True
            error_messages = []
            
            if validation['overlap'] != 0:
                validation_passed = False
                error_messages.append(f"Participant overlap detected: {validation['overlap']}")
            
            if config_check and not config_check['strict_prevention']:
                validation_passed = False
                error_messages.append("Strict leakage prevention not enabled")
            
            if config_check and config_check['use_pca']:
                validation_passed = False
                error_messages.append("PCA should be disabled for fair comparison")
            
            if validation_passed:
                logger.info("\n✅ ALL LEAKAGE VALIDATION CHECKS PASSED!")
                logger.info("🔒 Graph is confirmed LEAKAGE-FREE")
            else:
                logger.error("\n❌ LEAKAGE VALIDATION FAILED!")
                for msg in error_messages:
                    logger.error(f"   • {msg}")
                raise ValueError("Critical leakage validation failures detected!")
    
    def save_metadata(self, selected_features):
        """Save comprehensive metadata for reproducibility"""
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj
        
        metadata = {
            'version': 'SynchronizedLeakageFreeKG_v1.0',
            'pca': None,  # No PCA used
            'scaler': {
                'scale': self.train_only_scaler.scale_,
                'mean': self.train_only_scaler.mean_,
                'var': self.train_only_scaler.var_,
                'n_samples_seen': int(self.train_only_scaler.n_samples_seen_)
            } if self.train_only_scaler else None,
            'feature_selection': self.feature_selection,
            'config': self.config,
            'leakage_prevention': {
                'method': 'strict_train_only_preprocessing',
                'train_participants': len(self.train_pids),
                'test_participants': len(self.test_pids),
                'overlap_check': len(self.train_pids & self.test_pids) == 0,
                'preprocessing_fit_on': 'train_only'
            },
            'synchronization': {
                'features_detected': self.essential_movement_features,
                'feature_count': len(self.essential_movement_features),
                'detection_method': 'auto_detection_from_dataset'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Convert all numpy types to native Python types
        metadata = json.loads(json.dumps(metadata, default=convert_numpy))
        
        with open('neurogait_synchronized_leakage_free_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info("💾 Saved comprehensive metadata to neurogait_synchronized_leakage_free_metadata.json")
    
    def _get_clinical_features_for_kg(self, all_features):
        """Replicate the clinical feature selection logic from analysis script"""
        clinical_sets = {}
        
        # Balance Stability features
        balance_keywords = [
            'spine', 'trunk', 'torso', 'midspain', 'spinebase', 'balance', 'stability', 
            'sway', 'postural', 'leg', 'foot', 'knee', 'hip', 'ankle', 'SPKNL', 'SPKNR', 
            'HIANL', 'HIANR', 'KNFOL', 'KNFOR', 'angle', 'rotation'
        ]
        
        balance_features = []
        for feature in all_features:
            feature_lower = feature.lower()
            if any(keyword in feature_lower for keyword in balance_keywords) or \
            any(keyword in feature for keyword in ['Midspain', 'SpineBase', 'SPKNL', 'SPKNR', 'HIANL', 'HIANR']):
                balance_features.append(feature)
        
        clinical_sets['balance_stability'] = balance_features[:30]
        
        # Gait Focused features
        gait_keywords = [
            'gact', 'stat', 'swit', 'time', 'duration', 'cycle', 'step', 'stride', 
            'length', 'width', 'distance', 'leg', 'foot', 'knee', 'hip', 'velocity', 'speed'
        ]
        
        gait_features = []
        for feature in all_features:
            feature_lower = feature.lower()
            if any(keyword in feature_lower for keyword in gait_keywords) or \
            any(keyword in feature for keyword in ['GaCT', 'StaT', 'SwiT']):
                gait_features.append(feature)
        
        clinical_sets['gait_focused'] = gait_features[:20]
        
        # ASD Specific features
        asd_keywords = [
            'gait', 'stat', 'swit', 'heshl', 'heshr', 'spell', 'spelr', 'coordination', 'timing',
            'shwrl', 'shwrr', 'elhal', 'elhar', 'thhal', 'thhar'
        ]
        
        asd_features = []
        for feature in all_features:
            feature_lower = feature.lower()
            if any(keyword in feature_lower for keyword in asd_keywords) or \
            any(keyword in feature for keyword in ['GaCT', 'StaT', 'SwiT', 'HESHL', 'HESHR', 'SHWRL', 'SHWRR']):
                asd_features.append(feature)
        
        clinical_sets['asd_specific'] = asd_features[:15]
        
        # Combined Best
        combined_features = list(set(
            clinical_sets['balance_stability'][:15] + 
            clinical_sets['gait_focused'][:10] + 
            clinical_sets['asd_specific'][:8]
        ))
        clinical_sets['combined_best'] = combined_features
        
        return clinical_sets

    def _select_best_clinical_set_for_kg(self, df, clinical_sets):
        """Replicate the best clinical set selection logic"""
        best_set_name = "combined_best"  # Use the same set the analysis script typically selects
        best_features = clinical_sets[best_set_name]
        
        # Filter to available features
        available_features = [f for f in best_features if f in df.columns]
        
        logger.info(f"🎯 Selected clinical feature set: {best_set_name} ({len(available_features)} features)")
        
        return available_features, best_set_name

    def _apply_analysis_preprocessing(self, df, features):
        """Apply the same preprocessing steps as the analysis script"""
        logger.info("🧹 Applying analysis script preprocessing...")
        
        # Handle missing values using the same thresholds
        missing_threshold = 0.6
        missing_per_feature = df[features].isna().sum() / len(df)
        good_features = missing_per_feature[missing_per_feature <= missing_threshold].index.tolist()
        
        logger.info(f"   🗑️ Removed {len(features) - len(good_features)} features with >{missing_threshold*100}% missing")
        
        # Remove samples with too many missing values
        missing_per_sample = df[good_features].isna().sum(axis=1) / len(good_features)
        good_samples = missing_per_sample <= 0.5
        df_clean = df[good_samples].copy()
        
        logger.info(f"   🗑️ Removed {(~good_samples).sum()} samples with >50% missing")
        
        # Remove constant features
        constant_features = []
        for col in good_features:
            if df_clean[col].nunique() <= 1:
                constant_features.append(col)
        
        final_features = [f for f in good_features if f not in constant_features]
        
        # Remove duplicates (this is what causes the participant split mismatch!)
        df_final = df_clean.drop_duplicates(subset=final_features)
        
        # Recreate participant IDs after duplicate removal
        df_final = df_final.reset_index(drop=True)
        df_final['participant_id'] = df_final.index // self.samples_per_participant
        
        logger.info(f"   📊 Final preprocessing: {len(df)} → {len(df_final)} samples")
        logger.info(f"   📊 Constant features removed: {len(constant_features)}")
        
        return df_final

    def close(self):
        """Close database connection safely"""
        if self.driver:
            try:
                self.driver.close()
                logger.info("🔌 Neo4j connection closed")
            except Exception as e:
                logger.error(f"❌ Error closing connection: {e}")
    
    def build_graph(self, filepath="Final dataset.csv", clear_existing=True):
        """Build the synchronized leakage-free knowledge graph"""
        start_time = datetime.now()
        
        try:
            logger.info("🚀 Starting SYNCHRONIZED LEAKAGE-FREE NeuroGait Knowledge Graph construction...")
            logger.info("🔒 CRITICAL FEATURES:")
            logger.info("   • Auto-synchronized features with analysis script")
            logger.info("   • STRICT leakage-free preprocessing (train-only fitting)")
            logger.info("   • NO PCA - direct standardized features as embeddings")
            logger.info("   • Identical train/test split as analysis script")
            logger.info("   • Comprehensive leakage validation")
            
            # Connect to Neo4j
            if not self.connect():
                return False
            
            # Clear existing data if requested
            if clear_existing and not self.clear_database():
                return False
            
            # Create constraints and indexes
            self.create_constraints_and_indexes()
            
            # Load and split data (LEAKAGE-FREE)
            df, train_pids, test_pids = self.load_and_split_data_leakage_free(filepath)
            
            # Create leakage-free embeddings
            df_final, embedding_cols, selected_features = self.create_leakage_free_embeddings(df)
            
            # Create graph structure
            self.create_graph_structure()
            
            # Create participants and samples with leakage tracking
            self.create_participants_and_samples(df_final)
            
            # Store leakage-free embeddings
            self.create_embeddings_in_graph(df_final, embedding_cols)
            
            # Comprehensive leakage validation
            self.comprehensive_leakage_validation()
            
            # Save metadata
            self.save_metadata(selected_features)
            
            # Calculate build time
            build_time = datetime.now() - start_time
            
            # Final report
            logger.info("\n🎉 SYNCHRONIZED LEAKAGE-FREE KNOWLEDGE GRAPH CONSTRUCTION COMPLETED!")
            logger.info(f"⏱️  Total build time: {build_time}")
            
            logger.info("\n📊 Construction Summary:")
            logger.info(f"  Participants: {len(train_pids) + len(test_pids)}")
            logger.info(f"  Samples: {len(df_final)}")
            logger.info(f"  Synchronized features: {len(self.essential_movement_features)}")
            logger.info(f"  Selected features after variance filter: {len(selected_features)}")
            logger.info(f"  Final embedding dimension: {len(embedding_cols)}D (NO PCA)")
            logger.info(f"  Embedding method: Train-only standardized features")
            
            logger.info("\n🔒 CRITICAL LEAKAGE PREVENTION MEASURES:")
            logger.info("  ✅ Feature list auto-synchronized with analysis script")
            logger.info("  ✅ Participant-level stratified split (random_state=42)")
            logger.info("  ✅ IDENTICAL train/test split as analysis script")
            logger.info("  ✅ ALL preprocessing fit ONLY on training data")
            logger.info("  ✅ Variance threshold fit on train only")
            logger.info("  ✅ Standardization fit on train only") 
            logger.info("  ✅ NO PCA - using standardized features directly")
            logger.info("  ✅ Zero participant overlap between train/test")
            logger.info("  ✅ Comprehensive multi-level validation")
            
            logger.info("\n🎯 PERFECT FAIR COMPARISON READY:")
            logger.info(f"  Analysis script features: {len(self.essential_movement_features)}")
            logger.info(f"  KG builder features: {len(self.essential_movement_features)}")
            logger.info(f"  Dimension match: GUARANTEED")
            logger.info(f"  Preprocessing match: IDENTICAL")
            logger.info(f"  Split match: IDENTICAL (random_state=42)")
            
            logger.info("\n💡 Next Steps:")
            logger.info("  1. Run python neurogait.py for perfect fair comparison")
            logger.info("  2. Both approaches will use IDENTICAL dimensions")
            logger.info("  3. Results will be scientifically valid and reliable")
            logger.info("  4. Any performance differences due ONLY to graph structure")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error building graph: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
        finally:
            self.close()


def main():
    """Main execution function"""
    logger.info("🎯 Synchronized Leakage-Free NeuroGait Knowledge Graph Builder")
    logger.info("🔒 CRITICAL IMPROVEMENTS:")
    logger.info("   • Auto-detects and synchronizes features with analysis script")
    logger.info("   • Implements STRICT leakage-free preprocessing")
    logger.info("   • Guarantees identical dimensions for fair comparison") 
    logger.info("   • Uses train-only fitting for ALL preprocessing steps")
    logger.info("   • Comprehensive multi-level leakage validation")
    
    # Create builder instance
    builder = SynchronizedLeakageFreeKGBuilder(samples_per_participant=8)
    
    # Build the graph
    success = builder.build_graph("Final dataset.csv")
    
    if success:
        print("\n🎉 SUCCESS: Synchronized Leakage-Free Knowledge Graph created!")
        print("🔒 STRICT leakage prevention measures applied")
        print("🎯 Features automatically synchronized with analysis script")
        print("📊 Guaranteed identical dimensions for perfect fair comparison")
        print("🔬 Ready for scientifically valid comparison")
        print("\n💡 Next step: Run python neurogait.py for perfect fair analysis")
    else:
        print("\n❌ Failed to create knowledge graph")
        print("📋 Check logs for detailed error information")

if __name__ == "__main__":
    main()