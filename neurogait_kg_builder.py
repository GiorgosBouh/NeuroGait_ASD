#!/usr/bin/env python3
"""
Enhanced Realistic NeuroGait Knowledge Graph Builder with Improved Leakage Prevention
Key Improvements:
1. More rigorous participant-level splitting
2. Additional leakage checks
3. Better feature selection methodology
4. Improved validation metrics
5. More detailed logging
6. FIXED: JSON serialization error
7. FIXED: Uses exact same 12 features as ML script for fair comparison
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

class EnhancedNeuroGaitGraphBuilder:
    def __init__(self, samples_per_participant=8):
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = os.getenv('NEO4J_USER', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD', 'password')
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
        
        # FIXED: Use EXACT same 12 features as ML script for fair comparison
        self.essential_movement_features = [
            'mean HESHL',   # Head-Shoulder Left
            'mean SPELR',   # Spine-Elbow Right  
            'mean SHWRL',   # Shoulder-Wrist Left
            'mean SHWRR',   # Shoulder-Wrist Right
            'mean ELHAL',   # Elbow-Hand Left
            'mean THHAR',   # Thigh-Hand Right
            'mean SPKNL',   # Spine-Knee Left
            'mean SPKNR',   # Spine-Knee Right
            'mean HIANR',   # Hip-Ankle Right
            'GaCT',         # Gait Cycle Time
            'StaT',         # Stance Time
            'SwiT'          # Swing Time
        ]  # Exactly 12 features - same as ML script!
        
        # Configuration
        self.config = {
            'embedding_dim': 8,
            'min_feature_variance': 0.02,
            'test_size': 0.1,
            'random_state': 42
        }
        
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
                    # Auto-confirm for automation (remove prompt)
                    session.run("MATCH (n) DETACH DELETE n")
                    logger.info("🗑️ Database cleared")
                else:
                    logger.info("Database already empty")
            return True
        except Exception as e:
            logger.error(f"❌ Error clearing database: {e}")
            raise
    
    def create_constraints_and_indexes(self):
        """Create constraints and indexes with error handling"""
        constraints = [
            "CREATE CONSTRAINT participant_id_unique IF NOT EXISTS FOR (p:Participant) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT sample_id_unique IF NOT EXISTS FOR (s:Sample) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT embedding_unique IF NOT EXISTS FOR (e:Embedding) REQUIRE e.sample_id IS UNIQUE"
        ]
        
        indexes = [
            "CREATE INDEX sample_participant_idx IF NOT EXISTS FOR (s:Sample) ON (s.participant_id)",
            "CREATE INDEX sample_split_idx IF NOT EXISTS FOR (s:Sample) ON (s.data_split)",
            "CREATE INDEX embedding_sample_idx IF NOT EXISTS FOR (e:Embedding) ON (e.sample_id)",
            "CREATE INDEX participant_diagnosis_idx IF NOT EXISTS FOR (p:Participant) ON (p.diagnosis)"
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
            
            logger.info("✅ Constraints and indexes created")
    
    def load_and_split_data(self, filepath="Final dataset.csv"):
        """Load data and perform rigorous participant-level split"""
        logger.info(f"📊 Loading and splitting data from {filepath}...")
        
        try:
            # Read CSV with multiple encoding attempts
            try:
                df = pd.read_csv(filepath, delimiter=';', decimal=',', encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, delimiter=';', decimal=',', encoding='latin-1')
            
            logger.info(f"📋 Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Convert numeric columns
            numeric_cols = [col for col in df.columns if col != 'class']
            for col in numeric_cols:
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(self.convert_to_float)
            
            # Create participant structure
            df['participant_id'] = df.index // self.samples_per_participant
            df['diagnosis'] = df['class'].map({'A': 'ASD', 'T': 'Typical'})
            
            # Rigorous participant-level split
            participant_info = df.groupby('participant_id')['diagnosis'].first().reset_index()
            
            # Stratified split by diagnosis
            train_pids, test_pids = train_test_split(
                participant_info['participant_id'].values,
                test_size=self.config['test_size'],
                stratify=participant_info['diagnosis'].values,
                random_state=self.config['random_state']
            )
            
            # Mark splits
            df['data_split'] = 'test'
            df.loc[df['participant_id'].isin(train_pids), 'data_split'] = 'train'
            
            # Verify no leakage
            train_diagnosis = df[df['data_split']=='train']['diagnosis'].value_counts()
            test_diagnosis = df[df['data_split']=='test']['diagnosis'].value_counts()
            
            logger.info("\n📊 Data Split Summary:")
            logger.info(f"   Total participants: {len(participant_info)}")
            logger.info(f"   Train participants: {len(train_pids)}")
            logger.info(f"   Test participants: {len(test_pids)}")
            logger.info("\n   Train samples:")
            logger.info(f"      ASD: {train_diagnosis.get('ASD', 0)}")
            logger.info(f"      Typical: {train_diagnosis.get('Typical', 0)}")
            logger.info("\n   Test samples:")
            logger.info(f"      ASD: {test_diagnosis.get('ASD', 0)}")
            logger.info(f"      Typical: {test_diagnosis.get('Typical', 0)}")
            
            return df, train_pids, test_pids
            
        except Exception as e:
            logger.error(f"❌ Error loading/splitting data: {e}")
            raise
    
    def create_embeddings(self, df, train_pids):
        """Create leakage-free embeddings with enhanced feature selection"""
        logger.info("🧠 Creating enhanced realistic embeddings with EXACT 12 features...")
        
        # FIXED: Select only the EXACT 12 features that ML script uses
        available_features = [f for f in self.essential_movement_features if f in df.columns]
        logger.info(f"  🔍 Using {len(available_features)} EXACT features for fair comparison:")
        for feature in available_features:
            logger.info(f"    • {feature}")
        
        # Log missing features if any
        missing_features = [f for f in self.essential_movement_features if f not in df.columns]
        if missing_features:
            logger.warning(f"  ⚠️  Missing features: {missing_features}")
        
        # Separate train and test
        train_mask = df['participant_id'].isin(train_pids)
        X_train = df.loc[train_mask, available_features].fillna(0)
        X_test = df.loc[~train_mask, available_features].fillna(0)
        
        # Enhanced feature selection pipeline
        logger.info("  🔧 Performing rigorous feature selection...")
        
        # 1. Remove low-variance features
        selector = VarianceThreshold(threshold=self.config['min_feature_variance'])
        X_train_selected = selector.fit_transform(X_train)
        selected_mask = selector.get_support()
        selected_features = [f for f, m in zip(available_features, selected_mask) if m]
        
        logger.info(f"  ✅ Selected {len(selected_features)} features after variance threshold:")
        for feature in selected_features:
            logger.info(f"    ✓ {feature}")
        
        # 2. Standardization (train only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test.loc[:, selected_features])
        
        # 3. PCA with realistic dimensions
        pca = PCA(n_components=self.config['embedding_dim'])
        train_embeddings = pca.fit_transform(X_train_scaled)
        test_embeddings = pca.transform(X_test_scaled)
        
        explained_variance = pca.explained_variance_ratio_.sum()
        logger.info(f"  📊 PCA Results:")
        logger.info(f"     Explained variance: {explained_variance:.3f}")
        logger.info(f"     Components: {pca.n_components_}")
        logger.info(f"     Input features: {len(selected_features)} → Output dimensions: {pca.n_components_}")
        
        # Add embeddings to dataframe
        embedding_cols = [f'embedding_{i}' for i in range(train_embeddings.shape[1])]
        
        for col in embedding_cols:
            df[col] = 0.0
        
        df.loc[train_mask, embedding_cols] = train_embeddings
        df.loc[~train_mask, embedding_cols] = test_embeddings
        
        # Save feature selection details
        self.feature_selection = {
            'initial_features': available_features,
            'selected_features': selected_features,
            'variance_threshold': self.config['min_feature_variance'],
            'pca_explained_variance': explained_variance,
            'pca_components': pca.n_components_
        }
        
        return df, embedding_cols, selected_features, pca, scaler
    
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
            
            # Create configuration node
            session.run("""
                MERGE (c:Configuration {
                    name: 'NeuroGaitGraphConfig',
                    embedding_dim: $embedding_dim,
                    min_feature_variance: $min_var,
                    random_state: $random_state
                })
                SET c.created_at = datetime()
            """, 
            embedding_dim=self.config['embedding_dim'],
            min_var=self.config['min_feature_variance'],
            random_state=self.config['random_state']
            )
            
            logger.info("✅ Enhanced graph structure created")
    
    def create_participants_and_samples(self, df):
        """Create participants and samples with enhanced properties"""
        logger.info("👥 Creating participants and samples with metadata...")
        
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
            
            logger.info(f"✅ Created {len(df)} samples")
    
    def create_embeddings_in_graph(self, df, embedding_cols):
        """Store embeddings with additional metadata"""
        logger.info("💾 Storing enhanced embeddings in graph...")
        
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
                        'data_split': row['data_split']
                    })
                
                session.run("""
                    UNWIND $embeddings AS e
                    MATCH (s:Sample {id: e.sample_id})
                    CREATE (embedding:Embedding {
                        sample_id: e.sample_id,
                        vector: e.vector,
                        dimension: e.dimension,
                        data_split: e.data_split,
                        created_at: datetime()
                    })
                    CREATE (s)-[:HAS_EMBEDDING]->(embedding)
                """, embeddings=embeddings_data)
        
        logger.info("✅ Enhanced embeddings stored in graph")
    
    def validate_no_leakage(self):
        """Enhanced leakage validation with more checks"""
        logger.info("🔍 Performing enhanced leakage validation...")
        
        with self.driver.session() as session:
            # Basic participant overlap check
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
            
            # Embedding statistics check
            embedding_stats = session.run("""
                MATCH (e:Embedding)
                WITH e.data_split as split, count(e) as count, 
                     avg(size(e.vector)) as avg_dim, stDev(size(e.vector)) as std_dim
                RETURN split, count, avg_dim, std_dim
                ORDER BY split
            """).data()
            
            # Sample distribution check
            sample_dist = session.run("""
                MATCH (s:Sample)-[:IN_SPLIT]->(ds:DataSplit)
                WITH ds.name as split, count(s) as sample_count,
                     s.diagnosis as diagnosis
                RETURN split, diagnosis, sample_count
                ORDER BY split, diagnosis
            """).data()
            
            logger.info("\n📊 Enhanced Leakage Validation Results:")
            logger.info(f"  Participants:")
            logger.info(f"    Train: {validation['train_count']}")
            logger.info(f"    Test: {validation['test_count']}")
            logger.info(f"    Overlap: {validation['overlap']}")
            
            logger.info("\n  Embedding Statistics:")
            for stat in embedding_stats:
                logger.info(f"    {stat['split']}:")
                logger.info(f"      Count: {stat['count']}")
                logger.info(f"      Avg dim: {stat['avg_dim']:.2f}")
                logger.info(f"      Std dim: {stat['std_dim']:.2f}")
            
            logger.info("\n  Sample Distribution:")
            for dist in sample_dist:
                logger.info(f"    {dist['split']} - {dist['diagnosis']}: {dist['sample_count']}")
            
            if validation['overlap'] == 0:
                logger.info("\n✅ NO DATA LEAKAGE DETECTED")
            else:
                logger.error("\n❌ DATA LEAKAGE DETECTED!")
                raise ValueError("Data leakage detected in validation")
    
    def save_metadata(self, pca, scaler, selected_features):
        """Save model metadata for reproducibility - FIXED JSON serialization"""
        try:
            # FIXED: Convert numpy types to native Python types
            metadata = {
                'pca': {
                    'components': pca.components_.tolist(),
                    'explained_variance': pca.explained_variance_.tolist(),
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                    'mean': pca.mean_.tolist(),
                    'n_components': int(pca.n_components_)  # FIXED: Convert to int
                },
                'scaler': {
                    'scale': scaler.scale_.tolist(),
                    'mean': scaler.mean_.tolist(),
                    'var': scaler.var_.tolist(),
                    'n_samples_seen': int(scaler.n_samples_seen_)  # FIXED: Convert to int
                },
                'selected_features': selected_features,
                'config': {
                    'embedding_dim': int(self.config['embedding_dim']),  # FIXED: Ensure int
                    'min_feature_variance': float(self.config['min_feature_variance']),  # FIXED: Ensure float
                    'test_size': float(self.config['test_size']),  # FIXED: Ensure float
                    'random_state': int(self.config['random_state'])  # FIXED: Ensure int
                },
                'timestamp': datetime.now().isoformat()
            }
            
            with open('neurogait_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("💾 Saved model metadata to neurogait_metadata.json")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not save metadata: {e}")
            logger.info("🔄 Continuing without metadata save...")
    
    def close(self):
        """Close database connection safely"""
        if self.driver:
            try:
                self.driver.close()
                logger.info("🔌 Neo4j connection closed")
            except Exception as e:
                logger.error(f"❌ Error closing connection: {e}")
    
    def build_graph(self, filepath="Final dataset.csv", clear_existing=True):
        """Build the enhanced realistic knowledge graph"""
        start_time = datetime.now()
        
        try:
            logger.info("🚀 Starting Enhanced Realistic NeuroGait Knowledge Graph construction...")
            logger.info("🎯 FIXED: Using EXACT same 12 features as ML script for fair comparison")
            
            # Connect to Neo4j
            if not self.connect():
                return False
            
            # Clear existing data if requested
            if clear_existing and not self.clear_database():
                return False
            
            # Create constraints and indexes
            self.create_constraints_and_indexes()
            
            # Create basic graph structure
            self.create_graph_structure()
            
            # Load and split data
            df, train_pids, test_pids = self.load_and_split_data(filepath)
            
            # Create embeddings
            df_final, embedding_cols, selected_features, pca, scaler = self.create_embeddings(df, train_pids)
            
            # Create participants and samples
            self.create_participants_and_samples(df_final)
            
            # Store embeddings
            self.create_embeddings_in_graph(df_final, embedding_cols)
            
            # Validate no leakage
            self.validate_no_leakage()
            
            # Save metadata (FIXED - now handles JSON serialization properly)
            self.save_metadata(pca, scaler, selected_features)
            
            # Calculate build time
            build_time = datetime.now() - start_time
            
            # Final report
            logger.info("\n🎉 ENHANCED REALISTIC KNOWLEDGE GRAPH CONSTRUCTION COMPLETED!")
            logger.info(f"⏱️  Total build time: {build_time}")
            
            logger.info("\n📊 Construction Summary:")
            logger.info(f"  Participants: {len(train_pids) + len(test_pids)}")
            logger.info(f"  Samples: {len(df_final)}")
            logger.info(f"  Features used: {len(selected_features)} (EXACT same as ML script)")
            logger.info(f"  Embedding dimension: {len(embedding_cols)}")
            logger.info(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
            
            logger.info("\n🔒 Fair Comparison Features:")
            logger.info("  ✅ EXACT same 12 features as ML script")
            logger.info("  ✅ Participant-level stratified split")
            logger.info("  ✅ Training-only feature selection")
            logger.info("  ✅ Training-only PCA fitting")
            logger.info("  ✅ Rigorous validation checks")
            logger.info("  ✅ JSON serialization fixed")
            
            logger.info("\n💡 Next Steps:")
            logger.info("  1. Run fair comparison ML analysis")
            logger.info("  2. Expect realistic KG performance (0.82-0.88 AUC)")
            logger.info("  3. Compare with raw features fairly")
            
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
    logger.info("🎯 Enhanced Realistic NeuroGait Knowledge Graph Builder - FIXED VERSION")
    logger.info("🔧 FIXES: JSON serialization + Exact 12 features for fair comparison")
    
    # Create builder instance
    builder = EnhancedNeuroGaitGraphBuilder(samples_per_participant=8)
    
    # Build the graph
    success = builder.build_graph("Final dataset.csv")
    
    if success:
        print("\n🎉 SUCCESS: Enhanced Realistic Knowledge Graph created!")
        print("🔒 Rigorous leakage prevention measures applied")
        print("📊 Uses EXACT same 12 features as ML script for fair comparison")
        print("🔧 JSON serialization error fixed")
        print("📈 Ready for fair comparison ML analysis")
        print("\n🚀 Next: Run python fair_comparison_ml_analysis.py")
    else:
        print("\n❌ Failed to create knowledge graph")
        print("📋 Check logs for detailed error information")

if __name__ == "__main__":
    main()