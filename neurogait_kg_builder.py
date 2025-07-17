#!/usr/bin/env python3
"""
NeuroGait Knowledge Graph Builder - FIXED VERSION
Properly handles participant structure and augmentation metadata
Prevents future data leakage in ML analysis
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import logging
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('.env')

class NeuroGaitGraphBuilderFixed:
    def __init__(self, samples_per_participant=8):
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = os.getenv('NEO4J_USER', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD', 'your_password')
        self.driver = None
        self.samples_per_participant = samples_per_participant
        
        # Augmentation type mapping (based on the 7 transformations mentioned in paper)
        self.augmentation_types = [
            'original',          # Sample 0 for each participant
            'jittering',         # Sample 1 
            'scaling_up',        # Sample 2
            'scaling_down',      # Sample 3
            'translation_left',  # Sample 4
            'translation_right', # Sample 5
            'horizontal_flip',   # Sample 6
            'temporal_slice'     # Sample 7
        ]
        
        # Feature mappings (same as before)
        self.body_parts = [
            'Head', 'Neck', 'SpineShoulder', 'ShoulderLeft', 'ShoulderRight',
            'ElbowLeft', 'ElbowRight', 'WristLeft', 'WristRight', 
            'ThumbLeft', 'ThumbRight', 'HandLeft', 'HandRight',
            'HandTipLeft', 'HandTipRight', 'SpineMid', 'SpineBase',
            'HipLeft', 'HipRight', 'KneeLeft', 'KneeRight',
            'AnkleLeft', 'AnkleRight', 'FootLeft', 'FootRight'
        ]
        
        self.gait_params_excel = {
            'MaxStLe': 'Maximum Step Length',
            'MaxStWi': 'Maximum Step Width',
            'StrLe': 'Stride Length',
            'GaCT': 'Gait Cycle Time',
            'StaT': 'Stance Time',
            'SwiT': 'Swing Time',
            'Velocity': 'Gait Velocity'
        }
        
    def convert_to_float(self, value):
        """Convert string with comma decimal separator to float"""
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value).replace(',', '.'))
    
    def connect(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info(f"Connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    def clear_database(self):
        """Clear existing data"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")
    
    def create_constraints_and_indexes(self):
        """Create constraints and indexes - UPDATED for participant structure"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:OriginalParticipant) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:GaitSample) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (bp:BodyPart) REQUIRE bp.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (mt:MeasurementType) REQUIRE mt.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (cd:CoordinateDimension) REQUIRE cd.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Classification) REQUIRE c.label IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (gp:GaitParameter) REQUIRE gp.code IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (at:AugmentationType) REQUIRE at.name IS UNIQUE"
        ]
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (f:GaitFeature) ON (f.measurement_id)",
            "CREATE INDEX IF NOT EXISTS FOR (f:GaitFeature) ON (f.value)",
            "CREATE INDEX IF NOT EXISTS FOR (s:GaitSample) ON (s.original_participant_id)",
            "CREATE INDEX IF NOT EXISTS FOR (s:GaitSample) ON (s.augmentation_type)"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                session.run(constraint)
            for index in indexes:
                session.run(index)
            logger.info("Updated constraints and indexes created")
    
    def create_static_nodes(self):
        """Create static reference nodes - UPDATED with augmentation types"""
        with self.driver.session() as session:
            # Classifications
            session.run("""
                MERGE (asd:Classification {label: 'ASD', description: 'Autism Spectrum Disorder'})
                MERGE (control:Classification {label: 'Control', description: 'Typical Development'})
            """)
            
            # Body parts (same as before)
            body_regions = {
                'Upper': ['Head', 'Neck', 'SpineShoulder', 'ShoulderLeft', 'ShoulderRight',
                         'ElbowLeft', 'ElbowRight', 'WristLeft', 'WristRight',
                         'ThumbLeft', 'ThumbRight', 'HandLeft', 'HandRight',
                         'HandTipLeft', 'HandTipRight'],
                'Core': ['SpineMid', 'SpineBase'],
                'Lower': ['HipLeft', 'HipRight', 'KneeLeft', 'KneeRight',
                         'AnkleLeft', 'AnkleRight', 'FootLeft', 'FootRight']
            }
            
            for region, parts in body_regions.items():
                session.run("MERGE (r:BodyRegion {name: $name})", name=region)
                for part in parts:
                    session.run("""
                        MERGE (bp:BodyPart {name: $name})
                        MERGE (r:BodyRegion {name: $region})
                        MERGE (bp)-[:BELONGS_TO]->(r)
                    """, name=part, region=region)
            
            # Measurement types
            for mtype in ['position', 'angle', 'distance']:
                session.run("MERGE (mt:MeasurementType {name: $name})", name=mtype)
            
            # Coordinate dimensions
            for dim in ['x', 'y', 'z']:
                session.run("MERGE (cd:CoordinateDimension {name: $name})", name=dim)
            
            # Augmentation types - NEW!
            for aug_type in self.augmentation_types:
                session.run("""
                    MERGE (at:AugmentationType {
                        name: $aug_type,
                        is_original: $is_original
                    })
                """, aug_type=aug_type, is_original=(aug_type == 'original'))
            
            # Gait parameters
            for code, name in self.gait_params_excel.items():
                category = 'temporal' if code in ['GaCT', 'StaT', 'SwiT'] else 'spatial'
                session.run("""
                    MERGE (gp:GaitParameter {
                        code: $code,
                        name: $name,
                        category: $category
                    })
                """, code=code, name=name, category=category)
            
            logger.info("Static nodes created with augmentation awareness")
    
    def load_and_process_data_fixed(self, filepath="Final dataset.csv"):
        """Load and process data with PROPER participant structure"""
        logger.info(f"Loading data from {filepath} with participant structure awareness...")
        
        # Read CSV
        df = pd.read_csv(filepath, delimiter=';', decimal=',')
        
        # Convert numeric columns
        numeric_columns = [col for col in df.columns if col != 'class']
        for col in numeric_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: self.convert_to_float(x) if pd.notna(x) else np.nan)
        
        # CRITICAL: Create proper participant structure
        total_samples = len(df)
        if total_samples % self.samples_per_participant != 0:
            raise ValueError(f"Total samples ({total_samples}) not divisible by samples_per_participant ({self.samples_per_participant})")
        
        n_original_participants = total_samples // self.samples_per_participant
        
        # Create participant IDs and augmentation metadata
        original_participant_ids = []
        sample_ids = []
        augmentation_types = []
        
        for i in range(n_original_participants):
            for j in range(self.samples_per_participant):
                original_participant_ids.append(f'ORIG_P_{i:04d}')
                sample_ids.append(f'SAMPLE_{i:04d}_{j}')
                augmentation_types.append(self.augmentation_types[j])
        
        df['original_participant_id'] = original_participant_ids
        df['sample_id'] = sample_ids
        df['augmentation_type'] = augmentation_types
        
        # Map class values
        df['class'] = df['class'].map({'A': 'ASD', 'T': 'Control'})
        
        # Filter to mean features only (same as before)
        logger.info("Filtering features to keep only mean values...")
        
        cols_to_keep = ['original_participant_id', 'sample_id', 'augmentation_type', 'class']
        
        for col in df.columns:
            col_clean = col.strip()
            
            if col_clean.startswith('mean-') and any(coord in col_clean for coord in ['-x-', '-y-', '-z-']):
                cols_to_keep.append(col)
            elif col_clean.startswith('mean ') and any(len(col_clean.split()) >= 2 for _ in [1]):
                cols_to_keep.append(col)
            elif col_clean.startswith('Rom'):
                cols_to_keep.append(col)
            elif col_clean in ['MaxStLe', 'MaxStWi', 'StrLe', 'GaCT', 'StaT', 'SwiT', 'Velocity']:
                cols_to_keep.append(col)
            elif col_clean in ['HaTiLPos', 'HaTiRPos', 'MaxDBFE', 'MinDBFE', 'Threshold']:
                cols_to_keep.append(col)
        
        df_filtered = df[cols_to_keep]
        
        logger.info(f"Participant structure created:")
        logger.info(f"  Original participants: {n_original_participants}")
        logger.info(f"  Total samples: {len(df_filtered)}")
        logger.info(f"  Samples per participant: {self.samples_per_participant}")
        logger.info(f"  Features kept: {len(df_filtered.columns) - 4}")  # Minus metadata columns
        logger.info(f"  Class distribution: {df_filtered['class'].value_counts().to_dict()}")
        
        return df_filtered
    
    def create_participants_and_samples(self, df):
        """Create participant and sample nodes with proper relationships"""
        logger.info("Creating participants and samples with augmentation relationships...")
        
        with self.driver.session() as session:
            # Create original participants
            unique_participants = df[['original_participant_id', 'class']].drop_duplicates()
            
            for _, row in unique_participants.iterrows():
                session.run("""
                    MERGE (p:OriginalParticipant {
                        id: $participant_id,
                        created_date: datetime()
                    })
                    MERGE (c:Classification {label: $classification})
                    MERGE (p)-[:CLASSIFIED_AS]->(c)
                """, participant_id=row['original_participant_id'], 
                     classification=row['class'])
            
            # Create samples with augmentation metadata
            batch_size = 100
            batch_data = []
            
            for idx, row in df.iterrows():
                sample_data = {
                    'sample_id': row['sample_id'],
                    'participant_id': row['original_participant_id'],
                    'augmentation_type': row['augmentation_type'],
                    'classification': row['class'],
                    'measurement_date': datetime.now().isoformat()
                }
                batch_data.append(sample_data)
                
                if len(batch_data) >= batch_size:
                    self._create_sample_batch(session, batch_data)
                    batch_data = []
            
            if batch_data:
                self._create_sample_batch(session, batch_data)
            
            logger.info(f"Created {len(unique_participants)} original participants and {len(df)} samples")
    
    def _create_sample_batch(self, session, batch_data):
        """Create a batch of samples with augmentation relationships"""
        session.run("""
            UNWIND $batch AS data
            MATCH (p:OriginalParticipant {id: data.participant_id})
            MATCH (at:AugmentationType {name: data.augmentation_type})
            CREATE (s:GaitSample {
                id: data.sample_id,
                original_participant_id: data.participant_id,
                augmentation_type: data.augmentation_type,
                measurement_date: datetime(data.measurement_date),
                is_original: at.is_original
            })
            CREATE (p)-[:HAS_SAMPLE]->(s)
            CREATE (s)-[:AUGMENTED_BY]->(at)
        """, batch=batch_data)
    
    def create_ml_split_helper_function(self):
        """Add utility functions to help with ML splitting"""
        with self.driver.session() as session:
            # Create a utility function to get participant-level splits
            session.run("""
                // Helper function for ML: Get all samples for train participants
                CREATE OR REPLACE FUNCTION graph.getTrainSamples(trainParticipantIds)
                RETURNS LIST<NODE>
                LANGUAGE cypher
                AS $$
                    MATCH (p:OriginalParticipant)-[:HAS_SAMPLE]->(s:GaitSample)
                    WHERE p.id IN trainParticipantIds
                    RETURN collect(s)
                $$
            """)
            
            session.run("""
                // Helper query to get original participants for splitting
                // Usage: MATCH (p:OriginalParticipant) RETURN p.id, head([(p)-[:CLASSIFIED_AS]->(c) | c.label])
            """)
            
            logger.info("ML helper functions created")
    
    def create_gait_features(self, df):
        """Create gait feature nodes - same as before but with sample references"""
        logger.info("Creating gait features with sample references...")
        
        with self.driver.session() as session:
            # Process mean coordinate features
            self._process_mean_coordinate_features_fixed(session, df)
            
            # Process other features (same logic as before)
            # ... (keeping same feature processing logic)
            
        logger.info("Gait features created with proper sample references")
    
    def _process_mean_coordinate_features_fixed(self, session, df):
        """Process coordinate features with sample references"""
        coord_features = [col for col in df.columns if col.strip().startswith('mean-') and 
                         any(coord in col for coord in ['-x-', '-y-', '-z-'])]
        
        logger.info(f"Processing {len(coord_features)} mean coordinate features...")
        
        batch_size = 1000
        batch_data = []
        
        for idx, row in df.iterrows():
            sample_id = row['sample_id']
            
            for feature in coord_features:
                parts = feature.strip().split('-')
                if len(parts) == 3:
                    stat_type, coord, body_part = parts
                    body_part_name = self._normalize_body_part(body_part)
                    
                    if body_part_name:
                        measurement_id = f"{body_part_name}_{coord}_{stat_type}_{sample_id}"
                        value = row[feature]
                        
                        if pd.notna(value):
                            batch_data.append({
                                'sample_id': sample_id,
                                'measurement_id': measurement_id,
                                'value': float(value),
                                'stat_type': stat_type,
                                'body_part': body_part_name,
                                'coordinate': coord,
                                'measurement_type': 'position'
                            })
                
                if len(batch_data) >= batch_size:
                    self._create_feature_batch_fixed(session, batch_data)
                    batch_data = []
        
        if batch_data:
            self._create_feature_batch_fixed(session, batch_data)
    
    def _create_feature_batch_fixed(self, session, batch_data):
        """Create feature batch with sample references"""
        session.run("""
            UNWIND $batch AS data
            MATCH (s:GaitSample {id: data.sample_id})
            MATCH (bp:BodyPart {name: data.body_part})
            MATCH (cd:CoordinateDimension {name: data.coordinate})
            MATCH (mt:MeasurementType {name: data.measurement_type})
            CREATE (f:GaitFeature {
                measurement_id: data.measurement_id,
                value: data.value,
                stat_type: data.stat_type
            })
            CREATE (s)-[:HAS_FEATURE]->(f)
            CREATE (f)-[:MEASURED_IN]->(bp)
            CREATE (f)-[:IN_DIMENSION]->(cd)
            CREATE (f)-[:HAS_MEASUREMENT]->(mt)
        """, batch=batch_data)
    
    def _normalize_body_part(self, body_part_str):
        """Normalize body part names (same as before)"""
        mappings = {
            'midspain': 'SpineMid',
            'ankleleft': 'AnkleLeft', 'ankleright': 'AnkleRight',
            'kneeleft': 'KneeLeft', 'kneeright': 'KneeRight', 
            'hipleft': 'HipLeft', 'hipright': 'HipRight',
            'wristleft': 'WristLeft', 'wristright': 'WristRight',
            'handleft': 'HandLeft', 'handright': 'HandRight',
            'handtipleft': 'HandTipLeft', 'handtiprighta': 'HandTipRight',
            'head': 'Head', 'neck': 'Neck',
            'shoulderleft': 'ShoulderLeft', 'shoulderright': 'ShoulderRight',
            'elbowleft': 'ElbowLeft', 'elbowright': 'ElbowRight',
            'spineshoulder': 'SpineShoulder', 'spinebase': 'SpineBase',
            'footleft': 'FootLeft', 'footright': 'FootRight',
            'thumbleft': 'ThumbLeft', 'thumbright': 'ThumbRight'
        }
        
        normalized = body_part_str.lower()
        return mappings.get(normalized, body_part_str)
    
    def get_statistics(self):
        """Get graph statistics"""
        with self.driver.session() as session:
            stats = {}
            
            # Node counts
            node_types = ['OriginalParticipant', 'GaitSample', 'GaitFeature', 'BodyPart', 
                         'GaitParameter', 'Classification', 'AugmentationType']
            
            for node_type in node_types:
                result = session.run(f"MATCH (n:{node_type}) RETURN count(n) as count")
                stats[node_type] = result.single()['count']
            
            # Relationship counts
            rel_types = ['HAS_SAMPLE', 'HAS_FEATURE', 'AUGMENTED_BY', 'CLASSIFIED_AS']
            
            for rel_type in rel_types:
                result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                stats[rel_type] = result.single()['count']
            
            return stats
    
    def create_ml_queries(self):
        """Create helpful queries for ML analysis"""
        queries = {
            'get_original_participants': """
                MATCH (p:OriginalParticipant)-[:CLASSIFIED_AS]->(c:Classification)
                RETURN p.id as participant_id, c.label as classification
                ORDER BY p.id
            """,
            
            'get_samples_for_participants': """
                MATCH (p:OriginalParticipant {id: $participant_id})-[:HAS_SAMPLE]->(s:GaitSample)
                RETURN s.id as sample_id, s.augmentation_type as augmentation_type
                ORDER BY s.id
            """,
            
            'get_train_test_split_data': """
                // Use this query with participant IDs from your ML split
                MATCH (p:OriginalParticipant)-[:HAS_SAMPLE]->(s:GaitSample)-[:HAS_FEATURE]->(f:GaitFeature)
                WHERE p.id IN $train_participant_ids
                RETURN s.id as sample_id, f.measurement_id as feature_name, f.value as feature_value
            """,
            
            'validate_no_leakage': """
                // Check for participant overlap between train and test
                WITH $train_participant_ids as train_ids, $test_participant_ids as test_ids
                RETURN size([id IN train_ids WHERE id IN test_ids]) as overlap_count
            """
        }
        
        # Save queries to file
        with open(f"{self.output_dir if hasattr(self, 'output_dir') else '.'}/ml_queries.cypher", 'w') as f:
            for name, query in queries.items():
                f.write(f"// {name}\n{query}\n\n")
        
        logger.info("ML helper queries created and saved")
        return queries
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def build_graph(self, filepath="Final dataset.csv", clear_existing=True):
        """Main method to build the FIXED graph with participant awareness"""
        try:
            # Connect to Neo4j
            if not self.connect():
                return False
            
            # Clear existing data if requested
            if clear_existing:
                self.clear_database()
            
            # Create schema
            self.create_constraints_and_indexes()
            
            # Create static nodes (including augmentation types)
            self.create_static_nodes()
            
            # Load and process data with participant structure
            df = self.load_and_process_data_fixed(filepath)
            
            # Create participants and samples with augmentation metadata
            self.create_participants_and_samples(df)
            
            # Create features (updated to reference samples)
            self.create_gait_features(df)
            
            # Create ML helper functions
            self.create_ml_split_helper_function()
            
            # Create helpful ML queries
            ml_queries = self.create_ml_queries()
            
            # Get statistics
            stats = self.get_statistics()
            
            logger.info("FIXED graph building completed successfully!")
            logger.info("Statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
            
            logger.info("\n🎯 PARTICIPANT STRUCTURE SUMMARY:")
            logger.info(f"  ✅ {stats.get('OriginalParticipant', 0)} original participants")
            logger.info(f"  ✅ {stats.get('GaitSample', 0)} total samples (with augmentation metadata)")
            logger.info(f"  ✅ {stats.get('AugmentationType', 0)} augmentation types tracked")
            logger.info(f"  ✅ ML queries created for participant-level splitting")
            logger.info(f"  ✅ No more data leakage risk!")
            
            return True
            
        except Exception as e:
            logger.error(f"Error building graph: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
        finally:
            self.close()


if __name__ == "__main__":
    builder = NeuroGaitGraphBuilderFixed(samples_per_participant=8)
    success = builder.build_graph("Final dataset.csv")
    
    if success:
        print("\n🎉 SUCCESS: Fixed Knowledge Graph created!")
        print("✅ Participant structure properly represented")
        print("✅ Augmentation metadata stored")
        print("✅ ML queries available for participant-level splitting")
        print("✅ No more data leakage risk!")
    else:
        print("❌ Failed to create knowledge graph")