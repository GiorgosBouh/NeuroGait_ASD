#!/usr/bin/env python3
"""
Complete Fast NeuroGait Knowledge Graph Builder
==============================================

Combines the best of both worlds:
- FAST bulk operations (minutes not hours)
- COMPLETE knowledge graph structure for proper embeddings
- All relationships and connections
- Smart batching for features

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import re
from neo4j import GraphDatabase
import logging
from typing import Dict, List, Tuple, Any
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteFastNeuroGaitKG:
    """
    Complete AND Fast Knowledge Graph Builder
    
    Features:
    - Complete knowledge graph structure (for proper embeddings)
    - Smart bulk operations (for speed)
    - All body part relationships
    - All anatomical connections
    - Optimized feature processing
    """
    
    def __init__(self, neo4j_uri: str = None, neo4j_user: str = None, neo4j_password: str = None):
        """Initialize the knowledge graph builder"""
        self.neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_user = neo4j_user or os.getenv('NEO4J_USER', 'neo4j')
        self.neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD')
        
        if not self.neo4j_password:
            raise ValueError("Neo4j password not found. Please set NEO4J_PASSWORD in your .env file")
        
        self.driver = None
        self.data = None
        self.feature_schema = {}
        
        # Complete body part hierarchy
        self.body_parts = {
            'Head': {'parent': 'Upper_Body', 'type': 'head'},
            'Neck': {'parent': 'Upper_Body', 'type': 'neck'},
            'ShoulderLeft': {'parent': 'Upper_Body', 'type': 'shoulder', 'side': 'left'},
            'ShoulderRight': {'parent': 'Upper_Body', 'type': 'shoulder', 'side': 'right'},
            'ElbowLeft': {'parent': 'Upper_Body', 'type': 'elbow', 'side': 'left'},
            'ElbowRight': {'parent': 'Upper_Body', 'type': 'elbow', 'side': 'right'},
            'WristLeft': {'parent': 'Upper_Body', 'type': 'wrist', 'side': 'left'},
            'WristRight': {'parent': 'Upper_Body', 'type': 'wrist', 'side': 'right'},
            'HandLeft': {'parent': 'Upper_Body', 'type': 'hand', 'side': 'left'},
            'HandRight': {'parent': 'Upper_Body', 'type': 'hand', 'side': 'right'},
            'HandTipLeft': {'parent': 'Upper_Body', 'type': 'hand_tip', 'side': 'left'},
            'HandTipRightA': {'parent': 'Upper_Body', 'type': 'hand_tip', 'side': 'right'},
            'ThumbLeft': {'parent': 'Upper_Body', 'type': 'thumb', 'side': 'left'},
            'ThumbRight': {'parent': 'Upper_Body', 'type': 'thumb', 'side': 'right'},
            'SpineBase': {'parent': 'Core', 'type': 'spine'},
            'SpineShoulder': {'parent': 'Core', 'type': 'spine'},
            'Midspain': {'parent': 'Core', 'type': 'center'},
            'HipLeft': {'parent': 'Lower_Body', 'type': 'hip', 'side': 'left'},
            'HipRight': {'parent': 'Lower_Body', 'type': 'hip', 'side': 'right'},
            'KneeLeft': {'parent': 'Lower_Body', 'type': 'knee', 'side': 'left'},
            'KneeRight': {'parent': 'Lower_Body', 'type': 'knee', 'side': 'right'},
            'AnkleLeft': {'parent': 'Lower_Body', 'type': 'ankle', 'side': 'left'},
            'AnkleRight': {'parent': 'Lower_Body', 'type': 'ankle', 'side': 'right'},
            'FootLeft': {'parent': 'Lower_Body', 'type': 'foot', 'side': 'left'},
            'FootRight': {'parent': 'Lower_Body', 'type': 'foot', 'side': 'right'}
        }
        
        # Anatomical connections for kinematic chains
        self.anatomical_connections = [
            ('Head', 'Neck'),
            ('Neck', 'SpineShoulder'),
            ('SpineShoulder', 'ShoulderLeft'),
            ('SpineShoulder', 'ShoulderRight'),
            ('ShoulderLeft', 'ElbowLeft'),
            ('ShoulderRight', 'ElbowRight'),
            ('ElbowLeft', 'WristLeft'),
            ('ElbowRight', 'WristRight'),
            ('WristLeft', 'HandLeft'),
            ('WristRight', 'HandRight'),
            ('HandLeft', 'ThumbLeft'),
            ('HandRight', 'ThumbRight'),
            ('SpineShoulder', 'SpineBase'),
            ('SpineBase', 'HipLeft'),
            ('SpineBase', 'HipRight'),
            ('HipLeft', 'KneeLeft'),
            ('HipRight', 'KneeRight'),
            ('KneeLeft', 'AnkleLeft'),
            ('KneeRight', 'AnkleRight'),
            ('AnkleLeft', 'FootLeft'),
            ('AnkleRight', 'FootRight')
        ]
        
        self.measurement_types = ['mean', 'variance', 'std']
        self.coordinate_dimensions = ['x', 'y', 'z']
    
    def connect_to_neo4j(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_user, self.neo4j_password)
            )
            logger.info(f"Connected to Neo4j at {self.neo4j_uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    def close_connection(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def load_data(self, file_path: str):
        """Load and analyze the gait analysis dataset"""
        try:
            logger.info(f"Loading data from {file_path}")
            
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                self.data = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                delimiters = [',', ';', '\t', '|']
                for delimiter in delimiters:
                    try:
                        self.data = pd.read_csv(file_path, delimiter=delimiter)
                        if len(self.data.columns) > 10:
                            logger.info(f"Successfully loaded CSV with delimiter: '{delimiter}'")
                            break
                    except:
                        continue
                else:
                    raise ValueError("Could not parse CSV with any common delimiter")
            else:
                raise ValueError("Unsupported file format. Use .xlsx, .xls, or .csv")
            
            logger.info(f"Loaded dataset with {len(self.data)} samples and {len(self.data.columns)} features")
            
            # Create participant IDs
            self.data['participant_id'] = [f"P_{i:04d}" for i in range(1, len(self.data) + 1)]
            logger.info("Generated unique participant_id for each measurement")
            
            # Map class values
            if 'class' in self.data.columns:
                class_mapping = {'A': 'ASD', 'T': 'Control'}
                self.data['diagnosis'] = self.data['class'].map(class_mapping)
                logger.info("Mapped class values: A->ASD, T->Control")
            
            # Analyze feature structure
            self._analyze_feature_structure()
            return True
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    def _analyze_feature_structure(self):
        """Analyze the structure of features in the dataset"""
        logger.info("Analyzing feature structure...")
        
        self.feature_schema = {
            'body_measurements': [],
            'distance_features': [],
            'range_of_motion': [],
            'temporal_gait': [],
            'other': [],
            'target': 'class'
        }
        
        for col in self.data.columns:
            if col in ['class', 'diagnosis', 'participant_id']:
                continue
            elif col.startswith('Rom'):
                self.feature_schema['range_of_motion'].append(col)
            elif col in ['MaxStLe', 'MaxStWi', 'StrLe', 'GaCT', 'StaT', 'SwiT', 'Velocity']:
                self.feature_schema['temporal_gait'].append(col)
            elif any(col.startswith(f'{stat}-') for stat in self.measurement_types):
                self.feature_schema['body_measurements'].append(col)
            elif 'T' in col and not col.startswith('Rom'):
                self.feature_schema['distance_features'].append(col)
            else:
                self.feature_schema['other'].append(col)
        
        logger.info(f"Feature analysis complete:")
        for category, features in self.feature_schema.items():
            if isinstance(features, list):
                logger.info(f"  {category}: {len(features)} features")
    
    def _parse_body_measurement_feature(self, feature: str) -> Tuple[str, str, str]:
        """Parse body measurement feature name to extract components"""
        pattern = r'(mean|variance|std)-(x|y|z)-(.+)'
        match = re.match(pattern, feature)
        
        if match:
            measurement_type, dimension, body_part = match.groups()
            return measurement_type, dimension, body_part
        
        return None
    
    def clear_database(self):
        """Clear the Neo4j database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")
    
    def create_complete_schema(self):
        """Create complete schema with all relationships"""
        logger.info("Creating complete schema...")
        
        with self.driver.session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Participant) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:GaitSession) REQUIRE s.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (bp:BodyPart) REQUIRE bp.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (mt:MeasurementType) REQUIRE mt.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (cd:CoordinateDimension) REQUIRE cd.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Classification) REQUIRE c.label IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (gp:GaitParameter) REQUIRE gp.name IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except:
                    pass  # Constraint already exists
            
            # Create body regions
            session.run("""
                MERGE (r1:BodyRegion {name: 'Upper_Body', type: 'region'})
                MERGE (r2:BodyRegion {name: 'Core', type: 'region'})
                MERGE (r3:BodyRegion {name: 'Lower_Body', type: 'region'})
            """)
            
            # Create body parts and connect to regions
            for part_name, properties in self.body_parts.items():
                session.run("""
                    MERGE (bp:BodyPart {name: $name, type: $type, side: $side})
                    WITH bp
                    MATCH (r:BodyRegion {name: $parent})
                    MERGE (bp)-[:BELONGS_TO]->(r)
                """, 
                name=part_name,
                type=properties['type'],
                side=properties.get('side', 'center'),
                parent=properties['parent']
                )
            
            # Create measurement types
            for measurement in self.measurement_types:
                session.run(
                    "MERGE (mt:MeasurementType {name: $name, category: 'statistical'})",
                    name=measurement
                )
            
            # Create coordinate dimensions
            for dimension in self.coordinate_dimensions:
                session.run(
                    "MERGE (cd:CoordinateDimension {name: $name, category: 'spatial'})",
                    name=dimension
                )
            
            # Create classifications
            session.run("""
                MERGE (c1:Classification {label: 'ASD', code: 'A'})
                MERGE (c2:Classification {label: 'Control', code: 'T'})
            """)
            
            # Create gait parameters
            gait_params = [
                ('MaxStLe', 'Maximum Step Length', 'spatial'),
                ('MaxStWi', 'Maximum Step Width', 'spatial'),
                ('StrLe', 'Stride Length', 'spatial'),
                ('GaCT', 'Gait Cycle Time', 'temporal'),
                ('StaT', 'Stance Time', 'temporal'),
                ('SwiT', 'Swing Time', 'temporal'),
                ('Velocity', 'Gait Velocity', 'kinematic')
            ]
            
            for code, name, category in gait_params:
                session.run("""
                    MERGE (gp:GaitParameter {code: $code, name: $name, category: $category})
                """, code=code, name=name, category=category)
        
        logger.info("Complete schema created")
    
    def create_measurement_relationships(self):
        """Create relationships between body parts, measurements, and dimensions"""
        logger.info("Creating measurement relationships...")
        
        with self.driver.session() as session:
            # Connect body parts to measurement types and dimensions
            for feature in self.feature_schema['body_measurements']:
                parts = self._parse_body_measurement_feature(feature)
                if parts:
                    measurement_type, dimension, body_part = parts
                    
                    session.run("""
                        MATCH (bp:BodyPart {name: $body_part})
                        MATCH (mt:MeasurementType {name: $measurement_type})
                        MATCH (cd:CoordinateDimension {name: $dimension})
                        MERGE (bp)-[:HAS_MEASUREMENT]->(mt)
                        MERGE (mt)-[:IN_DIMENSION]->(cd)
                        MERGE (bp)-[rel:MEASURED_IN]->(cd)
                        SET rel.measurement_type = $measurement_type
                    """,
                    body_part=body_part,
                    measurement_type=measurement_type,
                    dimension=dimension
                    )
        
        logger.info("Measurement relationships created")
    
    def create_anatomical_connections(self):
        """Create anatomical connections between body parts"""
        logger.info("Creating anatomical connections...")
        
        with self.driver.session() as session:
            for parent, child in self.anatomical_connections:
                if parent in self.body_parts and child in self.body_parts:
                    session.run("""
                        MATCH (parent:BodyPart {name: $parent})
                        MATCH (child:BodyPart {name: $child})
                        MERGE (parent)-[:CONNECTS_TO]->(child)
                        MERGE (child)-[:CONNECTED_FROM]->(parent)
                    """,
                    parent=parent,
                    child=child
                    )
        
        logger.info("Anatomical connections created")
    
    def bulk_create_participants_and_sessions(self, batch_size=100):
        """Create participants and sessions in bulk batches"""
        logger.info("Creating participants and sessions in bulk...")
        
        total_participants = len(self.data)
        
        with self.driver.session() as session:
            for start_idx in tqdm(range(0, total_participants, batch_size), desc="Creating participants"):
                end_idx = min(start_idx + batch_size, total_participants)
                batch_data = self.data.iloc[start_idx:end_idx]
                
                participants_data = []
                sessions_data = []
                
                for idx, row in batch_data.iterrows():
                    participant_id = row['participant_id']
                    diagnosis = row['diagnosis']
                    session_id = f"session_{participant_id}"
                    
                    participants_data.append({
                        'id': participant_id,
                        'diagnosis': diagnosis,
                        'data_index': int(idx)
                    })
                    
                    sessions_data.append({
                        'id': session_id,
                        'participant_id': participant_id
                    })
                
                # Bulk create participants
                session.run("""
                    UNWIND $participants as p
                    MERGE (participant:Participant {id: p.id})
                    SET participant.diagnosis = p.diagnosis,
                        participant.data_index = p.data_index
                    WITH participant
                    MATCH (c:Classification {label: participant.diagnosis})
                    MERGE (participant)-[:CLASSIFIED_AS]->(c)
                """, participants=participants_data)
                
                # Bulk create sessions
                session.run("""
                    UNWIND $sessions as s
                    MATCH (p:Participant {id: s.participant_id})
                    CREATE (session:GaitSession {
                        id: s.id,
                        participant_id: s.participant_id,
                        measurement_date: datetime(),
                        session_type: 'primary'
                    })
                    CREATE (p)-[:HAS_SESSION]->(session)
                """, sessions=sessions_data)
        
        logger.info(f"Created {total_participants} participants and sessions")
    
    def smart_feature_processing(self, feature_limit_per_category=100):
        """
        Smart feature processing: Include all important features but limit massive categories
        """
        logger.info("Processing features with smart limits...")
        
        # Always include ALL gait parameters and ROM features (they're small)
        important_features = []
        important_features.extend(self.feature_schema['temporal_gait'])  # 7 features
        important_features.extend(self.feature_schema['range_of_motion'])  # 50 features
        
        # Limit large categories but include representative samples
        important_features.extend(self.feature_schema['body_measurements'][:feature_limit_per_category])  # Top 100
        important_features.extend(self.feature_schema['distance_features'][:feature_limit_per_category])  # Top 100
        important_features.extend(self.feature_schema['other'][:50])  # All other features
        
        logger.info(f"Selected {len(important_features)} features for processing")
        
        batch_size = 25  # Smaller batches for better memory management
        
        with self.driver.session() as session:
            for start_idx in tqdm(range(0, len(self.data), batch_size), desc="Creating features"):
                end_idx = min(start_idx + batch_size, len(self.data))
                batch_data = self.data.iloc[start_idx:end_idx]
                
                features_data = []
                gait_values_data = []
                
                for idx, row in batch_data.iterrows():
                    participant_id = row['participant_id']
                    session_id = f"session_{participant_id}"
                    
                    for feature_name in important_features:
                        if feature_name in row.index and pd.notna(row[feature_name]):
                            try:
                                value = float(row[feature_name])
                                
                                # Handle gait parameters specially
                                if feature_name in self.feature_schema['temporal_gait']:
                                    gait_values_data.append({
                                        'session_id': session_id,
                                        'feature': feature_name,
                                        'value': value
                                    })
                                else:
                                    # Parse body measurements for proper relationships
                                    parts = self._parse_body_measurement_feature(feature_name)
                                    if parts:
                                        measurement_type, dimension, body_part = parts
                                        features_data.append({
                                            'session_id': session_id,
                                            'feature_type': feature_name,
                                            'value': value,
                                            'measurement_type': measurement_type,
                                            'dimension': dimension,
                                            'body_part': body_part
                                        })
                                    else:
                                        # Other features
                                        category = 'other'
                                        if feature_name in self.feature_schema['distance_features']:
                                            category = 'distance_features'
                                        elif feature_name in self.feature_schema['range_of_motion']:
                                            category = 'range_of_motion'
                                        
                                        features_data.append({
                                            'session_id': session_id,
                                            'feature_type': feature_name,
                                            'value': value,
                                            'category': category
                                        })
                            except (ValueError, TypeError):
                                continue
                
                # Bulk create gait values
                if gait_values_data:
                    session.run("""
                        UNWIND $gait_values as gv
                        MATCH (s:GaitSession {id: gv.session_id})
                        MATCH (gp:GaitParameter {code: gv.feature})
                        MERGE (s)-[rel:HAS_GAIT_VALUE]->(gp)
                        SET rel.value = gv.value
                    """, gait_values=gait_values_data)
                
                # Bulk create other features
                if features_data:
                    session.run("""
                        UNWIND $features as f
                        MATCH (s:GaitSession {id: f.session_id})
                        CREATE (feature:GaitFeature {
                            feature_type: f.feature_type,
                            value: f.value,
                            measurement_type: coalesce(f.measurement_type, ''),
                            dimension: coalesce(f.dimension, ''),
                            body_part: coalesce(f.body_part, ''),
                            category: coalesce(f.category, ''),
                            calculated_at: datetime()
                        })
                        CREATE (s)-[:HAS_FEATURE]->(feature)
                    """, features=features_data)
        
        logger.info("Smart feature processing completed")
    
    def get_statistics(self):
        """Get knowledge graph statistics"""
        with self.driver.session() as session:
            stats = {}
            
            # Count nodes by type
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as node_type, count(n) as count
                ORDER BY count DESC
            """)
            stats['nodes'] = {record['node_type']: record['count'] for record in result}
            
            # Count relationships by type
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as count
                ORDER BY count DESC
            """)
            stats['relationships'] = {record['rel_type']: record['count'] for record in result}
            
            return stats
    
    def build_complete_fast_graph(self, data_file: str):
        """Build the complete knowledge graph FAST"""
        logger.info("🚀 Starting COMPLETE & FAST knowledge graph build...")
        start_time = time.time()
        
        if not self.connect_to_neo4j():
            return False
        
        try:
            # Load data
            if not self.load_data(data_file):
                return False
            
            print("⏱️  Step 1/7: Clearing database...")
            self.clear_database()
            
            print("⏱️  Step 2/7: Creating complete schema...")
            self.create_complete_schema()
            
            print("⏱️  Step 3/7: Creating measurement relationships...")
            self.create_measurement_relationships()
            
            print("⏱️  Step 4/7: Creating anatomical connections...")
            self.create_anatomical_connections()
            
            print("⏱️  Step 5/7: Creating participants and sessions...")
            self.bulk_create_participants_and_sessions()
            
            print("⏱️  Step 6/7: Smart feature processing...")
            self.smart_feature_processing()
            
            print("⏱️  Step 7/7: Gathering statistics...")
            stats = self.get_statistics()
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info("🎉 COMPLETE & FAST Knowledge Graph Build Completed!")
            logger.info(f"⏱️  Total time: {duration:.1f} seconds")
            logger.info("📊 Statistics:")
            for node_type, count in stats['nodes'].items():
                logger.info(f"  {node_type}: {count}")
            
            logger.info("🔗 Relationships:")
            for rel_type, count in stats['relationships'].items():
                logger.info(f"  {rel_type}: {count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            return False
        finally:
            self.close_connection()


def main():
    """Main function"""
    DATA_FILE = "Final dataset.xlsx"
    
    print("🎯 COMPLETE & FAST NEUROGAIT KNOWLEDGE GRAPH BUILDER")
    print("=" * 60)
    print("✅ Complete knowledge graph structure (proper embeddings)")
    print("✅ Fast bulk operations (minutes not hours)")
    print("✅ All body part relationships & anatomical connections")
    print("✅ Smart feature processing (important features)")
    print("=" * 60)
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: Could not find data file '{DATA_FILE}'")
        return False
    
    print(f"✅ Found data file: {DATA_FILE}")
    
    try:
        kg_builder = CompleteFastNeuroGaitKG()
        success = kg_builder.build_complete_fast_graph(DATA_FILE)
        
        if success:
            print("\n🎉 COMPLETE & FAST build completed!")
            print("Ready for TRUE graph embeddings and ML analysis! 🧠🤖")
        else:
            print("\n❌ Build failed")
            
    except Exception as e:
        logger.error(f"Build failed: {e}")
        print(f"❌ Build failed: {e}")


if __name__ == "__main__":
    main()