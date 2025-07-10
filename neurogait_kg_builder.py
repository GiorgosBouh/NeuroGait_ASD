"""
NeuroGait ASD Knowledge Graph Builder - Mean Features Only
Eliminates redundancy by using only mean features (not variance/std)
Based on Kinect v2 3D skeletal data
Fixed to handle comma decimal separators
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

# Load environment variables
load_dotenv('.env')

class NeuroGaitGraphBuilderMeanOnly:
    def __init__(self):
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = os.getenv('NEO4J_USER', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD', 'your_password')
        self.driver = None
        
        # Feature mappings based on documentation
        self.body_parts = [
            'Head', 'Neck', 'SpineShoulder', 'ShoulderLeft', 'ShoulderRight',
            'ElbowLeft', 'ElbowRight', 'WristLeft', 'WristRight', 
            'ThumbLeft', 'ThumbRight', 'HandLeft', 'HandRight',
            'HandTipLeft', 'HandTipRight', 'SpineMid', 'SpineBase',
            'HipLeft', 'HipRight', 'KneeLeft', 'KneeRight',
            'AnkleLeft', 'AnkleRight', 'FootLeft', 'FootRight'
        ]
        
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
        
        # Gait parameters from Excel
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
        # Replace comma with dot for decimal separator
        return float(str(value).replace(',', '.'))
    
    def connect(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logging.info(f"Connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    def clear_database(self):
        """Clear existing data"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logging.info("Database cleared")
    
    def create_constraints_and_indexes(self):
        """Create constraints and indexes for performance"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Participant) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:GaitSession) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (bp:BodyPart) REQUIRE bp.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (mt:MeasurementType) REQUIRE mt.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (cd:CoordinateDimension) REQUIRE cd.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Classification) REQUIRE c.label IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (gp:GaitParameter) REQUIRE gp.code IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (at:AngleType) REQUIRE at.code IS UNIQUE"
        ]
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (f:GaitFeature) ON (f.measurement_id)",
            "CREATE INDEX IF NOT EXISTS FOR (f:GaitFeature) ON (f.value)",
            "CREATE INDEX IF NOT EXISTS FOR ()-[r:HAS_GAIT_VALUE]-() ON (r.value)"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                session.run(constraint)
            for index in indexes:
                session.run(index)
            logging.info("Constraints and indexes created")
    
    def create_static_nodes(self):
        """Create static reference nodes"""
        with self.driver.session() as session:
            # Classifications
            session.run("""
                MERGE (asd:Classification {label: 'ASD', description: 'Autism Spectrum Disorder'})
                MERGE (control:Classification {label: 'Control', description: 'Typical Development'})
            """)
            
            # Body parts with regions
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
            
            # Angle types
            for code, description in self.angle_mappings.items():
                session.run("""
                    MERGE (at:AngleType {
                        code: $code,
                        description: $description
                    })
                """, code=code, description=description)
            
            logging.info("Static nodes created")
    
    def create_anatomical_connections(self):
        """Create anatomical connections between body parts"""
        connections = [
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
            ('WristLeft', 'ThumbLeft'),
            ('WristRight', 'ThumbRight'),
            ('HandLeft', 'HandTipLeft'),
            ('HandRight', 'HandTipRight'),
            ('SpineShoulder', 'SpineMid'),
            ('SpineMid', 'SpineBase'),
            ('SpineBase', 'HipLeft'),
            ('SpineBase', 'HipRight'),
            ('HipLeft', 'KneeLeft'),
            ('HipRight', 'KneeRight'),
            ('KneeLeft', 'AnkleLeft'),
            ('KneeRight', 'AnkleRight'),
            ('AnkleLeft', 'FootLeft'),
            ('AnkleRight', 'FootRight')
        ]
        
        with self.driver.session() as session:
            for part1, part2 in connections:
                session.run("""
                    MATCH (bp1:BodyPart {name: $part1})
                    MATCH (bp2:BodyPart {name: $part2})
                    MERGE (bp1)-[:CONNECTS_TO]->(bp2)
                    MERGE (bp2)-[:CONNECTED_FROM]->(bp1)
                """, part1=part1, part2=part2)
            
            logging.info("Anatomical connections created")
    
    def load_and_process_data(self, filepath="Final dataset.csv"):
        """Load and process the CSV dataset with mean features only"""
        logging.info(f"Loading data from {filepath}")
        
        # Read CSV with semicolon delimiter and comma as decimal separator
        df = pd.read_csv(filepath, delimiter=';', decimal=',')
        
        # If decimal parameter didn't work (older pandas), convert manually
        numeric_columns = [col for col in df.columns if col != 'class']
        for col in numeric_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: self.convert_to_float(x) if pd.notna(x) else np.nan)
        
        # Generate participant IDs
        df['participant_id'] = [f'P_{i:04d}' for i in range(1, len(df) + 1)]
        
        # Map class values
        df['class'] = df['class'].map({'A': 'ASD', 'T': 'Control'})
        
        # Filter to keep only mean features (eliminate redundancy)
        logging.info("Filtering features to keep only mean values...")
        
        original_cols = len(df.columns)
        
        # Keep only mean features + other non-redundant features
        cols_to_keep = []
        
        for col in df.columns:
            col_clean = col.strip()
            
            # Keep mean coordinate features
            if col_clean.startswith('mean-') and any(coord in col_clean for coord in ['-x-', '-y-', '-z-']):
                cols_to_keep.append(col)
            
            # Keep mean angle features
            elif col_clean.startswith('mean ') and any(angle in col_clean for angle in self.angle_mappings.keys()):
                cols_to_keep.append(col)
            
            # Keep mean distance features
            elif col_clean.startswith('mean ') and len(col_clean.split()) == 2 and not col_clean.startswith('mean-'):
                cols_to_keep.append(col)
            
            # Keep ROM features (no redundancy here)
            elif col_clean.startswith('Rom'):
                cols_to_keep.append(col)
            
            # Keep gait parameters (no redundancy here)
            elif col_clean in ['MaxStLe', 'MaxStWi', 'StrLe', 'GaCT', 'StaT', 'SwiT', 'Velocity']:
                cols_to_keep.append(col)
            
            # Keep other single features
            elif col_clean in ['HaTiLPos', 'HaTiRPos', 'MaxDBFE', 'MinDBFE', 'Threshold', 'class', 'participant_id']:
                cols_to_keep.append(col)
        
        # Filter dataset
        df_filtered = df[cols_to_keep]
        
        logging.info(f"Feature filtering results:")
        logging.info(f"  Original features: {original_cols}")
        logging.info(f"  Filtered features: {len(df_filtered.columns)}")
        logging.info(f"  Redundancy eliminated: {original_cols - len(df_filtered.columns)} features")
        logging.info(f"  Data reduction: {((original_cols - len(df_filtered.columns)) / original_cols * 100):.1f}%")
        
        logging.info(f"Loaded {len(df_filtered)} samples")
        logging.info(f"Class distribution: {df_filtered['class'].value_counts().to_dict()}")
        
        return df_filtered
    
    def create_participants_and_sessions(self, df):
        """Create participant and session nodes"""
        with self.driver.session() as session:
            batch_data = []
            
            for idx, row in df.iterrows():
                participant_data = {
                    'participant_id': row['participant_id'],
                    'session_id': f"session_{row['participant_id']}",
                    'classification': row['class'],
                    'measurement_date': datetime.now().isoformat()
                }
                batch_data.append(participant_data)
                
                if len(batch_data) >= 100:
                    self._create_participant_batch(session, batch_data)
                    batch_data = []
            
            if batch_data:
                self._create_participant_batch(session, batch_data)
            
            logging.info(f"Created {len(df)} participants and sessions")
    
    def _create_participant_batch(self, session, batch_data):
        """Create a batch of participants"""
        session.run("""
            UNWIND $batch AS data
            MERGE (p:Participant {id: data.participant_id})
            MERGE (s:GaitSession {
                id: data.session_id,
                participant_id: data.participant_id,
                measurement_date: datetime(data.measurement_date),
                session_type: 'primary'
            })
            MERGE (c:Classification {label: data.classification})
            MERGE (p)-[:HAS_SESSION]->(s)
            MERGE (s)-[:CLASSIFIED_AS]->(c)
        """, batch=batch_data)
    
    def create_gait_features(self, df):
        """Create gait feature nodes from data - mean features only"""
        logging.info("Creating gait features (mean values only)...")
        
        with self.driver.session() as session:
            # Process coordinate features (mean only)
            self._process_mean_coordinate_features(session, df)
            
            # Process angle features (mean only)
            self._process_mean_angle_features(session, df)
            
            # Process distance features (mean only)
            self._process_mean_distance_features(session, df)
            
            # Process ROM features
            self._process_rom_features(session, df)
            
            # Process gait parameters
            self._process_gait_parameters(session, df)
    
    def _process_mean_coordinate_features(self, session, df):
        """Process coordinate-based features - mean only"""
        coord_features = [col for col in df.columns if col.strip().startswith('mean-') and 
                         any(coord in col for coord in ['-x-', '-y-', '-z-'])]
        
        logging.info(f"Processing {len(coord_features)} mean coordinate features...")
        
        batch_size = 1000
        batch_data = []
        
        for idx, row in df.iterrows():
            participant_id = row['participant_id']
            session_id = f"session_{participant_id}"
            
            for feature in coord_features:
                parts = feature.strip().split('-')
                if len(parts) == 3:
                    stat_type, coord, body_part = parts
                    
                    # Convert body part name
                    body_part_name = self._normalize_body_part(body_part)
                    
                    if body_part_name:
                        measurement_id = f"{body_part_name}_{coord}_{stat_type}"
                        value = row[feature]
                        
                        # Skip if value is NaN
                        if pd.notna(value):
                            batch_data.append({
                                'session_id': session_id,
                                'measurement_id': measurement_id,
                                'value': float(value),  # Already converted during load
                                'stat_type': stat_type,
                                'body_part': body_part_name,
                                'coordinate': coord,
                                'measurement_type': 'position'
                            })
                
                if len(batch_data) >= batch_size:
                    self._create_feature_batch(session, batch_data)
                    batch_data = []
        
        if batch_data:
            self._create_feature_batch(session, batch_data)
    
    def _process_mean_angle_features(self, session, df):
        """Process angle features - mean only"""
        angle_features = [col for col in df.columns if col.strip().startswith('mean ') and 
                         any(angle in col for angle in self.angle_mappings.keys())]
        
        logging.info(f"Processing {len(angle_features)} mean angle features...")
        
        batch_data = []
        for idx, row in df.iterrows():
            participant_id = row['participant_id']
            session_id = f"session_{participant_id}"
            
            for feature in angle_features:
                for angle_code in self.angle_mappings.keys():
                    if angle_code in feature:
                        stat_type = 'mean'
                        measurement_id = f"{angle_code}_{stat_type}"
                        value = row[feature]
                        
                        if pd.notna(value):
                            batch_data.append({
                                'session_id': session_id,
                                'measurement_id': measurement_id,
                                'value': float(value),
                                'stat_type': stat_type,
                                'angle_code': angle_code,
                                'measurement_type': 'angle'
                            })
                        break
        
        if batch_data:
            self._create_angle_feature_batch(session, batch_data)
    
    def _process_mean_distance_features(self, session, df):
        """Process distance features - mean only"""
        distance_features = [col for col in df.columns if col.strip().startswith('mean ') and 
                           len(col.strip().split()) == 2 and not col.strip().startswith('mean-')]
        
        # Exclude angle features that might match the pattern
        distance_features = [f for f in distance_features if not any(angle in f for angle in self.angle_mappings.keys())]
        
        logging.info(f"Processing {len(distance_features)} mean distance features...")
        
        batch_data = []
        for idx, row in df.iterrows():
            participant_id = row['participant_id']
            session_id = f"session_{participant_id}"
            
            for feature in distance_features:
                parts = feature.strip().split(' ')
                if len(parts) == 2:
                    stat_type, distance_code = parts
                    measurement_id = f"{distance_code}_{stat_type}"
                    value = row[feature]
                    
                    if pd.notna(value):
                        batch_data.append({
                            'session_id': session_id,
                            'measurement_id': measurement_id,
                            'value': float(value),
                            'stat_type': stat_type,
                            'distance_code': distance_code,
                            'measurement_type': 'distance'
                        })
        
        if batch_data:
            self._create_distance_feature_batch(session, batch_data)
    
    def _process_rom_features(self, session, df):
        """Process Range of Motion features"""
        rom_features = [col for col in df.columns if col.strip().startswith('Rom')]
        
        logging.info(f"Processing {len(rom_features)} ROM features...")
        
        batch_data = []
        for idx, row in df.iterrows():
            participant_id = row['participant_id']
            session_id = f"session_{participant_id}"
            
            for feature in rom_features:
                measurement_id = feature.strip()
                value = row[feature]
                
                if pd.notna(value):
                    batch_data.append({
                        'session_id': session_id,
                        'measurement_id': measurement_id,
                        'value': float(value),
                        'measurement_type': 'range_of_motion'
                    })
        
        if batch_data:
            session.run("""
                UNWIND $batch AS data
                MATCH (s:GaitSession {id: data.session_id})
                CREATE (f:GaitFeature {
                    measurement_id: data.measurement_id,
                    value: data.value,
                    type: data.measurement_type
                })
                CREATE (s)-[:HAS_FEATURE]->(f)
            """, batch=batch_data)
    
    def _process_gait_parameters(self, session, df):
        """Process gait parameters"""
        logging.info("Processing gait parameters...")
        
        # Check which gait parameters exist in the dataset
        existing_params = {}
        for code, name in self.gait_params_excel.items():
            if code in df.columns:
                existing_params[code] = name
            elif name in df.columns:
                existing_params[name] = name
        
        if 'Velocity' in df.columns:
            existing_params['Velocity'] = 'Gait Velocity'
        
        logging.info(f"Found gait parameters: {list(existing_params.keys())}")
        
        # Process each participant's gait parameters
        for idx, row in df.iterrows():
            participant_id = row['participant_id']
            session_id = f"session_{participant_id}"
            
            for param_col, param_name in existing_params.items():
                if param_col in row and pd.notna(row[param_col]):
                    value = float(row[param_col])
                    
                    # Find the code for this parameter
                    param_code = param_col
                    for code, name in self.gait_params_excel.items():
                        if name == param_name:
                            param_code = code
                            break
                    
                    session.run("""
                        MATCH (s:GaitSession {id: $session_id})
                        MATCH (gp:GaitParameter {code: $param_code})
                        CREATE (s)-[:HAS_GAIT_VALUE {value: $value}]->(gp)
                    """, session_id=session_id, param_code=param_code, value=value)
    
    def _create_feature_batch(self, session, batch_data):
        """Create a batch of coordinate features"""
        session.run("""
            UNWIND $batch AS data
            MATCH (s:GaitSession {id: data.session_id})
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
    
    def _create_angle_feature_batch(self, session, batch_data):
        """Create a batch of angle features"""
        session.run("""
            UNWIND $batch AS data
            MATCH (s:GaitSession {id: data.session_id})
            MATCH (at:AngleType {code: data.angle_code})
            MATCH (mt:MeasurementType {name: data.measurement_type})
            CREATE (f:GaitFeature {
                measurement_id: data.measurement_id,
                value: data.value,
                stat_type: data.stat_type
            })
            CREATE (s)-[:HAS_FEATURE]->(f)
            CREATE (f)-[:HAS_ANGLE_TYPE]->(at)
            CREATE (f)-[:HAS_MEASUREMENT]->(mt)
        """, batch=batch_data)
    
    def _create_distance_feature_batch(self, session, batch_data):
        """Create a batch of distance features"""
        session.run("""
            UNWIND $batch AS data
            MATCH (s:GaitSession {id: data.session_id})
            MATCH (mt:MeasurementType {name: data.measurement_type})
            CREATE (f:GaitFeature {
                measurement_id: data.measurement_id,
                value: data.value,
                stat_type: data.stat_type,
                distance_code: data.distance_code
            })
            CREATE (s)-[:HAS_FEATURE]->(f)
            CREATE (f)-[:HAS_MEASUREMENT]->(mt)
        """, batch=batch_data)
    
    def _normalize_body_part(self, body_part_str):
        """Normalize body part names"""
        # Mapping from Excel names to standard names
        mappings = {
            'midspain': 'SpineMid',
            'ankleleft': 'AnkleLeft',
            'ankleright': 'AnkleRight',
            'kneeleft': 'KneeLeft',
            'kneeright': 'KneeRight',
            'hipleft': 'HipLeft',
            'hipright': 'HipRight',
            'wristleft': 'WristLeft',
            'wristright': 'WristRight',
            'handleft': 'HandLeft',
            'handright': 'HandRight',
            'handtipleft': 'HandTipLeft',
            'handtiprighta': 'HandTipRight',  # Note the 'A' suffix in the data
            'head': 'Head',
            'neck': 'Neck',
            'shoulderleft': 'ShoulderLeft',
            'shoulderright': 'ShoulderRight',
            'elbowleft': 'ElbowLeft',
            'elbowright': 'ElbowRight',
            'spineshoulder': 'SpineShoulder',
            'spinebase': 'SpineBase',
            'footleft': 'FootLeft',
            'footright': 'FootRight',
            'thumbleft': 'ThumbLeft',
            'thumbright': 'ThumbRight'
        }
        
        normalized = body_part_str.lower()
        return mappings.get(normalized, body_part_str)  # Return original if not found
    
    def get_statistics(self):
        """Get graph statistics"""
        with self.driver.session() as session:
            stats = {}
            
            # Node counts
            node_types = ['Participant', 'GaitSession', 'GaitFeature', 'BodyPart', 
                         'GaitParameter', 'Classification', 'AngleType']
            
            for node_type in node_types:
                result = session.run(f"MATCH (n:{node_type}) RETURN count(n) as count")
                stats[node_type] = result.single()['count']
            
            # Relationship counts
            rel_types = ['HAS_SESSION', 'HAS_FEATURE', 'HAS_GAIT_VALUE', 'CLASSIFIED_AS']
            
            for rel_type in rel_types:
                result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                stats[rel_type] = result.single()['count']
            
            return stats
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            logging.info("Neo4j connection closed")
    
    def build_graph(self, filepath="Final dataset.csv", clear_existing=True):
        """Main method to build the complete graph with mean features only"""
        try:
            # Connect to Neo4j
            if not self.connect():
                return False
            
            # Clear existing data if requested
            if clear_existing:
                self.clear_database()
            
            # Create schema
            self.create_constraints_and_indexes()
            
            # Create static nodes
            self.create_static_nodes()
            
            # Create anatomical connections
            self.create_anatomical_connections()
            
            # Load and process data (mean features only)
            df = self.load_and_process_data(filepath)
            
            # Create participants and sessions
            self.create_participants_and_sessions(df)
            
            # Create features
            self.create_gait_features(df)
            
            # Get statistics
            stats = self.get_statistics()
            
            logging.info("Graph building completed successfully!")
            logging.info("Statistics:")
            for key, value in stats.items():
                logging.info(f"  {key}: {value}")
            
            logging.info("\n🎯 REDUNDANCY ELIMINATION SUMMARY:")
            logging.info("  ✅ Used only MEAN features (eliminated variance & std)")
            logging.info("  ✅ Reduced feature space by ~67%")
            logging.info("  ✅ Eliminated mathematical redundancy")
            logging.info("  ✅ Should achieve realistic classification performance")
            
            return True
            
        except Exception as e:
            logging.error(f"Error building graph: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False
            
        finally:
            self.close()


if __name__ == "__main__":
    builder = NeuroGaitGraphBuilderMeanOnly()
    builder.build_graph("Final dataset.csv")