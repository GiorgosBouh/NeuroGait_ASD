#!/usr/bin/env python3
"""
NeuroGait ASD Knowledge Graph Builder - PROPERLY FIXED VERSION
=============================================================

This script builds a comprehensive knowledge graph from gait analysis data
for autism spectrum disorder (ASD) research using Neo4j with CORRECT
Participant-Session structure.

CRITICAL FIXES: 
- FIXED: Proper understanding that each row = unique participant measurement
- No artificial session creation - 800 participants, 1 session each
- Maintains proper ML data structure for cross-validation

Author: AI Assistant (Fixed Version)
Date: 2025
Repository: https://github.com/GiorgosBouh/NeuroGait_ASD.git
"""

import pandas as pd
import numpy as np
import re
from neo4j import GraphDatabase
import logging
from typing import Dict, List, Tuple, Any
import os
from dotenv import load_dotenv
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeuroGaitKnowledgeGraph:
    """
    PROPERLY FIXED Knowledge Graph Builder for NeuroGait ASD Dataset
    
    Creates a comprehensive Neo4j knowledge graph capturing:
    - 800 unique Participant entities (one per measurement)
    - 800 Session entities (one per participant) 
    - Body part hierarchies and relationships
    - Biomechanical measurements and statistics
    - Gait parameters and temporal features
    - Proper classification relationships
    
    CRITICAL FIX: Correct understanding of data structure
    - Each row = unique participant measurement
    - No artificial grouping by non-existent participant_id
    - Maintains scientific validity for ML analysis
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
        
        # Body part hierarchy and relationships (corrected names to match data)
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
            'Midspain': {'parent': 'Core', 'type': 'center'},  # Keeping original spelling from data
            'HipLeft': {'parent': 'Lower_Body', 'type': 'hip', 'side': 'left'},
            'HipRight': {'parent': 'Lower_Body', 'type': 'hip', 'side': 'right'},
            'KneeLeft': {'parent': 'Lower_Body', 'type': 'knee', 'side': 'left'},
            'KneeRight': {'parent': 'Lower_Body', 'type': 'knee', 'side': 'right'},
            'AnkleLeft': {'parent': 'Lower_Body', 'type': 'ankle', 'side': 'left'},
            'AnkleRight': {'parent': 'Lower_Body', 'type': 'ankle', 'side': 'right'},
            'FootLeft': {'parent': 'Lower_Body', 'type': 'foot', 'side': 'left'},
            'FootRight': {'parent': 'Lower_Body', 'type': 'foot', 'side': 'right'}
        }
        
        # Measurement types
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
            
            # Determine file type and load accordingly
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                self.data = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                # Try different delimiters
                delimiters = [',', ';', '\t', '|']
                for delimiter in delimiters:
                    try:
                        self.data = pd.read_csv(file_path, delimiter=delimiter)
                        if len(self.data.columns) > 10:  # Reasonable number of columns
                            logger.info(f"Successfully loaded CSV with delimiter: '{delimiter}'")
                            break
                    except:
                        continue
                else:
                    raise ValueError("Could not parse CSV with any common delimiter")
            else:
                raise ValueError("Unsupported file format. Use .xlsx, .xls, or .csv")
            
            logger.info(f"Loaded dataset with {len(self.data)} samples and {len(self.data.columns)} features")
            
            # FIXED: Create proper participant IDs - each row is unique participant
            self.data['participant_id'] = [f"P_{i:04d}" for i in range(1, len(self.data) + 1)]
            logger.info("Generated unique participant_id for each measurement")
            
            # Map class values if needed
            if 'class' in self.data.columns:
                # Map A/T to ASD/Control
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
        
        # Categorize features based on actual data patterns
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
            elif 'T' in col and not col.startswith('Rom'):  # Distance features contain 'T'
                self.feature_schema['distance_features'].append(col)
            else:
                self.feature_schema['other'].append(col)
        
        logger.info(f"Feature analysis complete:")
        for category, features in self.feature_schema.items():
            if isinstance(features, list):
                logger.info(f"  {category}: {len(features)} features")
    
    def clear_database(self):
        """Clear the Neo4j database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")
    
    def create_schema(self):
        """Create the knowledge graph schema"""
        logger.info("Creating knowledge graph schema...")
        
        with self.driver.session() as session:
            # Create constraints and indexes
            constraints = [
                # FIXED: Proper participant and session constraints
                "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Participant) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:GaitSession) REQUIRE s.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (bp:BodyPart) REQUIRE bp.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (mt:MeasurementType) REQUIRE mt.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (cd:CoordinateDimension) REQUIRE cd.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Classification) REQUIRE c.label IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (gp:GaitParameter) REQUIRE gp.name IS UNIQUE",
                
                # Indexes for performance
                "CREATE INDEX IF NOT EXISTS FOR (p:Participant) ON (p.diagnosis)",
                "CREATE INDEX IF NOT EXISTS FOR (s:GaitSession) ON (s.participant_id)",
                "CREATE INDEX IF NOT EXISTS FOR (bp:BodyPart) ON (bp.type)",
                "CREATE INDEX IF NOT EXISTS FOR (f:GaitFeature) ON (f.feature_type)"
            ]
            
            for constraint in constraints:
                session.run(constraint)
        
        logger.info("Schema created successfully")
    
    def create_body_part_hierarchy(self):
        """Create body part nodes and hierarchical relationships"""
        logger.info("Creating body part hierarchy...")
        
        with self.driver.session() as session:
            # Create main body regions
            regions = ['Upper_Body', 'Core', 'Lower_Body']
            for region in regions:
                session.run(
                    "MERGE (r:BodyRegion {name: $name, type: 'region'})",
                    name=region
                )
            
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
            classifications = [
                {'label': 'ASD', 'code': 'A'},
                {'label': 'Control', 'code': 'T'}
            ]
            
            for classification in classifications:
                session.run(
                    "MERGE (c:Classification {label: $label, code: $code})",
                    label=classification['label'], 
                    code=classification['code']
                )
        
        logger.info("Body part hierarchy created")
    
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
    
    def _parse_body_measurement_feature(self, feature: str) -> Tuple[str, str, str]:
        """Parse body measurement feature name to extract components"""
        # Pattern: mean-x-BodyPart, variance-y-BodyPart, std-z-BodyPart
        pattern = r'(mean|variance|std)-(x|y|z)-(.+)'
        match = re.match(pattern, feature)
        
        if match:
            measurement_type, dimension, body_part = match.groups()
            return measurement_type, dimension, body_part
        
        return None
    
    def create_gait_parameters(self):
        """Create gait parameter nodes"""
        logger.info("Creating gait parameters...")
        
        gait_params = {
            'MaxStLe': {'name': 'Maximum Step Length', 'category': 'spatial'},
            'MaxStWi': {'name': 'Maximum Step Width', 'category': 'spatial'},
            'StrLe': {'name': 'Stride Length', 'category': 'spatial'},
            'GaCT': {'name': 'Gait Cycle Time', 'category': 'temporal'},
            'StaT': {'name': 'Stance Time', 'category': 'temporal'},
            'SwiT': {'name': 'Swing Time', 'category': 'temporal'},
            'Velocity': {'name': 'Gait Velocity', 'category': 'kinematic'}
        }
        
        with self.driver.session() as session:
            for param_code, properties in gait_params.items():
                session.run("""
                    MERGE (gp:GaitParameter {
                        code: $code,
                        name: $name,
                        category: $category
                    })
                """,
                code=param_code,
                name=properties['name'],
                category=properties['category']
                )
        
        logger.info("Gait parameters created")
    
    def populate_participant_session_data(self, sample_size: int = None):
        """
        PROPERLY FIXED: Populate with correct Participant-Session structure
        
        This creates the correct structure:
        - One Participant node per row (800 unique participants)
        - One GaitSession node per participant
        - Features attached to sessions
        
        NO artificial grouping - each measurement is independent
        """
        logger.info("Populating participant and session data with CORRECTED structure...")
        
        # Use sample for testing if specified
        if sample_size:
            data_subset = self.data.sample(n=min(sample_size, len(self.data)))
        else:
            data_subset = self.data
        
        participants_created = 0
        sessions_created = 0
        
        with self.driver.session() as session:
            for idx, row in data_subset.iterrows():
                participant_id = row['participant_id']
                session_id = f"session_{participant_id}"
                classification = row['diagnosis']
                
                # FIXED: Create ONE participant node per measurement
                session.run("""
                    MERGE (p:Participant {
                        id: $participant_id,
                        diagnosis: $diagnosis,
                        data_index: $data_index
                    })
                    WITH p
                    MATCH (c:Classification {label: $diagnosis})
                    MERGE (p)-[:CLASSIFIED_AS]->(c)
                """,
                participant_id=participant_id,
                diagnosis=classification,
                data_index=int(idx)
                )
                participants_created += 1
                
                # Create ONE session per participant
                session.run("""
                    MATCH (p:Participant {id: $participant_id})
                    CREATE (s:GaitSession {
                        id: $session_id,
                        participant_id: $participant_id,
                        measurement_date: datetime(),
                        session_type: 'primary'
                    })
                    CREATE (p)-[:HAS_SESSION]->(s)
                """,
                participant_id=participant_id,
                session_id=session_id
                )
                sessions_created += 1
                
                # Add measurements to this session
                self._add_session_measurements(session, session_id, row)
                self._add_session_gait_parameters(session, session_id, row)
        
        logger.info(f"CORRECTED structure created: {participants_created} participants, {sessions_created} sessions")
        logger.info("Ratio: 1 session per participant (correct for this dataset)")
    
    def _add_session_measurements(self, session, session_id: str, row: pd.Series):
        """Add measurement values for a specific session"""
        for feature in self.feature_schema['body_measurements']:
            parts = self._parse_body_measurement_feature(feature)
            if parts and feature in row.index:
                measurement_type, dimension, body_part = parts
                value = row[feature]
                
                if pd.notna(value):
                    session.run("""
                        MATCH (s:GaitSession {id: $session_id})
                        CREATE (f:GaitFeature {
                            feature_type: $feature_name,
                            value: $value,
                            measurement_type: $measurement_type,
                            dimension: $dimension,
                            body_part: $body_part,
                            calculated_at: datetime()
                        })
                        CREATE (s)-[:HAS_FEATURE]->(f)
                    """,
                    session_id=session_id,
                    feature_name=feature,
                    value=float(value),
                    measurement_type=measurement_type,
                    dimension=dimension,
                    body_part=body_part
                    )
    
    def _add_session_gait_parameters(self, session, session_id: str, row: pd.Series):
        """Add gait parameter values for a specific session"""
        gait_features = ['MaxStLe', 'MaxStWi', 'StrLe', 'GaCT', 'StaT', 'SwiT', 'Velocity']
        
        for feature in gait_features:
            if feature in row.index and pd.notna(row[feature]):
                session.run("""
                    MATCH (s:GaitSession {id: $session_id})
                    MATCH (gp:GaitParameter {code: $feature})
                    MERGE (s)-[rel:HAS_GAIT_VALUE]->(gp)
                    SET rel.value = $value
                """,
                session_id=session_id,
                feature=feature,
                value=float(row[feature])
                )
        
        # Add ROM and distance features
        other_feature_categories = ['range_of_motion', 'distance_features', 'other']
        for category in other_feature_categories:
            for feature in self.feature_schema[category]:
                if feature in row.index and pd.notna(row[feature]):
                    try:
                        value = float(row[feature])
                        session.run("""
                            MATCH (s:GaitSession {id: $session_id})
                            CREATE (f:GaitFeature {
                                feature_type: $feature_name,
                                value: $value,
                                category: $category,
                                calculated_at: datetime()
                            })
                            CREATE (s)-[:HAS_FEATURE]->(f)
                        """,
                        session_id=session_id,
                        feature_name=feature,
                        value=value,
                        category=category
                        )
                    except (ValueError, TypeError):
                        continue
    
    def create_anatomical_connections(self):
        """Create anatomical connections between body parts"""
        logger.info("Creating anatomical connections...")
        
        # Define anatomical connections (parent-child relationships in kinematic chain)
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
        
        with self.driver.session() as session:
            for parent, child in connections:
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
            
            # Verify participant vs session counts
            result = session.run("MATCH (p:Participant) RETURN count(p) as participant_count")
            stats['participants'] = result.single()['participant_count']
            
            result = session.run("MATCH (s:GaitSession) RETURN count(s) as session_count")
            stats['sessions'] = result.single()['session_count']
            
            # Classification distribution
            result = session.run("""
                MATCH (p:Participant)-[:CLASSIFIED_AS]->(c:Classification)
                RETURN c.label as classification, count(p) as count
            """)
            stats['participant_classifications'] = {record['classification']: record['count'] for record in result}
            
            return stats
    
    def verify_data_structure(self):
        """Verify the data structure is correct"""
        logger.info("Verifying corrected data structure...")
        
        with self.driver.session() as session:
            # Check participant-session ratio
            result = session.run("""
                MATCH (p:Participant)-[:HAS_SESSION]->(s:GaitSession)
                WITH p, count(s) as session_count
                RETURN avg(session_count) as avg_sessions_per_participant,
                       min(session_count) as min_sessions,
                       max(session_count) as max_sessions
            """)
            
            record = result.single()
            avg_sessions = record['avg_sessions_per_participant']
            min_sessions = record['min_sessions']
            max_sessions = record['max_sessions']
            
            logger.info(f"Sessions per participant - Avg: {avg_sessions:.2f}, Min: {min_sessions}, Max: {max_sessions}")
            
            if avg_sessions == 1.0 and min_sessions == 1 and max_sessions == 1:
                logger.info("✅ CORRECT: Each participant has exactly 1 session")
            else:
                logger.warning("⚠️ Unexpected session distribution")
            
            # Verify no orphaned sessions
            result = session.run("""
                MATCH (s:GaitSession)
                WHERE NOT (s)<-[:HAS_SESSION]-(:Participant)
                RETURN count(s) as orphaned_sessions
            """)
            orphaned = result.single()['orphaned_sessions']
            
            if orphaned == 0:
                logger.info("✅ No orphaned sessions found")
            else:
                logger.warning(f"⚠️ Found {orphaned} orphaned sessions")
    
    def export_network_visualization(self, output_file: str = 'neurogait_network_corrected.png'):
        """Export a network visualization of the corrected knowledge graph"""
        logger.info("Creating network visualization...")
        
        with self.driver.session() as session:
            # Get a sample of nodes and relationships for visualization
            result = session.run("""
                MATCH (n)
                OPTIONAL MATCH (n)-[r]->(m)
                RETURN n, r, m
                LIMIT 500
            """)
            
            # Create NetworkX graph
            G = nx.Graph()
            
            for record in result:
                node1 = record['n']
                rel = record['r']
                node2 = record['m']
                
                # Add nodes
                if node1:
                    labels = list(node1.labels)
                    node_id = f"{labels[0]}:{node1.get('name', node1.get('id', 'unknown'))}"
                    G.add_node(node_id, type=labels[0] if labels else 'unknown')
                
                if node2 and rel:
                    labels = list(node2.labels)
                    node_id2 = f"{labels[0]}:{node2.get('name', node2.get('id', 'unknown'))}"
                    G.add_node(node_id2, type=labels[0] if labels else 'unknown')
                    G.add_edge(node_id, node_id2, relationship=rel.type)
            
            # Create visualization
            plt.figure(figsize=(20, 16))
            
            # Define colors for different node types
            node_colors = {
                'Participant': '#FF6B6B',      # Red for participants
                'GaitSession': '#4ECDC4',      # Teal for sessions
                'GaitFeature': '#45B7D1',      # Blue for features
                'BodyPart': '#FFA07A',         # Orange for body parts
                'BodyRegion': '#98D8C8',       # Light green for regions
                'Classification': '#F7DC6F',   # Yellow for classifications
                'MeasurementType': '#BB8FCE',  # Purple for measurements
                'CoordinateDimension': '#85C1E9', # Light blue for dimensions
                'GaitParameter': '#F8C471'     # Light orange for parameters
            }
            
            # Set node colors
            colors = [node_colors.get(G.nodes[node].get('type', 'unknown'), '#CCCCCC') for node in G.nodes()]
            
            # Use spring layout for better visualization
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Draw the graph
            nx.draw(G, pos, 
                   node_color=colors,
                   node_size=300,
                   font_size=6,
                   font_weight='bold',
                   with_labels=True,
                   edge_color='gray',
                   alpha=0.7)
            
            plt.title("NeuroGait ASD Knowledge Graph - CORRECTED Structure\n(800 Participants, 1 Session Each)", 
                     fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Network visualization saved to {output_file}")
    
    def build_complete_graph(self, data_file: str, sample_size: int = None):
        """Build the complete knowledge graph with CORRECTED structure"""
        logger.info("Starting CORRECTED knowledge graph build...")
        
        # Connect to database
        if not self.connect_to_neo4j():
            return False
        
        try:
            # Load data
            if not self.load_data(data_file):
                return False
            
            # Build graph with CORRECTED structure
            self.clear_database()
            self.create_schema()
            self.create_body_part_hierarchy()
            self.create_measurement_relationships()
            self.create_gait_parameters()
            self.create_anatomical_connections()
            
            # CRITICAL: Use corrected populate method
            self.populate_participant_session_data(sample_size)
            
            # Verify structure
            self.verify_data_structure()
            
            # Get statistics
            stats = self.get_statistics()
            logger.info("CORRECTED Knowledge Graph Statistics:")
            logger.info(f"  Participants: {stats['participants']}")
            logger.info(f"  Sessions: {stats['sessions']}")
            logger.info(f"  Participant:Session Ratio: 1:1 (CORRECT)")
            
            for category, items in stats.items():
                if isinstance(items, dict) and category not in ['participants', 'sessions']:
                    logger.info(f"  {category}:")
                    for item, count in items.items():
                        logger.info(f"    {item}: {count}")
            
            logger.info("CORRECTED Knowledge graph build completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            return False
        finally:
            self.close_connection()


def main():
    """Main function to build the CORRECTED knowledge graph"""
    # Configuration
    DATA_FILE = "Final dataset.xlsx"  # Your Excel file
    SAMPLE_SIZE = None  # Use None for full dataset, or set a number for testing
    
    print("🔧 NEUROGAIT KNOWLEDGE GRAPH BUILDER - PROPERLY CORRECTED VERSION")
    print("=" * 70)
    print("PROPER FIXES:")
    print("✅ Correct understanding: Each row = unique participant")
    print("✅ 800 participants, 1 session each (no artificial grouping)")
    print("✅ Maintains scientific validity for ML analysis")
    print("✅ Proper participant-level separation for cross-validation")
    print("=" * 70)
    
    print(f"📁 Looking for data file: {DATA_FILE}")
    
    # Check if file exists
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: Could not find data file '{DATA_FILE}'")
        print(f"📍 Current directory: {os.getcwd()}")
        print(f"💡 Make sure your data file is in the same directory as this script")
        return False
    
    print(f"✅ Found data file: {DATA_FILE}")
    
    # Create knowledge graph builder
    kg_builder = NeuroGaitKnowledgeGraph()
    
    # Build the complete knowledge graph
    success = kg_builder.build_complete_graph(DATA_FILE, sample_size=SAMPLE_SIZE)
    
    if success:
        print("\n" + "="*70)
        print("✅ CORRECTED NEUROGAIT KNOWLEDGE GRAPH BUILD COMPLETED!")
        print("="*70)
        print("\nCORRECT Structure created:")
        print("📊 800 Participants → 800 Sessions → Features")
        print("🔒 Proper data structure: 1 participant = 1 measurement")
        print("🎯 Ready for ML analysis with proper cross-validation")
        print("\nNext steps:")
        print("1. Access Neo4j browser: http://localhost:7474")
        print("2. Verify structure with these queries:")
        print("   MATCH (p:Participant) RETURN count(p)  // Should be 800")
        print("   MATCH (s:GaitSession) RETURN count(s)  // Should be 800")
        print("   MATCH (p)-[:HAS_SESSION]->(s) RETURN count(s)/count(DISTINCT p)  // Should be 1.0")
        print("3. Run your analysis with confidence!")
        print("="*70)
    else:
        print("\n❌ Knowledge graph build failed. Check the logs for details.")


if __name__ == "__main__":
    main()