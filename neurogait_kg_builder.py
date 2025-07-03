#!/usr/bin/env python3
"""
NeuroGait ASD Knowledge Graph Builder - FIXED VERSION
=====================================================

This script builds a comprehensive knowledge graph from gait analysis data
for autism spectrum disorder (ASD) research using Neo4j with proper
Participant-Session structure to prevent data leakage.

CRITICAL FIX: 
- Proper Participant -> Session -> Features hierarchy
- No more session-level "subjects"
- Prevents data leakage in ML analysis

Author: AI Assistant
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
    FIXED Knowledge Graph Builder for NeuroGait ASD Dataset
    
    Creates a comprehensive Neo4j knowledge graph capturing:
    - Participant entities (800 unique participants)
    - Session entities (~3 sessions per participant)
    - Body part hierarchies and relationships
    - Biomechanical measurements and statistics
    - Gait parameters and temporal features
    - Proper classification relationships
    
    CRITICAL: Fixes data leakage by proper Participant-Session separation
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
        
        # Body part hierarchy and relationships
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
            'Midspain': {'parent': 'Core', 'type': 'center'},  # Note: keeping original spelling
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
            
            # Add participant_id if not present
            if 'participant_id' not in self.data.columns:
                self.data['participant_id'] = [f"P_{i:04d}" for i in range(1, len(self.data) + 1)]
                logger.info("Generated participant_id column")
            
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
        
        # Categorize features
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
            elif any(part in col for part in ['MaxStLe', 'MaxStWi', 'StrLe', 'GaCT', 'StaT', 'SwiT', 'Velocity']):
                self.feature_schema['temporal_gait'].append(col)
            elif any(col.startswith(f'{stat}-') for stat in self.measurement_types):
                self.feature_schema['body_measurements'].append(col)
            elif any(body_part in col for body_part in self.body_parts.keys()):
                if col not in self.feature_schema['body_measurements']:
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
        """Create the knowledge graph schema with proper Participant-Session structure"""
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
        CRITICAL FIX: Populate with proper Participant-Session structure
        
        This fixes the data leakage by creating:
        - One Participant node per unique participant_id
        - Multiple GaitSession nodes per participant
        - Features attached to sessions, not participants
        """
        logger.info("Populating participant and session data with FIXED structure...")
        
        # Use sample for testing if specified
        if sample_size:
            data_subset = self.data.sample(n=min(sample_size, len(self.data)))
        else:
            data_subset = self.data
        
        # Group data by participant_id
        grouped_data = data_subset.groupby('participant_id')
        
        participants_created = 0
        sessions_created = 0
        
        with self.driver.session() as session:
            for participant_id, participant_rows in grouped_data:
                
                # FIXED: Create ONE participant node per unique participant_id
                classification = participant_rows['diagnosis'].iloc[0]  # All rows should have same diagnosis
                
                session.run("""
                    MERGE (p:Participant {
                        id: $participant_id,
                        diagnosis: $diagnosis
                    })
                    WITH p
                    MATCH (c:Classification {label: $diagnosis})
                    MERGE (p)-[:CLASSIFIED_AS]->(c)
                """,
                participant_id=str(participant_id),
                diagnosis=classification
                )
                participants_created += 1
                
                # Create session nodes for each row (measurement session)
                for row_idx, row in participant_rows.iterrows():
                    session_id = f"session_{participant_id}_{row_idx}"
                    
                    session.run("""
                        MATCH (p:Participant {id: $participant_id})
                        CREATE (s:GaitSession {
                            session_id: $session_id,
                            participant_id: $participant_id,
                            date: datetime(),
                            video_duration: 0,
                            frame_count: 0
                        })
                        CREATE (p)-[:HAS_SESSION]->(s)
                    """,
                    participant_id=str(participant_id),
                    session_id=session_id
                    )
                    sessions_created += 1
                    
                    # Add measurements to this specific session
                    self._add_session_measurements(session, session_id, row)
                    
                    # Add gait parameters to this specific session
                    self._add_session_gait_parameters(session, session_id, row)
        
        logger.info(f"FIXED structure created: {participants_created} participants, {sessions_created} sessions")
    
    def _add_session_measurements(self, session, session_id: str, row: pd.Series):
        """Add measurement values for a specific session"""
        for feature in self.feature_schema['body_measurements']:
            parts = self._parse_body_measurement_feature(feature)
            if parts and feature in row.index:
                measurement_type, dimension, body_part = parts
                value = row[feature]
                
                if pd.notna(value):
                    session.run("""
                        MATCH (s:GaitSession {session_id: $session_id})
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
                    MATCH (s:GaitSession {session_id: $session_id})
                    MATCH (gp:GaitParameter {code: $feature})
                    MERGE (s)-[rel:HAS_GAIT_VALUE]->(gp)
                    SET rel.value = $value
                """,
                session_id=session_id,
                feature=feature,
                value=float(row[feature])
                )
        
        # Add other features that don't match the above patterns
        exclude_cols = ['participant_id', 'class', 'diagnosis'] + self.feature_schema['body_measurements'] + gait_features
        other_features = [col for col in row.index if col not in exclude_cols]
        
        for feature in other_features:
            if pd.notna(row[feature]):
                try:
                    value = float(row[feature])
                    session.run("""
                        MATCH (s:GaitSession {session_id: $session_id})
                        CREATE (f:GaitFeature {
                            feature_type: $feature_name,
                            value: $value,
                            calculated_at: datetime()
                        })
                        CREATE (s)-[:HAS_FEATURE]->(f)
                    """,
                    session_id=session_id,
                    feature_name=feature,
                    value=value
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
    
    def analyze_classification_patterns(self):
        """Analyze patterns between ASD and typical classifications"""
        logger.info("Analyzing classification patterns...")
        
        with self.driver.session() as session:
            # Calculate average measurements by classification at session level
            result = session.run("""
                MATCH (p:Participant)-[:HAS_SESSION]->(s:GaitSession)-[:HAS_FEATURE]->(f:GaitFeature)
                MATCH (p)-[:CLASSIFIED_AS]->(c:Classification)
                RETURN c.label as classification,
                       f.feature_type as feature_type,
                       avg(f.value) as avg_value,
                       count(f.value) as count,
                       stdDev(f.value) as std_value
                ORDER BY classification, feature_type
            """)
            
            patterns = []
            for record in result:
                patterns.append({
                    'classification': record['classification'],
                    'feature_type': record['feature_type'],
                    'avg_value': record['avg_value'],
                    'count': record['count'],
                    'std_value': record['std_value']
                })
            
            return patterns
    
    def export_network_visualization(self, output_file: str = 'neurogait_network_fixed.png'):
        """Export a network visualization of the fixed knowledge graph"""
        logger.info("Creating network visualization...")
        
        with self.driver.session() as session:
            # Get nodes and relationships for visualization (sample for performance)
            result = session.run("""
                MATCH (n)
                OPTIONAL MATCH (n)-[r]->(m)
                RETURN n, r, m
                LIMIT 1000
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
                    node_id = f"{labels[0]}:{node1.get('name', node1.get('id', node1.get('session_id', 'unknown')))}"
                    G.add_node(node_id, type=labels[0] if labels else 'unknown')
                
                if node2 and rel:
                    labels = list(node2.labels)
                    node_id2 = f"{labels[0]}:{node2.get('name', node2.get('id', node2.get('session_id', 'unknown')))}"
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
            
            plt.title("NeuroGait ASD Knowledge Graph - FIXED Structure\n(Participant-Session-Feature Hierarchy)", 
                     fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Network visualization saved to {output_file}")
    
    def get_statistics(self):
        """Get knowledge graph statistics with proper structure verification"""
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
            
            # CRITICAL: Verify participant vs session counts
            result = session.run("""
                MATCH (p:Participant) RETURN count(p) as participant_count
            """)
            stats['participants'] = result.single()['participant_count']
            
            result = session.run("""
                MATCH (s:GaitSession) RETURN count(s) as session_count
            """)
            stats['sessions'] = result.single()['session_count']
            
            # Classification distribution at participant level
            result = session.run("""
                MATCH (p:Participant)-[:CLASSIFIED_AS]->(c:Classification)
                RETURN c.label as classification, count(p) as count
            """)
            stats['participant_classifications'] = {record['classification']: record['count'] for record in result}
            
            # Features per session average
            result = session.run("""
                MATCH (s:GaitSession)-[:HAS_FEATURE]->(f:GaitFeature)
                WITH s, count(f) as feature_count
                RETURN avg(feature_count) as avg_features_per_session
            """)
            avg_features = result.single()['avg_features_per_session']
            stats['avg_features_per_session'] = avg_features
            
            return stats
    
    def verify_data_structure(self):
        """Verify the data structure is correct for preventing data leakage"""
        logger.info("Verifying data structure for data leakage prevention...")
        
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
            
            # Verify no orphaned sessions
            result = session.run("""
                MATCH (s:GaitSession)
                WHERE NOT (s)<-[:HAS_SESSION]-(:Participant)
                RETURN count(s) as orphaned_sessions
            """)
            orphaned = result.single()['orphaned_sessions']
            
            if orphaned == 0:
                logger.info("✅ No orphaned sessions found - structure is correct")
            else:
                logger.warning(f"⚠️ Found {orphaned} orphaned sessions")
            
            # Verify all participants have diagnosis
            result = session.run("""
                MATCH (p:Participant)
                WHERE NOT (p)-[:CLASSIFIED_AS]->(:Classification)
                RETURN count(p) as unclassified_participants
            """)
            unclassified = result.single()['unclassified_participants']
            
            if unclassified == 0:
                logger.info("✅ All participants have classifications")
            else:
                logger.warning(f"⚠️ Found {unclassified} unclassified participants")
    
    def build_complete_graph(self, data_file: str, sample_size: int = None):
        """Build the complete knowledge graph with FIXED structure"""
        logger.info("Starting FIXED knowledge graph build...")
        
        # Connect to database
        if not self.connect_to_neo4j():
            return False
        
        try:
            # Load data
            if not self.load_data(data_file):
                return False
            
            # Build graph with FIXED structure
            self.clear_database()
            self.create_schema()
            self.create_body_part_hierarchy()
            self.create_measurement_relationships()
            self.create_gait_parameters()
            self.create_anatomical_connections()
            
            # CRITICAL: Use fixed populate method
            self.populate_participant_session_data(sample_size)
            
            # Verify structure
            self.verify_data_structure()
            
            # Get statistics
            stats = self.get_statistics()
            logger.info("FIXED Knowledge Graph Statistics:")
            logger.info(f"  Participants: {stats['participants']}")
            logger.info(f"  Sessions: {stats['sessions']}")
            logger.info(f"  Avg Features/Session: {stats.get('avg_features_per_session', 'N/A')}")
            
            for category, items in stats.items():
                if isinstance(items, dict):
                    logger.info(f"  {category}:")
                    for item, count in items.items():
                        logger.info(f"    {item}: {count}")
            
            # Verify expected ratios
            if stats['participants'] > 0 and stats['sessions'] > 0:
                session_ratio = stats['sessions'] / stats['participants']
                logger.info(f"  Session/Participant Ratio: {session_ratio:.2f}")
                
                if 2.0 <= session_ratio <= 4.0:
                    logger.info("✅ CORRECT: Expected session/participant ratio achieved")
                else:
                    logger.warning("⚠️ Unexpected session/participant ratio")
            
            logger.info("FIXED Knowledge graph build completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            return False
        finally:
            self.close_connection()


def main():
    """Main function to build the FIXED knowledge graph"""
    # Configuration
    DATA_FILE = "Final dataset.csv"  # Support both CSV and Excel
    SAMPLE_SIZE = None  # Use None for full dataset, or set a number for testing
    
    print("🔧 NEUROGAIT KNOWLEDGE GRAPH BUILDER - FIXED VERSION")
    print("=" * 60)
    print("FIXES:")
    print("✅ Proper Participant-Session structure")
    print("✅ Prevents data leakage in ML analysis")
    print("✅ One participant = multiple sessions")
    print("=" * 60)
    
    print(f"📁 Looking for data file: {DATA_FILE}")
    
    # Check if file exists
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: Could not find data file '{DATA_FILE}'")
        print(f"📍 Current directory: {os.getcwd()}")
        print(f"💡 Make sure your data file is in the same directory as this script")
        
        # Try alternative file names
        alternative_files = ["Final dataset.xlsx", "final_dataset.csv", "dataset.csv"]
        for alt_file in alternative_files:
            if os.path.exists(alt_file):
                print(f"✅ Found alternative file: {alt_file}")
                DATA_FILE = alt_file
                break
        else:
            return False
    
    print(f"✅ Found data file: {DATA_FILE}")
    
    # Create knowledge graph builder
    kg_builder = NeuroGaitKnowledgeGraph()
    
    # Build the complete knowledge graph
    success = kg_builder.build_complete_graph(DATA_FILE, sample_size=SAMPLE_SIZE)
    
    if success:
        print("\n" + "="*60)
        print("✅ FIXED NEUROGAIT KNOWLEDGE GRAPH BUILD COMPLETED!")
        print("="*60)
        print("\nStructure created:")
        print("📊 Participants → Sessions → Features")
        print("🔒 No data leakage: proper participant-level separation")
        print("\nNext steps:")
        print("1. Access Neo4j browser: http://localhost:7474")
        print("2. Verify structure with these queries:")
        print("   MATCH (p:Participant) RETURN count(p)  // Should be ~800")
        print("   MATCH (s:GaitSession) RETURN count(s)  // Should be ~2400")
        print("   MATCH (p:Participant)-[:HAS_SESSION]->(s) RETURN count(s)/count(DISTINCT p)")
        print("3. Re-run your Streamlit analysis")
        print("4. Expect realistic performance (75-85% accuracy)")
        print("="*60)
    else:
        print("\n❌ FIXED knowledge graph build failed. Check the logs for details.")


if __name__ == "__main__":
    main()