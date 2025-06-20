#!/usr/bin/env python3
"""
NeuroGait ASD Knowledge Graph Builder
=====================================

This script builds a comprehensive knowledge graph from gait analysis data
for autism spectrum disorder (ASD) research using Neo4j.

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
    Knowledge Graph Builder for NeuroGait ASD Dataset
    
    Creates a comprehensive Neo4j knowledge graph capturing:
    - Body part hierarchies and relationships
    - Biomechanical measurements and statistics
    - Gait parameters and temporal features
    - Subject classifications and patterns
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
            self.data = pd.read_excel(file_path)
            logger.info(f"Loaded dataset with {len(self.data)} samples and {len(self.data.columns)} features")
            
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
            if col == 'class':
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
        """Create the knowledge graph schema"""
        logger.info("Creating knowledge graph schema...")
        
        with self.driver.session() as session:
            # Create constraints and indexes
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (bp:BodyPart) REQUIRE bp.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (mt:MeasurementType) REQUIRE mt.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (cd:CoordinateDimension) REQUIRE cd.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Subject) REQUIRE s.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Classification) REQUIRE c.label IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (gp:GaitParameter) REQUIRE gp.name IS UNIQUE",
                "CREATE INDEX IF NOT EXISTS FOR (bp:BodyPart) ON (bp.type)",
                "CREATE INDEX IF NOT EXISTS FOR (s:Subject) ON (s.classification)"
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
            for classification in ['A', 'T']:
                label = 'ASD' if classification == 'A' else 'Typical'
                session.run(
                    "MERGE (c:Classification {label: $label, code: $code})",
                    label=label, code=classification
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
    
    def populate_subject_data(self, sample_size: int = None):
        """Populate the graph with subject data"""
        logger.info("Populating subject data...")
        
        # Use sample for testing if specified
        if sample_size:
            data_subset = self.data.sample(n=min(sample_size, len(self.data)))
        else:
            data_subset = self.data
        
        with self.driver.session() as session:
            for idx, row in data_subset.iterrows():
                # Create subject node
                classification = row['class']
                session.run("""
                    MERGE (s:Subject {
                        id: $subject_id,
                        classification_code: $classification
                    })
                    WITH s
                    MATCH (c:Classification {code: $classification})
                    MERGE (s)-[:CLASSIFIED_AS]->(c)
                """,
                subject_id=f"subject_{idx}",
                classification=classification
                )
                
                # Add body measurements
                self._add_subject_measurements(session, idx, row)
                
                # Add gait parameters
                self._add_subject_gait_parameters(session, idx, row)
        
        logger.info(f"Subject data populated for {len(data_subset)} subjects")
    
    def _add_subject_measurements(self, session, subject_id: int, row: pd.Series):
        """Add measurement values for a subject"""
        for feature in self.feature_schema['body_measurements']:
            parts = self._parse_body_measurement_feature(feature)
            if parts and feature in row.index:
                measurement_type, dimension, body_part = parts
                value = row[feature]
                
                if pd.notna(value):
                    session.run("""
                        MATCH (s:Subject {id: $subject_id})
                        MATCH (bp:BodyPart {name: $body_part})
                        MATCH (mt:MeasurementType {name: $measurement_type})
                        MATCH (cd:CoordinateDimension {name: $dimension})
                        MERGE (s)-[rel:HAS_VALUE]->(bp)
                        SET rel.measurement_type = $measurement_type,
                            rel.dimension = $dimension,
                            rel.value = $value
                    """,
                    subject_id=f"subject_{subject_id}",
                    body_part=body_part,
                    measurement_type=measurement_type,
                    dimension=dimension,
                    value=float(value)
                    )
    
    def _add_subject_gait_parameters(self, session, subject_id: int, row: pd.Series):
        """Add gait parameter values for a subject"""
        gait_features = ['MaxStLe', 'MaxStWi', 'StrLe', 'GaCT', 'StaT', 'SwiT', 'Velocity']
        
        for feature in gait_features:
            if feature in row.index and pd.notna(row[feature]):
                session.run("""
                    MATCH (s:Subject {id: $subject_id})
                    MATCH (gp:GaitParameter {code: $feature})
                    MERGE (s)-[rel:HAS_GAIT_VALUE]->(gp)
                    SET rel.value = $value
                """,
                subject_id=f"subject_{subject_id}",
                feature=feature,
                value=float(row[feature])
                )
    
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
            # Calculate average measurements by classification
            result = session.run("""
                MATCH (s:Subject)-[rel:HAS_VALUE]->(bp:BodyPart)
                MATCH (s)-[:CLASSIFIED_AS]->(c:Classification)
                RETURN c.label as classification,
                       bp.name as body_part,
                       rel.measurement_type as measurement_type,
                       rel.dimension as dimension,
                       avg(rel.value) as avg_value,
                       count(rel.value) as count
                ORDER BY classification, body_part, measurement_type, dimension
            """)
            
            patterns = []
            for record in result:
                patterns.append({
                    'classification': record['classification'],
                    'body_part': record['body_part'],
                    'measurement_type': record['measurement_type'],
                    'dimension': record['dimension'],
                    'avg_value': record['avg_value'],
                    'count': record['count']
                })
            
            return patterns
    
    def export_network_visualization(self, output_file: str = 'neurogait_network.png'):
        """Export a network visualization of the knowledge graph"""
        logger.info("Creating network visualization...")
        
        with self.driver.session() as session:
            # Get nodes and relationships for visualization
            result = session.run("""
                MATCH (n)
                OPTIONAL MATCH (n)-[r]->(m)
                RETURN n, r, m
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
                'BodyPart': '#FF6B6B',
                'BodyRegion': '#4ECDC4',
                'MeasurementType': '#45B7D1',
                'CoordinateDimension': '#FFA07A',
                'Classification': '#98D8C8',
                'GaitParameter': '#F7DC6F',
                'Subject': '#BB8FCE'
            }
            
            # Set node colors
            colors = [node_colors.get(G.nodes[node].get('type', 'unknown'), '#CCCCCC') for node in G.nodes()]
            
            # Use spring layout for better visualization
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Draw the graph
            nx.draw(G, pos, 
                   node_color=colors,
                   node_size=300,
                   font_size=8,
                   font_weight='bold',
                   with_labels=True,
                   edge_color='gray',
                   alpha=0.7)
            
            plt.title("NeuroGait ASD Knowledge Graph Structure", fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Network visualization saved to {output_file}")
    
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
            
            # Classification distribution
            result = session.run("""
                MATCH (s:Subject)-[:CLASSIFIED_AS]->(c:Classification)
                RETURN c.label as classification, count(s) as count
            """)
            stats['classifications'] = {record['classification']: record['count'] for record in result}
            
            return stats
    
    def build_complete_graph(self, excel_file: str, sample_size: int = None):
        """Build the complete knowledge graph from the Excel file"""
        logger.info("Starting complete knowledge graph build...")
        
        # Connect to database
        if not self.connect_to_neo4j():
            return False
        
        try:
            # Load data
            if not self.load_data(excel_file):
                return False
            
            # Build graph
            self.clear_database()
            self.create_schema()
            self.create_body_part_hierarchy()
            self.create_measurement_relationships()
            self.create_gait_parameters()
            self.create_anatomical_connections()
            self.populate_subject_data(sample_size)
            
            # Get statistics
            stats = self.get_statistics()
            logger.info("Knowledge Graph Statistics:")
            for category, items in stats.items():
                logger.info(f"  {category}:")
                for item, count in items.items():
                    logger.info(f"    {item}: {count}")
            
            logger.info("Knowledge graph build completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            return False
        finally:
            self.close_connection()


def main():
    """Main function to build the knowledge graph"""
    # Configuration - Update this path to match your file location
    EXCEL_FILE = "Final dataset.xlsx"  # Your data file 
    SAMPLE_SIZE = None  # Use None for full dataset, or set a number for testing (e.g., 100)
    
    print(f"📁 Looking for data file: {EXCEL_FILE}")
    
    # Check if file exists
    if not os.path.exists(EXCEL_FILE):
        print(f"❌ Error: Could not find Excel file '{EXCEL_FILE}'")
        print(f"📍 Current directory: {os.getcwd()}")
        print(f"💡 Make sure 'Final dataset.xlsx' is in the same directory as this script")
        return False
    
    print(f"✅ Found data file: {EXCEL_FILE}")
    
    # Create knowledge graph builder
    kg_builder = NeuroGaitKnowledgeGraph()
    
    # Build the complete knowledge graph
    success = kg_builder.build_complete_graph(EXCEL_FILE, sample_size=SAMPLE_SIZE)
    
    if success:
        print("\n" + "="*60)
        print("✅ NEUROGAIT KNOWLEDGE GRAPH BUILD COMPLETED!")
        print("="*60)
        print("\nNext steps:")
        print("1. Access your Neo4j browser at: http://localhost:7474")
        print("2. Use the browser to explore the graph visually")
        print("3. Run Cypher queries to analyze the data")
        print("\nExample queries:")
        print("- MATCH (n) RETURN count(n)  // Count all nodes")
        print("- MATCH (s:Subject)-[:CLASSIFIED_AS]->(c:Classification) RETURN c.label, count(s)")
        print("- MATCH (bp:BodyPart)-[:BELONGS_TO]->(br:BodyRegion) RETURN br.name, count(bp)")
        print("="*60)
    else:
        print("\n❌ Knowledge graph build failed. Check the logs for details.")


if __name__ == "__main__":
    main()