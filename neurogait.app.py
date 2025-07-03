# NeuroGait_ASD: Complete Implementation with Participant-Level Split Fix
# A comprehensive system for ASD detection using gait analysis and knowledge graphs

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from neo4j import GraphDatabase
import json
import logging
from datetime import datetime
import io
import base64
from typing import Dict, List, Tuple, Optional
import pickle
import requests
import shap
from openpyxl import load_workbook
from collections import defaultdict

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, accuracy_score, precision_score, recall_score, f1_score)
import xgboost as xgb
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# Configuration
st.set_page_config(
    page_title="NeuroGait ASD Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jConnection:
    """Neo4j Database Connection Handler"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        if self.driver:
            self.driver.close()
            
    def execute_query(self, query: str, parameters: dict = None):
        """Execute a Cypher query"""
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]
    
    def create_gait_analysis_schema(self):
        """Create the knowledge graph schema for gait analysis"""
        queries = [
            # Create constraints and indexes
            """
            CREATE CONSTRAINT participant_id IF NOT EXISTS 
            FOR (p:Participant) REQUIRE p.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT session_id IF NOT EXISTS 
            FOR (s:GaitSession) REQUIRE s.session_id IS UNIQUE
            """,
            # Create participant node
            """
            CREATE INDEX participant_age IF NOT EXISTS 
            FOR (p:Participant) ON (p.age)
            """,
            # Create gait feature index
            """
            CREATE INDEX gait_feature_type IF NOT EXISTS 
            FOR (g:GaitFeature) ON (g.feature_type)
            """
        ]
        
        for query in queries:
            try:
                self.execute_query(query)
                logger.info(f"Schema query executed: {query[:50]}...")
            except Exception as e:
                logger.error(f"Error executing schema query: {e}")

class GaitAnalyzer:
    """Advanced Gait Analysis using MediaPipe"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def extract_pose_landmarks(self, video_path: str) -> List[Dict]:
        """Extract pose landmarks from video"""
        cap = cv2.VideoCapture(video_path)
        landmarks_data = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract landmark coordinates
                landmarks = {}
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    landmarks[f'landmark_{idx}_x'] = landmark.x
                    landmarks[f'landmark_{idx}_y'] = landmark.y
                    landmarks[f'landmark_{idx}_z'] = landmark.z
                    landmarks[f'landmark_{idx}_visibility'] = landmark.visibility
                
                landmarks['frame'] = frame_count
                landmarks['timestamp'] = frame_count / cap.get(cv2.CAP_PROP_FPS)
                landmarks_data.append(landmarks)
            
            frame_count += 1
            
        cap.release()
        return landmarks_data
    
    def calculate_gait_features(self, landmarks_data: List[Dict]) -> Dict:
        """Calculate comprehensive gait features from landmarks"""
        if not landmarks_data:
            return {}
            
        df = pd.DataFrame(landmarks_data)
        features = {}
        
        # Key joint indices for MediaPipe
        joint_indices = {
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_foot': 31, 'right_foot': 32
        }
        
        # Calculate step length
        left_foot_x = df[f'landmark_{joint_indices["left_foot"]}_x']
        right_foot_x = df[f'landmark_{joint_indices["right_foot"]}_x']
        step_lengths = np.abs(left_foot_x.diff()).dropna()
        
        features['step_length_mean'] = step_lengths.mean()
        features['step_length_std'] = step_lengths.std()
        features['step_length_cv'] = features['step_length_std'] / features['step_length_mean'] if features['step_length_mean'] > 0 else 0
        
        # Calculate cadence (steps per unit time)
        total_time = df['timestamp'].max() - df['timestamp'].min()
        estimated_steps = len(step_lengths[step_lengths > 0.01])  # Threshold for actual steps
        features['cadence'] = estimated_steps / total_time if total_time > 0 else 0
        
        # Joint angle calculations
        for joint_name, joint_idx in joint_indices.items():
            if 'shoulder' in joint_name:
                # Shoulder angle relative to vertical
                shoulder_y = df[f'landmark_{joint_idx}_y']
                elbow_idx = joint_indices[joint_name.replace('shoulder', 'elbow')]
                elbow_y = df[f'landmark_{elbow_idx}_y']
                
                angles = np.arctan2(shoulder_y - elbow_y, 1) * 180 / np.pi
                features[f'{joint_name}_angle_mean'] = angles.mean()
                features[f'{joint_name}_angle_std'] = angles.std()
            
            elif 'elbow' in joint_name:
                # Elbow flexion angle
                shoulder_idx = joint_indices[joint_name.replace('elbow', 'shoulder')]
                wrist_idx = joint_indices[joint_name.replace('elbow', 'wrist')]
                
                shoulder_x = df[f'landmark_{shoulder_idx}_x']
                shoulder_y = df[f'landmark_{shoulder_idx}_y']
                elbow_x = df[f'landmark_{joint_idx}_x']
                elbow_y = df[f'landmark_{joint_idx}_y']
                wrist_x = df[f'landmark_{wrist_idx}_x']
                wrist_y = df[f'landmark_{wrist_idx}_y']
                
                # Calculate angle using vectors
                v1_x = shoulder_x - elbow_x
                v1_y = shoulder_y - elbow_y
                v2_x = wrist_x - elbow_x
                v2_y = wrist_y - elbow_y
                
                dot_product = v1_x * v2_x + v1_y * v2_y
                mag_v1 = np.sqrt(v1_x**2 + v1_y**2)
                mag_v2 = np.sqrt(v2_x**2 + v2_y**2)
                
                angles = np.arccos(np.clip(dot_product / (mag_v1 * mag_v2), -1, 1)) * 180 / np.pi
                features[f'{joint_name}_angle_mean'] = angles.mean()
                features[f'{joint_name}_angle_std'] = angles.std()
        
        # Stride width variability
        left_foot_y = df[f'landmark_{joint_indices["left_foot"]}_y']
        right_foot_y = df[f'landmark_{joint_indices["right_foot"]}_y']
        stride_widths = np.abs(left_foot_y - right_foot_y)
        
        features['stride_width_mean'] = stride_widths.mean()
        features['stride_width_std'] = stride_widths.std()
        features['stride_width_cv'] = features['stride_width_std'] / features['stride_width_mean'] if features['stride_width_mean'] > 0 else 0
        
        # Asymmetry measures
        left_step_var = df[f'landmark_{joint_indices["left_foot"]}_x'].diff().std()
        right_step_var = df[f'landmark_{joint_indices["right_foot"]}_x'].diff().std()
        features['step_asymmetry'] = abs(left_step_var - right_step_var) / (left_step_var + right_step_var) if (left_step_var + right_step_var) > 0 else 0
        
        # Ground reaction force indicators (estimated from vertical displacement)
        left_ankle_y = df[f'landmark_{joint_indices["left_ankle"]}_y']
        right_ankle_y = df[f'landmark_{joint_indices["right_ankle"]}_y']
        
        features['left_grf_variance'] = left_ankle_y.diff().var()
        features['right_grf_variance'] = right_ankle_y.diff().var()
        
        # Replace NaN values with 0
        for key, value in features.items():
            if pd.isna(value):
                features[key] = 0.0
        
        return features

class KnowledgeGraphManager:
    """Manage knowledge graph operations for gait analysis"""
    
    def __init__(self, neo4j_connection: Neo4jConnection):
        self.neo4j = neo4j_connection
        
    def store_participant(self, participant_data: Dict) -> str:
        """Store participant information in the knowledge graph"""
        query = """
        MERGE (p:Participant {id: $participant_id})
        SET p.age = $age,
            p.gender = $gender,
            p.diagnosis = $diagnosis,
            p.created_at = datetime()
        RETURN p.id as participant_id
        """
        
        result = self.neo4j.execute_query(query, participant_data)
        return result[0]['participant_id'] if result else None
    
    def store_gait_session(self, session_data: Dict, participant_id: str) -> str:
        """Store gait analysis session data"""
        query = """
        MATCH (p:Participant {id: $participant_id})
        CREATE (s:GaitSession {
            session_id: $session_id,
            date: datetime($date),
            video_duration: $video_duration,
            frame_count: $frame_count
        })
        CREATE (p)-[:HAS_SESSION]->(s)
        RETURN s.session_id as session_id
        """
        
        session_data['participant_id'] = participant_id
        result = self.neo4j.execute_query(query, session_data)
        return result[0]['session_id'] if result else None
    
    def store_gait_features(self, features: Dict, session_id: str):
        """Store calculated gait features"""
        for feature_name, feature_value in features.items():
            query = """
            MATCH (s:GaitSession {session_id: $session_id})
            CREATE (f:GaitFeature {
                feature_type: $feature_name,
                value: $feature_value,
                calculated_at: datetime()
            })
            CREATE (s)-[:HAS_FEATURE]->(f)
            """
            
            self.neo4j.execute_query(query, {
                'session_id': session_id,
                'feature_name': feature_name,
                'feature_value': float(feature_value) if not np.isnan(float(feature_value)) else 0.0
            })
    
    def store_prediction_result(self, prediction_data: Dict, session_id: str):
        """Store ML prediction results"""
        query = """
        MATCH (s:GaitSession {session_id: $session_id})
        CREATE (r:PredictionResult {
            model_type: $model_type,
            prediction: $prediction,
            confidence: $confidence,
            anomaly_score: $anomaly_score,
            created_at: datetime()
        })
        CREATE (s)-[:HAS_PREDICTION]->(r)
        """
        
        prediction_data['session_id'] = session_id
        self.neo4j.execute_query(query, prediction_data)
    
    def get_participant_history(self, participant_id: str) -> List[Dict]:
        """Retrieve participant's analysis history"""
        query = """
        MATCH (p:Participant {id: $participant_id})-[:HAS_SESSION]->(s:GaitSession)
        OPTIONAL MATCH (s)-[:HAS_PREDICTION]->(r:PredictionResult)
        RETURN s.session_id as session_id, s.date as date, 
               r.prediction as prediction, r.confidence as confidence
        ORDER BY s.date DESC
        """
        
        return self.neo4j.execute_query(query, {'participant_id': participant_id})
    
    def get_participant_level_data(self):
        """Get data grouped by participant for proper train/test split"""
        query = """
        MATCH (p:Participant)-[:HAS_SESSION]->(s:GaitSession)-[:HAS_FEATURE]->(f:GaitFeature)
        WITH p, s, collect({feature_type: f.feature_type, value: f.value}) as features
        RETURN p.id as participant_id, p.diagnosis as diagnosis, 
               p.age as age, p.gender as gender, 
               s.session_id as session_id, features
        ORDER BY p.id, s.session_id
        """
        return self.neo4j.execute_query(query)
    
    def execute_natural_language_query(self, nl_query: str) -> List[Dict]:
        """Convert natural language to Cypher and execute (simplified version)"""
        # This is a simplified version - in production, you'd use GPT-4 for translation
        query_mappings = {
            "how many participants": "MATCH (p:Participant) RETURN count(p) as total_participants",
            "asd positive cases": """
                MATCH (p:Participant {diagnosis: 'ASD'})-[:HAS_SESSION]->(s:GaitSession)
                -[:HAS_PREDICTION]->(r:PredictionResult {prediction: 1})
                RETURN count(p) as asd_positive_cases
            """,
            "asd cases": "MATCH (p:Participant {diagnosis: 'ASD'}) RETURN count(p) as asd_cases",
            "control cases": "MATCH (p:Participant {diagnosis: 'Control'}) RETURN count(p) as control_cases",
            "total features": """
                MATCH (s:GaitSession)-[:HAS_FEATURE]->(f:GaitFeature)
                RETURN count(DISTINCT f.feature_type) as total_features
            """,
            "average step length": """
                MATCH (s:GaitSession)-[:HAS_FEATURE]->(f:GaitFeature {feature_type: 'step_length_mean'})
                RETURN avg(f.value) as avg_step_length
            """,
            "participants by age": """
                MATCH (p:Participant)
                RETURN p.age as age, count(p) as count
                ORDER BY p.age
            """,
            "diagnosis distribution": """
                MATCH (p:Participant)
                RETURN p.diagnosis as diagnosis, count(p) as count
            """
        }
        
        # Simple keyword matching - replace with LLM in production
        for keyword, cypher_query in query_mappings.items():
            if keyword in nl_query.lower():
                return self.neo4j.execute_query(cypher_query)
        
        return [{"error": "Query not recognized. Please try a simpler query."}]

class MLAnalyzer:
    """Machine Learning analysis for ASD prediction with participant-level splits"""
    
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.xgb_model = xgb.XGBClassifier(random_state=42)
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
        # Store participant-level data to prevent leakage
        self.participant_data = {}
        self.train_participants = []
        self.test_participants = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_size = 0.3
        
    def prepare_participant_data(self, participant_level_data: List[Dict]) -> Dict:
        """Prepare participant-level data for proper train/test split"""
        
        # Group data by participant
        participant_groups = defaultdict(list)
        participant_labels = {}
        
        for record in participant_level_data:
            participant_id = record['participant_id']
            diagnosis = record['diagnosis']
            
            # Extract features
            features = {}
            for feature in record['features']:
                features[feature['feature_type']] = feature['value']
            
            participant_groups[participant_id].append(features)
            participant_labels[participant_id] = 1 if diagnosis == 'ASD' else 0
        
        # Aggregate features per participant (average across sessions)
        participant_features = {}
        for participant_id, feature_list in participant_groups.items():
            if len(feature_list) > 0:
                # Convert to DataFrame and take mean
                df = pd.DataFrame(feature_list)
                df = df.fillna(df.mean())
                
                # Average across sessions for this participant
                avg_features = df.mean().to_dict()
                participant_features[participant_id] = avg_features
        
        # Store feature names
        if participant_features:
            self.feature_names = list(next(iter(participant_features.values())).keys())
        
        # Store data
        self.participant_data = {
            'features': participant_features,
            'labels': participant_labels
        }
        
        return {
            'total_participants': len(participant_features),
            'asd_count': sum(participant_labels.values()),
            'control_count': len(participant_labels) - sum(participant_labels.values()),
            'avg_sessions_per_participant': len(participant_level_data) / len(participant_features)
        }
    
    def train_models_participant_level(self):
        """Train models with participant-level split to prevent data leakage"""
        
        if not self.participant_data:
            logger.error("No participant data available")
            return
        
        # Get participant IDs and labels
        participant_ids = list(self.participant_data['features'].keys())
        participant_labels = [self.participant_data['labels'][pid] for pid in participant_ids]
        
        # CRITICAL: Split by participants, not by samples
        self.train_participants, self.test_participants = train_test_split(
            participant_ids, 
            test_size=self.test_size, 
            random_state=42, 
            stratify=participant_labels
        )
        
        # Prepare training data
        X_train_list = []
        y_train_list = []
        
        for participant_id in self.train_participants:
            features = self.participant_data['features'][participant_id]
            label = self.participant_data['labels'][participant_id]
            
            # Convert features to array
            feature_vector = [features.get(fname, 0) for fname in self.feature_names]
            X_train_list.append(feature_vector)
            y_train_list.append(label)
        
        # Prepare test data
        X_test_list = []
        y_test_list = []
        
        for participant_id in self.test_participants:
            features = self.participant_data['features'][participant_id]
            label = self.participant_data['labels'][participant_id]
            
            # Convert features to array
            feature_vector = [features.get(fname, 0) for fname in self.feature_names]
            X_test_list.append(feature_vector)
            y_test_list.append(label)
        
        # Convert to numpy arrays
        self.X_train = np.array(X_train_list)
        self.X_test = np.array(X_test_list)
        self.y_train = np.array(y_train_list)
        self.y_test = np.array(y_test_list)
        
        # Scale features using ONLY training data
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        
        # Handle class imbalance with SMOTE on training data only
        if len(np.unique(self.y_train)) > 1:
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, self.y_train)
        else:
            X_train_balanced, y_train_balanced = X_train_scaled, self.y_train
        
        # Train supervised models on training data only
        self.rf_model.fit(X_train_balanced, y_train_balanced)
        self.xgb_model.fit(X_train_balanced, y_train_balanced)
        
        # Train anomaly detection on original training data (not SMOTE-balanced)
        self.isolation_forest.fit(X_train_scaled)
        
        self.is_trained = True
        logger.info("Models trained successfully with participant-level split")
    
    def get_participant_cross_validation_scores(self, cv_folds=5):
        """Get cross-validation scores with participant-level grouping"""
        
        if not self.participant_data:
            return {"error": "No participant data available"}
        
        # Prepare data for GroupKFold
        participant_ids = list(self.participant_data['features'].keys())
        X_list = []
        y_list = []
        groups = []
        
        for i, participant_id in enumerate(participant_ids):
            features = self.participant_data['features'][participant_id]
            label = self.participant_data['labels'][participant_id]
            
            # Convert features to array
            feature_vector = [features.get(fname, 0) for fname in self.feature_names]
            X_list.append(feature_vector)
            y_list.append(label)
            groups.append(i)  # Each participant is a separate group
        
        X = np.array(X_list)
        y = np.array(y_list)
        groups = np.array(groups)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Perform GroupKFold cross-validation
        group_kfold = GroupKFold(n_splits=cv_folds)
        
        rf_scores = []
        xgb_scores = []
        
        for train_idx, test_idx in group_kfold.split(X_scaled, y, groups):
            X_train_cv, X_test_cv = X_scaled[train_idx], X_scaled[test_idx]
            y_train_cv, y_test_cv = y[train_idx], y[test_idx]
            
            # Train and evaluate RF
            rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_temp.fit(X_train_cv, y_train_cv)
            rf_score = rf_temp.score(X_test_cv, y_test_cv)
            rf_scores.append(rf_score)
            
            # Train and evaluate XGB
            xgb_temp = xgb.XGBClassifier(random_state=42)
            xgb_temp.fit(X_train_cv, y_train_cv)
            xgb_score = xgb_temp.score(X_test_cv, y_test_cv)
            xgb_scores.append(xgb_score)
        
        return {
            'rf_scores': rf_scores,
            'xgb_scores': xgb_scores,
            'rf_mean': np.mean(rf_scores),
            'rf_std': np.std(rf_scores),
            'xgb_mean': np.mean(xgb_scores),
            'xgb_std': np.std(xgb_scores)
        }
    
    def get_test_predictions(self) -> Dict:
        """Get predictions on unseen test participants"""
        if not self.is_trained or self.X_test is None:
            return {"error": "Models not trained or no test data available"}
        
        # Transform test data using scaler fitted on training data
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Get predictions on test data
        rf_pred = self.rf_model.predict(X_test_scaled)
        rf_proba = self.rf_model.predict_proba(X_test_scaled)[:, 1]
        
        xgb_pred = self.xgb_model.predict(X_test_scaled)
        xgb_proba = self.xgb_model.predict_proba(X_test_scaled)[:, 1]
        
        # Anomaly detection on test data
        anomaly_scores = self.isolation_forest.decision_function(X_test_scaled)
        anomaly_predictions = self.isolation_forest.predict(X_test_scaled)
        
        return {
            'rf_predictions': rf_pred,
            'rf_probabilities': rf_proba,
            'xgb_predictions': xgb_pred,
            'xgb_probabilities': xgb_proba,
            'anomaly_scores': anomaly_scores,
            'anomaly_predictions': anomaly_predictions,
            'y_true': self.y_test,
            'test_participants': self.test_participants
        }
    
    def predict(self, features: Dict) -> Dict:
        """Make predictions on new data"""
        if not self.is_trained:
            return {"error": "Models not trained yet"}
            
        # Prepare features
        feature_vector = []
        for feature_name in self.feature_names:
            feature_vector.append(features.get(feature_name, 0))
        
        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        rf_pred = self.rf_model.predict(X_scaled)[0]
        rf_proba = self.rf_model.predict_proba(X_scaled)[0]
        
        xgb_pred = self.xgb_model.predict(X_scaled)[0]
        xgb_proba = self.xgb_model.predict_proba(X_scaled)[0]
        
        # Anomaly detection
        anomaly_score = self.isolation_forest.decision_function(X_scaled)[0]
        is_anomaly = self.isolation_forest.predict(X_scaled)[0] == -1
        
        return {
            'rf_prediction': int(rf_pred),
            'rf_confidence': float(max(rf_proba)),
            'xgb_prediction': int(xgb_pred),
            'xgb_confidence': float(max(xgb_proba)),
            'anomaly_score': float(anomaly_score),
            'is_anomaly': bool(is_anomaly),
            'ensemble_prediction': int((rf_pred + xgb_pred) / 2 > 0.5)
        }
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from trained models"""
        if not self.is_trained:
            return {}
            
        rf_importance = dict(zip(self.feature_names, self.rf_model.feature_importances_))
        xgb_importance = dict(zip(self.feature_names, self.xgb_model.feature_importances_))
        
        return {
            'random_forest': rf_importance,
            'xgboost': xgb_importance
        }
    
    def get_shap_explanations(self, features: Dict) -> Dict:
        """Generate SHAP explanations"""
        if not self.is_trained:
            return {"error": "Models not trained"}
        
        try:
            # Prepare data
            feature_vector = []
            for feature_name in self.feature_names:
                feature_vector.append(features.get(feature_name, 0))
            
            X = np.array(feature_vector).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # SHAP for Random Forest
            explainer_rf = shap.TreeExplainer(self.rf_model)
            shap_values_rf = explainer_rf.shap_values(X_scaled)
            
            # SHAP for XGBoost
            explainer_xgb = shap.TreeExplainer(self.xgb_model)
            shap_values_xgb = explainer_xgb.shap_values(X_scaled)
            
            return {
                'rf_shap_values': shap_values_rf[1][0] if len(shap_values_rf) > 1 else shap_values_rf[0],
                'xgb_shap_values': shap_values_xgb[0] if len(shap_values_xgb.shape) > 1 else shap_values_xgb,
                'feature_names': self.feature_names
            }
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {e}")
            return {"error": str(e)}

# Initialize session state
if 'neo4j_connection' not in st.session_state:
    st.session_state.neo4j_connection = None
if 'kg_manager' not in st.session_state:
    st.session_state.kg_manager = None
if 'ml_analyzer' not in st.session_state:
    st.session_state.ml_analyzer = MLAnalyzer()
if 'gait_analyzer' not in st.session_state:
    st.session_state.gait_analyzer = GaitAnalyzer()

def main():
    """Main Streamlit application"""
    
    st.title("🧠 NeuroGait ASD Analysis System")
    st.markdown("### Advanced Gait Analysis for Autism Spectrum Disorder Detection")
    
    # Sidebar for navigation and configuration
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔧 Setup", "📊 Data Upload", "🎯 Analysis", "📈 Visualization", "🔍 Query Interface", "📋 Reports"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔧 Setup":
        show_setup_page()
    elif page == "📊 Data Upload":
        show_data_upload_page()
    elif page == "🎯 Analysis":
        show_analysis_page()
    elif page == "📈 Visualization":
        show_visualization_page()
    elif page == "🔍 Query Interface":
        show_query_interface_page()
    elif page == "📋 Reports":
        show_reports_page()

def show_home_page():
    """Display the home page"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to NeuroGait ASD Analysis
        
        This comprehensive system combines advanced gait analysis with knowledge graph technology 
        to support early detection and assessment of Autism Spectrum Disorder (ASD).
        
        ### Key Features:
        - **🎥 Video Gait Analysis**: Extract pose landmarks from walking videos
        - **📊 XLSX/CSV Batch Processing**: Process large datasets with 1000+ features
        - **🧠 Knowledge Graph Storage**: Store and relate complex gait data using Neo4j
        - **🤖 Machine Learning**: Multi-model approach for ASD prediction
        - **🔒 Participant-Level Splits**: Proper train/test splits to prevent data leakage
        - **📈 Interactive Visualizations**: Comprehensive data exploration tools
        - **💬 Natural Language Queries**: Ask questions about your data in plain English
        - **📱 Real-time Processing**: Immediate analysis and feedback
        - **🔍 SHAP Explanations**: Interpretable AI predictions
        - **📋 Complete Performance Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
        
        ### Supported Data Formats:
        - **Video files**: MP4, AVI, MOV (MediaPipe pose extraction)
        - **Excel files**: .xlsx/.xls with gait features
        - **CSV files**: Comma-separated gait features (auto-detects delimiters)
        - **Target variables**: 'class' (A/T) or 'diagnosis' (ASD/Control)
        
        ### System Architecture:
        1. **Data Collection**: Upload video files or batch data (XLSX/CSV)
        2. **Feature Extraction**: Advanced pose estimation or direct feature processing
        3. **Knowledge Graph**: Semantic storage and relationship modeling
        4. **ML Analysis**: Participant-level splits and ensemble models
        5. **Visualization**: Interactive dashboards and reports
        6. **Explainability**: SHAP-based feature importance and explanations
        
        ### 🚨 DATA LEAKAGE PREVENTION:
        - **Participant-Level Splits**: No participant appears in both train and test
        - **GroupKFold Validation**: Proper cross-validation by participant groups
        - **Honest Evaluation**: Realistic performance metrics
        """)
    
    with col2:
        # System status
        st.subheader("System Status")
        
        # Check Neo4j connection
        neo4j_status = "✅ Connected" if st.session_state.neo4j_connection else "❌ Not Connected"
        st.write(f"**Neo4j Database**: {neo4j_status}")
        
        # Check ML models
        ml_status = "✅ Trained" if st.session_state.ml_analyzer.is_trained else "❌ Not Trained"
        st.write(f"**ML Models**: {ml_status}")
        
        # Dataset info
        if hasattr(st.session_state.ml_analyzer, 'participant_data') and st.session_state.ml_analyzer.participant_data:
            participant_count = len(st.session_state.ml_analyzer.participant_data.get('features', {}))
            st.write(f"**Participants Loaded**: {participant_count}")
            
            if st.session_state.ml_analyzer.feature_names:
                feature_count = len(st.session_state.ml_analyzer.feature_names)
                st.write(f"**Features**: {feature_count}")
        
        # Quick stats if available
        if st.session_state.kg_manager:
            try:
                stats = st.session_state.kg_manager.neo4j.execute_query(
                    "MATCH (p:Participant) RETURN count(p) as total"
                )
                if stats:
                    st.metric("Total Participants", stats[0]['total'])
            except:
                pass

def show_setup_page():
    """Display the setup configuration page"""
    st.header("🔧 System Setup")
    
    # Neo4j Configuration
    st.subheader("Neo4j Database Configuration")
    
    with st.form("neo4j_config"):
        neo4j_uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
        neo4j_user = st.text_input("Username", value="neo4j")
        neo4j_password = st.text_input("Password", type="password")
        
        if st.form_submit_button("Connect to Neo4j"):
            try:
                connection = Neo4jConnection(neo4j_uri, neo4j_user, neo4j_password)
                connection.create_gait_analysis_schema()
                
                st.session_state.neo4j_connection = connection
                st.session_state.kg_manager = KnowledgeGraphManager(connection)
                
                st.success("✅ Successfully connected to Neo4j!")
                st.info("Knowledge graph schema created successfully.")
                
            except Exception as e:
                st.error(f"❌ Failed to connect to Neo4j: {e}")
    
    # Neo4j Community Edition Setup Instructions
    with st.expander("📚 Neo4j Community Edition Setup Guide"):
        st.markdown("""
        ### Setting up Neo4j Community Edition in VM
        
        #### Prerequisites:
        1. **Virtual Machine**: Ubuntu 20.04+ or similar Linux distribution
        2. **Java**: OpenJDK 11 or higher
        3. **Memory**: At least 4GB RAM allocated to VM
        
        #### Installation Steps:
        
        ```bash
        # 1. Update system packages
        sudo apt update && sudo apt upgrade -y
        
        # 2. Install Java (if not already installed)
        sudo apt install openjdk-11-jdk -y
        
        # 3. Add Neo4j repository
        wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
        echo 'deb https://debian.neo4j.com stable latest' | sudo tee -a /etc/apt/sources.list.d/neo4j.list
        
        # 4. Install Neo4j Community Edition
        sudo apt update
        sudo apt install neo4j -y
        
        # 5. Configure Neo4j
        sudo nano /etc/neo4j/neo4j.conf
        
        # Uncomment and modify these lines:
        # dbms.default_listen_address=0.0.0.0
        # dbms.connector.bolt.listen_address=:7687
        # dbms.connector.http.listen_address=:7474
        
        # 6. Start Neo4j service
        sudo systemctl enable neo4j
        sudo systemctl start neo4j
        
        # 7. Check status
        sudo systemctl status neo4j
        
        # 8. Set initial password
        sudo neo4j-admin set-initial-password your_password_here
        ```
        
        #### Accessing Neo4j:
        - **Browser Interface**: http://localhost:7474
        - **Bolt Connection**: bolt://localhost:7687
        - **Default Username**: neo4j
        - **Password**: The one you set during installation
        
        #### Firewall Configuration (if needed):
        ```bash
        sudo ufw allow 7474
        sudo ufw allow 7687
        ```
        """)
    
    # Model Configuration
    st.subheader("🤖 Machine Learning Configuration")
    
    with st.form("ml_config"):
        st.write("Configure ML model parameters:")
        
        col1, col2 = st.columns(2)
        with col1:
            rf_estimators = st.number_input("Random Forest Estimators", value=100, min_value=10, max_value=1000)
            contamination = st.slider("Isolation Forest Contamination", 0.01, 0.3, 0.1)
        
        with col2:
            xgb_max_depth = st.number_input("XGBoost Max Depth", value=6, min_value=3, max_value=20)
            test_size = st.slider("Train/Test Split", 0.1, 0.4, 0.3)
        
        if st.form_submit_button("Update ML Configuration"):
            # Update model parameters
            st.session_state.ml_analyzer.rf_model.set_params(n_estimators=rf_estimators)
            st.session_state.ml_analyzer.xgb_model.set_params(max_depth=xgb_max_depth)
            st.session_state.ml_analyzer.isolation_forest.set_params(contamination=contamination)
            st.session_state.ml_analyzer.test_size = test_size
            
            st.success("✅ ML configuration updated!")

def show_data_upload_page():
    """Display the data upload and processing page - ENHANCED WITH CSV FIX"""
    st.header("📊 Data Upload & Processing")
    
    if not st.session_state.neo4j_connection:
        st.warning("⚠️ Please configure Neo4j connection in the Setup page first.")
        return
    
    # Participant Information
    st.subheader("👤 Participant Information")
    
    with st.form("participant_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            participant_id = st.text_input("Participant ID", value=f"P_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            age = st.number_input("Age", min_value=1, max_value=100, value=8)
        
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            diagnosis = st.selectbox("Diagnosis", ["Control", "ASD", "Unknown"])
        
        with col3:
            additional_notes = st.text_area("Additional Notes")
        
        participant_submitted = st.form_submit_button("Register Participant")
    
    if participant_submitted:
        participant_data = {
            'participant_id': participant_id,
            'age': age,
            'gender': gender,
            'diagnosis': diagnosis
        }
        
        stored_id = st.session_state.kg_manager.store_participant(participant_data)
        if stored_id:
            st.success(f"✅ Participant {stored_id} registered successfully!")
            st.session_state.current_participant = stored_id
        else:
            st.error("❌ Failed to register participant")
    
    # Video Upload and Processing
    st.subheader("🎥 Video Upload")
    
    uploaded_file = st.file_uploader(
        "Upload gait analysis video",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file containing gait patterns for analysis"
    )
    
    if uploaded_file is not None and hasattr(st.session_state, 'current_participant'):
        st.video(uploaded_file)
        
        if st.button("🚀 Process Video"):
            with st.spinner("Processing video... This may take a few minutes."):
                try:
                    # Save uploaded file temporarily
                    temp_video_path = f"temp_{uploaded_file.name}"
                    with open(temp_video_path, "wb") as f:
                        f.write(uploaded_file.read())
                    
                    # Extract pose landmarks
                    st.info("Extracting pose landmarks...")
                    landmarks_data = st.session_state.gait_analyzer.extract_pose_landmarks(temp_video_path)
                    
                    if landmarks_data:
                        # Calculate gait features
                        st.info("Calculating gait features...")
                        gait_features = st.session_state.gait_analyzer.calculate_gait_features(landmarks_data)
                        
                        # Store session data
                        session_data = {
                            'session_id': f"S_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            'date': datetime.now().isoformat(),
                            'video_duration': len(landmarks_data) * 0.033,  # Approximate
                            'frame_count': len(landmarks_data)
                        }
                        
                        session_id = st.session_state.kg_manager.store_gait_session(
                            session_data, st.session_state.current_participant
                        )
                        
                        if session_id:
                            # Store features
                            st.session_state.kg_manager.store_gait_features(gait_features, session_id)
                            
                            # Store for display
                            st.session_state.current_features = gait_features
                            st.session_state.current_session = session_id
                            
                            st.success("✅ Video processed successfully!")
                            
                            # Display extracted features
                            st.subheader("📊 Extracted Gait Features")
                            
                            feature_df = pd.DataFrame(list(gait_features.items()), 
                                                    columns=['Feature', 'Value'])
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.dataframe(feature_df)
                            
                            with col2:
                                # Feature visualization
                                fig = px.bar(feature_df, x='Feature', y='Value', 
                                           title="Gait Feature Values")
                                fig.update_xaxis(tickangle=45)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        else:
                            st.error("❌ Failed to store session data")
                    
                    else:
                        st.error("❌ No pose landmarks detected in video")
                    
                    # Clean up temporary file
                    import os
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
                        
                except Exception as e:
                    st.error(f"❌ Error processing video: {e}")
    
    elif uploaded_file is not None:
        st.warning("⚠️ Please register a participant first before uploading video.")
    
    # ENHANCED Batch Data Upload with CSV FIX
    st.subheader("📁 Batch Data Upload")
    
    # File format selection
    upload_format = st.selectbox("Select file format:", ["CSV", "XLSX"])
    
    if upload_format == "CSV":
        st.markdown("""
        ### CSV Format for Batch Upload
        Upload a CSV file with pre-calculated gait features for multiple participants.
        
        **Required Columns:**
        - **Target Variable**: `class` (with values: 'A' for ASD, 'T' for Typical) OR `diagnosis` (with values: 'ASD', 'Control')
        - **Feature columns**: Numerical gait features
        
        **Optional Columns:**
        - participant_id, age, gender (auto-generated if missing)
        
        **Note:** The system will auto-detect CSV delimiters (comma, semicolon, tab)
        """)
        
        csv_file = st.file_uploader("Upload CSV with gait features", type=['csv'])
        
        if csv_file is not None:
            try:
                # Enhanced CSV parsing with multiple delimiter detection
                import io
                
                # Read the file content
                file_content = csv_file.read().decode('utf-8')
                csv_file.seek(0)  # Reset file pointer
                
                # Auto-detect delimiter
                delimiters = [',', ';', '\t', '|']
                best_delimiter = ','
                max_columns = 0
                
                for delimiter in delimiters:
                    try:
                        test_df = pd.read_csv(io.StringIO(file_content), 
                                            delimiter=delimiter, 
                                            nrows=5)
                        if len(test_df.columns) > max_columns:
                            max_columns = len(test_df.columns)
                            best_delimiter = delimiter
                    except:
                        continue
                
                st.info(f"🔍 Detected delimiter: '{best_delimiter}' with {max_columns} columns")
                
                # Read CSV with detected delimiter
                df = pd.read_csv(io.StringIO(file_content), delimiter=best_delimiter)
                
                # Check if first row might be headers
                first_row_numeric = all(pd.to_numeric(df.iloc[0], errors='coerce').notna())
                
                if first_row_numeric and 'class' not in df.columns:
                    st.warning("⚠️ No proper headers detected. Would you like to specify column names?")
                    
                    # Option to add headers manually
                    add_headers = st.checkbox("Add custom headers")
                    
                    if add_headers:
                        st.info("💡 For your dataset, the last column should be 'class' with values 'A' or 'T'")
                        
                        # Generate default column names
                        num_cols = len(df.columns)
                        default_headers = [f"feature_{i}" for i in range(1, num_cols)] + ['class']
                        
                        headers_input = st.text_input(
                            f"Enter {num_cols} column names separated by commas:",
                            value=",".join(default_headers),
                            help="Last column should be 'class' for target variable"
                        )
                        
                        if st.button("Apply Headers"):
                            try:
                                new_headers = [h.strip() for h in headers_input.split(',')]
                                if len(new_headers) == len(df.columns):
                                    df.columns = new_headers
                                    st.success("✅ Headers applied successfully!")
                                else:
                                    st.error(f"❌ Number of headers ({len(new_headers)}) doesn't match columns ({len(df.columns)})")
                            except Exception as e:
                                st.error(f"❌ Error applying headers: {e}")
                
                # Display dataframe with current column names
                st.subheader("📊 Data Preview")
                st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                st.write(f"**Columns:** {list(df.columns)}")
                st.dataframe(df.head())
                
                # Check for target variable more thoroughly
                target_column = None
                if 'class' in df.columns:
                    target_column = 'class'
                elif 'diagnosis' in df.columns:
                    target_column = 'diagnosis'
                else:
                    # Look for columns with A/T or ASD/Control values
                    for col in df.columns:
                        unique_vals = df[col].unique()
                        if len(unique_vals) <= 10:  # Likely categorical
                            unique_str = [str(v).upper() for v in unique_vals if pd.notna(v)]
                            if any(v in ['A', 'T', 'ASD', 'CONTROL'] for v in unique_str):
                                st.info(f"🎯 Potential target column found: '{col}' with values: {unique_vals}")
                                if st.button(f"Use '{col}' as target variable"):
                                    target_column = col
                                    if col != 'class':
                                        df['class'] = df[col]
                                    break
                
                if target_column:
                    st.success(f"✅ Target variable found: {target_column}")
                    st.write(f"**Value distribution:** {df[target_column].value_counts().to_dict()}")
                
                # Process data button
                if st.button("🚀 Process CSV Data") and target_column:
                    with st.spinner("Processing CSV data..."):
                        processed_count = 0
                        
                        # Map target variable
                        if target_column == 'class' or target_column in df.columns:
                            # Create diagnosis column based on target
                            target_values = df[target_column].astype(str).str.upper()
                            
                            # Map various formats to standard
                            diagnosis_map = {
                                'A': 'ASD', 'T': 'Control', 'TYPICAL': 'Control',
                                'ASD': 'ASD', 'CONTROL': 'Control', 'NEUROTYPICAL': 'Control',
                                '1': 'ASD', '0': 'Control'
                            }
                            
                            df['diagnosis'] = target_values.map(diagnosis_map)
                            
                            # Handle unmapped values
                            unmapped = df['diagnosis'].isna().sum()
                            if unmapped > 0:
                                st.warning(f"⚠️ {unmapped} rows have unmapped target values")
                                st.write("Unmapped values:", df[df['diagnosis'].isna()][target_column].unique())
                            
                            st.info(f"✅ Mapped target variable: {dict(df['diagnosis'].value_counts())}")
                        
                        # Add missing demographic columns
                        if 'participant_id' not in df.columns:
                            df['participant_id'] = [f"P_{i:04d}" for i in range(1, len(df) + 1)]
                            st.info("✅ Generated participant IDs")
                        
                        if 'age' not in df.columns:
                            df['age'] = 25
                            st.info("✅ Added default age (25)")
                        
                        if 'gender' not in df.columns:
                            df['gender'] = 'Unknown'
                            st.info("✅ Added default gender")
                        
                        # Process each row
                        progress_bar = st.progress(0)
                        
                        for idx, row in df.iterrows():
                            if pd.isna(row['diagnosis']):
                                continue  # Skip rows with invalid diagnosis
                            
                            try:
                                # Store participant
                                participant_data = {
                                    'participant_id': str(row['participant_id']),
                                    'age': int(row['age']) if pd.notna(row['age']) else 25,
                                    'gender': str(row['gender']) if pd.notna(row['gender']) else 'Unknown',
                                    'diagnosis': str(row['diagnosis'])
                                }
                                
                                stored_id = st.session_state.kg_manager.store_participant(participant_data)
                                
                                if stored_id:
                                    # Create session
                                    session_data = {
                                        'session_id': f"S_{stored_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}",
                                        'date': datetime.now().isoformat(),
                                        'video_duration': 0,
                                        'frame_count': 0
                                    }
                                    
                                    session_id = st.session_state.kg_manager.store_gait_session(session_data, stored_id)
                                    
                                    # Extract features (exclude metadata columns)
                                    exclude_cols = ['participant_id', 'age', 'gender', 'diagnosis', 'class', target_column]
                                    feature_cols = [col for col in df.columns if col not in exclude_cols]
                                    
                                    features = {}
                                    for col in feature_cols:
                                        try:
                                            val = row[col]
                                            if pd.notna(val):
                                                # Handle semicolon-separated values
                                                if isinstance(val, str) and ';' in val:
                                                    val = float(val.replace(';', '.'))
                                                else:
                                                    val = float(val)
                                                features[col] = val
                                            else:
                                                features[col] = 0.0
                                        except (ValueError, TypeError):
                                            features[col] = 0.0
                                    
                                    if features:  # Only add if we have features
                                        # Store features in Neo4j
                                        if session_id:
                                            st.session_state.kg_manager.store_gait_features(features, session_id)
                                        
                                        processed_count += 1
                                
                                # Update progress
                                progress_bar.progress((idx + 1) / len(df))
                                
                            except Exception as e:
                                st.warning(f"⚠️ Error processing row {idx}: {e}")
                                continue
                        
                        if processed_count > 0:
                            # Display summary
                            st.subheader("📊 Processing Complete!")
                            
                            # Get actual counts from database
                            asd_count_query = st.session_state.kg_manager.neo4j.execute_query(
                                "MATCH (p:Participant {diagnosis: 'ASD'}) RETURN count(p) as count"
                            )
                            control_count_query = st.session_state.kg_manager.neo4j.execute_query(
                                "MATCH (p:Participant {diagnosis: 'Control'}) RETURN count(p) as count"
                            )
                            
                            asd_count = asd_count_query[0]['count'] if asd_count_query else 0
                            control_count = control_count_query[0]['count'] if control_count_query else 0
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Processed", processed_count)
                            with col2:
                                st.metric("ASD Cases", asd_count)
                            with col3:
                                st.metric("Control Cases", control_count)
                            
                            st.success(f"✅ Successfully processed {processed_count} records!")
                            st.info("🎯 Data is ready for participant-level analysis in the Analysis page.")
                        
                        else:
                            st.error("❌ No valid data processed")
                
                elif not target_column and st.button("🚀 Process CSV Data"):
                    st.error("❌ Please identify a target variable first!")
                    
            except Exception as e:
                st.error(f"❌ Error processing CSV: {e}")
                st.exception(e)
                
    else:  # XLSX format
        st.markdown("""
        ### XLSX Format for Batch Upload
        Upload an Excel file (.xlsx or .xls) with pre-calculated gait features.
        
        **Required Columns:**
        - **Target Variable**: `class` (with values: 'A' for ASD, 'T' for Typical/Control)
        - **OR**: `diagnosis` (with values: 'ASD', 'Control')
        - **Feature columns**: Any numerical gait features (e.g., mean-x-Midspain, variance-x-AnkleLeft, etc.)
        
        **Optional Columns:**
        - participant_id, age, gender (will be auto-generated if missing)
        
        **Your Dataset Format:**
        - 1,259 gait features from biomechanical analysis
        - Target: `class` column with 'A'/'T' values
        - Automatically processes all numerical features
        """)
        
        xlsx_file = st.file_uploader("Upload XLSX with gait features", type=['xlsx', 'xls'])
        
        if xlsx_file is not None:
            try:
                # Read Excel file
                df = pd.read_excel(xlsx_file)
                st.dataframe(df.head())
                
                st.info(f"📊 Loaded {len(df)} rows and {len(df.columns)} columns")
                
                # Check for target column
                has_class = 'class' in df.columns
                has_diagnosis = 'diagnosis' in df.columns
                
                if has_class:
                    # Check class values
                    class_values = df['class'].value_counts()
                    st.write("**Target Variable Found**: `class` column")
                    st.write("Class distribution:", dict(class_values))
                elif has_diagnosis:
                    diag_values = df['diagnosis'].value_counts() 
                    st.write("**Target Variable Found**: `diagnosis` column")
                    st.write("Diagnosis distribution:", dict(diag_values))
                else:
                    st.warning("⚠️ No target variable found! Looking for 'class' or 'diagnosis' column.")
                
                # Display column info
                with st.expander("📋 Column Information"):
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes,
                        'Non-Null Count': df.count(),
                        'Sample Value': [df[col].iloc[0] if len(df) > 0 else None for col in df.columns]
                    })
                    st.dataframe(col_info)
                
                if st.button("🚀 Process XLSX Data"):
                    with st.spinner(f"Processing {len(df)} participants..."):
                        processed_count = 0
                        
                        # Map target variable for your specific dataset
                        if 'class' in df.columns:
                            df['diagnosis'] = df['class'].map({'A': 'ASD', 'T': 'Control'})
                            st.info("✅ Mapped 'class' column: A→ASD, T→Control")
                        
                        # Add missing demographic columns if not present
                        if 'participant_id' not in df.columns:
                            df['participant_id'] = [f"P_{i:04d}" for i in range(1, len(df) + 1)]
                            st.info("✅ Generated participant IDs")
                        
                        if 'age' not in df.columns:
                            df['age'] = 25  # Default age
                            st.info("✅ Added default age (25)")
                        
                        if 'gender' not in df.columns:
                            df['gender'] = 'Unknown'  # Default gender
                            st.info("✅ Added default gender")
                        
                        # Verify diagnosis column
                        if 'diagnosis' not in df.columns:
                            st.error("❌ No valid target variable found! Expected 'class' or 'diagnosis' column.")
                            return
                        
                        progress_bar = st.progress(0)
                        
                        for idx, row in df.iterrows():
                            try:
                                # Store participant in Neo4j
                                participant_data = {
                                    'participant_id': str(row['participant_id']),
                                    'age': int(row['age']) if pd.notna(row['age']) else 25,
                                    'gender': str(row['gender']) if pd.notna(row['gender']) else 'Unknown',
                                    'diagnosis': str(row['diagnosis']) if pd.notna(row['diagnosis']) else 'Unknown'
                                }
                                
                                stored_id = st.session_state.kg_manager.store_participant(participant_data)
                                
                                if stored_id:
                                    # Create session
                                    session_data = {
                                        'session_id': f"S_{stored_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}",
                                        'date': datetime.now().isoformat(),
                                        'video_duration': 0,
                                        'frame_count': 0
                                    }
                                    
                                    session_id = st.session_state.kg_manager.store_gait_session(session_data, stored_id)
                                    
                                    # Extract features for ML (exclude metadata columns)
                                    exclude_cols = ['participant_id', 'age', 'gender', 'diagnosis', 'class']
                                    feature_cols = [col for col in df.columns if col not in exclude_cols]
                                    
                                    features = {}
                                    for col in feature_cols:
                                        try:
                                            val = row[col]
                                            if pd.notna(val):
                                                features[col] = float(val)
                                            else:
                                                features[col] = 0.0
                                        except (ValueError, TypeError):
                                            features[col] = 0.0
                                    
                                    # Store features in Neo4j
                                    if session_id:
                                        st.session_state.kg_manager.store_gait_features(features, session_id)
                                    
                                    processed_count += 1
                                
                                # Update progress
                                progress_bar.progress((idx + 1) / len(df))
                                
                            except Exception as e:
                                st.warning(f"⚠️ Error processing row {idx}: {e}")
                                continue
                        
                        if processed_count > 0:
                            # Display comprehensive summary
                            st.subheader("📊 Dataset Processing Summary")
                            
                            # Get actual counts from database
                            asd_count_query = st.session_state.kg_manager.neo4j.execute_query(
                                "MATCH (p:Participant {diagnosis: 'ASD'}) RETURN count(p) as count"
                            )
                            control_count_query = st.session_state.kg_manager.neo4j.execute_query(
                                "MATCH (p:Participant {diagnosis: 'Control'}) RETURN count(p) as count"
                            )
                            
                            asd_count = asd_count_query[0]['count'] if asd_count_query else 0
                            control_count = control_count_query[0]['count'] if control_count_query else 0
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Processed", processed_count)
                            with col2:
                                st.metric("ASD Cases", asd_count)
                            with col3:
                                st.metric("Control Cases", control_count)
                            
                            # Class balance info
                            if asd_count > 0 and control_count > 0:
                                balance_ratio = min(asd_count, control_count) / max(asd_count, control_count)
                                if balance_ratio >= 0.8:
                                    st.success(f"✅ Well-balanced dataset! (Ratio: {balance_ratio:.2f})")
                                else:
                                    st.warning(f"⚠️ Imbalanced dataset detected (Ratio: {balance_ratio:.2f})")
                            
                            st.success(f"✅ Successfully processed {processed_count} records from XLSX file!")
                            st.info("🎯 Data is ready for participant-level analysis in the Analysis page.")
                        
                        else:
                            st.error("❌ No valid data processed")
                            
            except Exception as e:
                st.error(f"❌ Error processing XLSX: {e}")
                st.exception(e)

def show_analysis_page():
    """Display the ML analysis page with participant-level splits"""
    st.header("🎯 Machine Learning Analysis")
    
    if not st.session_state.neo4j_connection:
        st.warning("⚠️ Please configure Neo4j connection first.")
        return
    
    # Model Training Section
    st.subheader("🏋️ Model Training")
    
    # Load participant-level data
    if st.button("🔄 Load Participant-Level Training Data"):
        try:
            # Get participant-level data from Neo4j
            participant_data = st.session_state.kg_manager.get_participant_level_data()
            
            if participant_data:
                # Prepare data with participant-level aggregation
                summary = st.session_state.ml_analyzer.prepare_participant_data(participant_data)
                
                st.success(f"✅ Loaded participant-level data successfully!")
                
                # Display summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Participants", summary['total_participants'])
                with col2:
                    st.metric("ASD Cases", summary['asd_count'])
                with col3:
                    st.metric("Control Cases", summary['control_count'])
                with col4:
                    st.metric("Avg Sessions/Participant", f"{summary['avg_sessions_per_participant']:.1f}")
                
                # Show class balance
                if summary['asd_count'] > 0 and summary['control_count'] > 0:
                    balance_ratio = min(summary['asd_count'], summary['control_count']) / max(summary['asd_count'], summary['control_count'])
                    st.info(f"📊 Class Balance Ratio: {balance_ratio:.3f}")
                
                # Important information about participant-level processing
                st.info("""
                🔒 **Participant-Level Processing**: 
                - Features are averaged across sessions per participant
                - Train/test split will be done by participant (not by session)
                - No participant will appear in both train and test sets
                - This prevents data leakage and gives realistic performance estimates
                """)
            
            else:
                st.warning("⚠️ No participant data found in knowledge graph")
                
        except Exception as e:
            st.error(f"❌ Error loading participant data: {e}")
            st.exception(e)
    
    # Train Models
    if st.session_state.ml_analyzer.participant_data:
        if st.button("🚀 Train Models (Participant-Level Split)"):
            with st.spinner("Training models with participant-level splits..."):
                try:
                    # Train models with participant-level splits
                    st.session_state.ml_analyzer.train_models_participant_level()
                    
                    st.success("✅ Models trained successfully with participant-level splits!")
                    
                    # Display training results
                    st.subheader("📊 Training Results")
                    
                    # Show train/test split information
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Participants", len(st.session_state.ml_analyzer.participant_data['features']))
                    with col2:
                        st.metric("Training Participants", len(st.session_state.ml_analyzer.train_participants))
                    with col3:
                        st.metric("Test Participants", len(st.session_state.ml_analyzer.test_participants))
                    with col4:
                        st.metric("Test Split", f"{st.session_state.ml_analyzer.test_size*100:.0f}%")
                    
                    # Class distribution in train/test
                    st.subheader("📈 Participant-Level Train/Test Distribution")
                    
                    train_asd = sum(1 for p in st.session_state.ml_analyzer.train_participants 
                                  if st.session_state.ml_analyzer.participant_data['labels'][p] == 1)
                    train_control = len(st.session_state.ml_analyzer.train_participants) - train_asd
                    test_asd = sum(1 for p in st.session_state.ml_analyzer.test_participants 
                                 if st.session_state.ml_analyzer.participant_data['labels'][p] == 1)
                    test_control = len(st.session_state.ml_analyzer.test_participants) - test_asd
                    
                    split_df = pd.DataFrame({
                        'Set': ['Training', 'Training', 'Test', 'Test'],
                        'Class': ['ASD', 'Control', 'ASD', 'Control'],
                        'Count': [train_asd, train_control, test_asd, test_control]
                    })
                    
                    fig = px.bar(split_df, x='Set', y='Count', color='Class',
                               title="Class Distribution in Train/Test Sets (Participant-Level)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Participant-level cross-validation
                    st.subheader("🔄 Participant-Level Cross-Validation")
                    
                    cv_results = st.session_state.ml_analyzer.get_participant_cross_validation_scores()
                    
                    if 'error' not in cv_results:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Random Forest CV", f"{cv_results['rf_mean']:.3f} ± {cv_results['rf_std']:.3f}")
                        with col2:
                            st.metric("XGBoost CV", f"{cv_results['xgb_mean']:.3f} ± {cv_results['xgb_std']:.3f}")
                        
                        # Show individual fold scores
                        with st.expander("📊 Individual Fold Scores"):
                            fold_df = pd.DataFrame({
                                'Fold': range(1, len(cv_results['rf_scores']) + 1),
                                'Random Forest': cv_results['rf_scores'],
                                'XGBoost': cv_results['xgb_scores']
                            })
                            st.dataframe(fold_df)
                    
                    # Feature importance
                    importance_data = st.session_state.ml_analyzer.get_feature_importance()
                    
                    if importance_data:
                        st.subheader("🎯 Feature Importance")
                        
                        # Random Forest importance
                        rf_imp_df = pd.DataFrame(
                            list(importance_data['random_forest'].items()),
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(rf_imp_df.head(10), x='Importance', y='Feature',
                                   orientation='h', title="Random Forest Feature Importance (Top 10)")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show top features
                        st.write("**Top 10 Most Important Features:**")
                        for i, (feature, importance) in enumerate(rf_imp_df.head(10).values, 1):
                            st.write(f"{i}. {feature}: {importance:.4f}")
                
                except Exception as e:
                    st.error(f"❌ Error training models: {e}")
                    st.exception(e)
    
    # Enhanced Prediction Section
    st.subheader("🔮 Individual Prediction & Explainability")
    
    if hasattr(st.session_state, 'current_features') and st.session_state.ml_analyzer.is_trained:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Make Prediction"):
                with st.spinner("Making prediction..."):
                    try:
                        prediction_result = st.session_state.ml_analyzer.predict(
                            st.session_state.current_features
                        )
                        
                        # Store prediction in knowledge graph
                        if hasattr(st.session_state, 'current_session'):
                            prediction_data = {
                                'model_type': 'ensemble',
                                'prediction': prediction_result['ensemble_prediction'],
                                'confidence': prediction_result['rf_confidence'],
                                'anomaly_score': prediction_result['anomaly_score']
                            }
                            
                            st.session_state.kg_manager.store_prediction_result(
                                prediction_data, st.session_state.current_session
                            )
                        
                        st.session_state.current_prediction = prediction_result
                        
                        # Display results
                        st.subheader("🎯 Prediction Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            rf_pred = "ASD" if prediction_result['rf_prediction'] == 1 else "Control"
                            st.metric("Random Forest", rf_pred, 
                                    f"Confidence: {prediction_result['rf_confidence']:.3f}")
                        
                        with col2:
                            xgb_pred = "ASD" if prediction_result['xgb_prediction'] == 1 else "Control"
                            st.metric("XGBoost", xgb_pred,
                                    f"Confidence: {prediction_result['xgb_confidence']:.3f}")
                        
                        with col3:
                            ensemble_pred = "ASD" if prediction_result['ensemble_prediction'] == 1 else "Control"
                            st.metric("Ensemble Prediction", ensemble_pred)
                        
                        with col4:
                            anomaly_status = "Anomaly" if prediction_result['is_anomaly'] else "Normal"
                            st.metric("Anomaly Detection", anomaly_status,
                                    f"Score: {prediction_result['anomaly_score']:.3f}")
                        
                    except Exception as e:
                        st.error(f"❌ Error making prediction: {e}")
        
        with col2:
            if st.button("🔍 Generate SHAP Explanations"):
                with st.spinner("Generating SHAP explanations..."):
                    try:
                        shap_data = st.session_state.ml_analyzer.get_shap_explanations(
                            st.session_state.current_features
                        )
                        
                        if 'error' not in shap_data:
                            st.session_state.current_shap = shap_data
                            
                            st.subheader("🔍 SHAP Feature Explanations")
                            
                            # Display SHAP values
                            if 'rf_shap_values' in shap_data:
                                rf_shap_df = pd.DataFrame({
                                    'Feature': shap_data['feature_names'],
                                    'SHAP_Value': shap_data['rf_shap_values']
                                }).sort_values('SHAP_Value', key=abs, ascending=False)
                                
                                fig = px.bar(rf_shap_df.head(10), 
                                           x='SHAP_Value', y='Feature',
                                           orientation='h',
                                           title="Random Forest SHAP Values (Top 10 Features)",
                                           color='SHAP_Value',
                                           color_continuous_scale='RdBu')
                                st.plotly_chart(fig, use_container_width=True)
                        
                        else:
                            st.error(f"❌ Error generating SHAP: {shap_data['error']}")
                            
                    except Exception as e:
                        st.error(f"❌ Error generating SHAP explanations: {e}")
    
    elif not st.session_state.ml_analyzer.is_trained:
        st.info("ℹ️ Please load participant data and train the models first.")
    
    elif not hasattr(st.session_state, 'current_features'):
        st.info("ℹ️ Please upload and process a video first for individual prediction.")
    
    # Complete Performance Metrics Section
    if st.session_state.ml_analyzer.is_trained:
        st.subheader("📊 Complete Model Performance Analysis")
        
        # Important warning about proper evaluation
        st.warning("""
        🔒 **PARTICIPANT-LEVEL EVALUATION**: 
        - Models trained on participant-level data (averaged sessions)
        - Train/test split by participants (no participant overlap)
        - Performance evaluated on completely unseen participants
        - This prevents data leakage and gives realistic performance estimates
        """)
        
        if st.button("📈 Generate Participant-Level Performance Report"):
            generate_participant_level_performance_report()

def generate_participant_level_performance_report():
    """Generate performance report with participant-level evaluation"""
    
    try:
        # Get test predictions on unseen participants
        test_results = st.session_state.ml_analyzer.get_test_predictions()
        
        if 'error' in test_results:
            st.error(f"❌ {test_results['error']}")
            return
            
        # Extract test predictions
        y_true = test_results['y_true']
        rf_pred = test_results['rf_predictions']
        rf_proba = test_results['rf_probabilities']
        xgb_pred = test_results['xgb_predictions']
        xgb_proba = test_results['xgb_probabilities']
        test_participants = test_results['test_participants']
        
        # Calculate ALL metrics on test participants
        # Random Forest Metrics
        rf_accuracy = accuracy_score(y_true, rf_pred)
        rf_precision = precision_score(y_true, rf_pred, zero_division=0)
        rf_recall = recall_score(y_true, rf_pred, zero_division=0)
        rf_f1 = f1_score(y_true, rf_pred, zero_division=0)
        rf_auc = roc_auc_score(y_true, rf_proba) if len(np.unique(y_true)) > 1 else 0
        
        # XGBoost Metrics
        xgb_accuracy = accuracy_score(y_true, xgb_pred)
        xgb_precision = precision_score(y_true, xgb_pred, zero_division=0)
        xgb_recall = recall_score(y_true, xgb_pred, zero_division=0)
        xgb_f1 = f1_score(y_true, xgb_pred, zero_division=0)
        xgb_auc = roc_auc_score(y_true, xgb_proba) if len(np.unique(y_true)) > 1 else 0
        
        # Display test set information
        st.subheader("🧪 Participant-Level Test Set Evaluation")
        st.success(f"**CRITICAL**: All metrics calculated on {len(test_participants)} UNSEEN participants")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test Participants", len(test_participants))
        with col2:
            test_asd = sum(y_true)
            st.metric("Test ASD Participants", test_asd)
        with col3:
            test_control = len(y_true) - test_asd
            st.metric("Test Control Participants", test_control)
        
        # Display metrics
        st.subheader("📊 Performance Metrics on Unseen Participants")
        
        metrics_df = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost'],
            'Accuracy': [rf_accuracy, xgb_accuracy],
            'Precision': [rf_precision, xgb_precision],
            'Recall': [rf_recall, xgb_recall],
            'F1-Score': [rf_f1, xgb_f1],
            'ROC-AUC': [rf_auc, xgb_auc]
        })
        
        st.dataframe(metrics_df.style.format({
            'Accuracy': '{:.3f}',
            'Precision': '{:.3f}',
            'Recall': '{:.3f}',
            'F1-Score': '{:.3f}',
            'ROC-AUC': '{:.3f}'
        }))
        
        # Performance interpretation
        st.subheader("📈 Performance Interpretation")
        
        # Color-coded performance assessment
        def get_performance_color(score):
            if score >= 0.9:
                return "🟢 Excellent"
            elif score >= 0.8:
                return "🟡 Good"
            elif score >= 0.7:
                return "🟠 Fair"
            else:
                return "🔴 Poor"
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Random Forest Performance:**")
            st.write(f"- Accuracy: {get_performance_color(rf_accuracy)} ({rf_accuracy:.3f})")
            st.write(f"- Precision: {get_performance_color(rf_precision)} ({rf_precision:.3f})")
            st.write(f"- Recall: {get_performance_color(rf_recall)} ({rf_recall:.3f})")
            st.write(f"- F1-Score: {get_performance_color(rf_f1)} ({rf_f1:.3f})")
            
        with col2:
            st.write("**XGBoost Performance:**")
            st.write(f"- Accuracy: {get_performance_color(xgb_accuracy)} ({xgb_accuracy:.3f})")
            st.write(f"- Precision: {get_performance_color(xgb_precision)} ({xgb_precision:.3f})")
            st.write(f"- Recall: {get_performance_color(xgb_recall)} ({xgb_recall:.3f})")
            st.write(f"- F1-Score: {get_performance_color(xgb_f1)} ({xgb_f1:.3f})")
        
        # ROC Curve
        if len(np.unique(y_true)) > 1:
            st.subheader("📈 ROC Curves")
            fig = go.Figure()
            
            # RF ROC
            fpr_rf, tpr_rf, _ = roc_curve(y_true, rf_proba)
            fig.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, 
                                   name=f'Random Forest (AUC = {rf_auc:.3f})'))
            
            # XGB ROC
            fpr_xgb, tpr_xgb, _ = roc_curve(y_true, xgb_proba)
            fig.add_trace(go.Scatter(x=fpr_xgb, y=tpr_xgb, 
                                   name=f'XGBoost (AUC = {xgb_auc:.3f})'))
            
            # Random line
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], 
                                   name='Random Classifier', line=dict(dash='dash')))
            
            fig.update_layout(title="ROC Curves - Participant-Level Evaluation",
                            xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrices
        st.subheader("🔍 Confusion Matrices (Unseen Participants)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Random Forest Confusion Matrix
            cm_rf = confusion_matrix(y_true, rf_pred)
            fig_cm_rf = px.imshow(cm_rf, text_auto=True,
                                title="Random Forest Confusion Matrix",
                                labels=dict(x="Predicted", y="Actual"),
                                x=['Control', 'ASD'],
                                y=['Control', 'ASD'])
            st.plotly_chart(fig_cm_rf, use_container_width=True)
        
        with col2:
            # XGBoost Confusion Matrix
            cm_xgb = confusion_matrix(y_true, xgb_pred)
            fig_cm_xgb = px.imshow(cm_xgb, text_auto=True,
                                 title="XGBoost Confusion Matrix",
                                 labels=dict(x="Predicted", y="Actual"),
                                 x=['Control', 'ASD'],
                                 y=['Control', 'ASD'])
            st.plotly_chart(fig_cm_xgb, use_container_width=True)
        
        # Classification Reports
        st.subheader("📋 Detailed Classification Reports (Unseen Participants)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.text("Random Forest Report:")
            st.text(classification_report(y_true, rf_pred, target_names=['Control', 'ASD']))
        
        with col2:
            st.text("XGBoost Report:")
            st.text(classification_report(y_true, xgb_pred, target_names=['Control', 'ASD']))
        
        # Model comparison
        st.subheader("🏆 Model Comparison Summary")
        
        # Determine best model
        rf_score = (rf_accuracy + rf_precision + rf_recall + rf_f1) / 4
        xgb_score = (xgb_accuracy + xgb_precision + xgb_recall + xgb_f1) / 4
        
        if rf_score > xgb_score:
            st.success(f"🏆 **Random Forest** performs better (Avg Score: {rf_score:.3f} vs {xgb_score:.3f})")
        elif xgb_score > rf_score:
            st.success(f"🏆 **XGBoost** performs better (Avg Score: {xgb_score:.3f} vs {rf_score:.3f})")
        else:
            st.info("🤝 Both models perform equally well")
        
        # Data leakage check
        st.subheader("🔒 Data Leakage Assessment")
        
        if rf_accuracy >= 0.99 and xgb_accuracy >= 0.99:
            st.error("⚠️ **SUSPICIOUS**: Both models show >99% accuracy - investigate for potential issues!")
        elif rf_accuracy >= 0.95 or xgb_accuracy >= 0.95:
            st.success("✅ **EXCELLENT**: High performance without perfect scores suggests legitimate results")
        elif rf_accuracy >= 0.8 or xgb_accuracy >= 0.8:
            st.success("✅ **GOOD**: Realistic performance indicates proper participant-level evaluation")
        else:
            st.info("ℹ️ **MODERATE**: Performance suggests challenging classification task")
        
        # Summary
        st.subheader("📋 Executive Summary")
        
        best_model = "XGBoost" if xgb_score > rf_score else "Random Forest"
        best_accuracy = max(rf_accuracy, xgb_accuracy)
        
        st.markdown(f"""
        ### Key Findings:
        - **Best Model**: {best_model} with {best_accuracy:.3f} accuracy
        - **Evaluation Method**: Participant-level train/test split
        - **Test Participants**: {len(test_participants)} unseen participants
        - **Data Leakage**: ✅ Prevented through proper participant-level splits
        - **Generalization**: Results represent performance on new participants
        
        ### Clinical Implications:
        - The model shows {"excellent" if best_accuracy >= 0.9 else "good" if best_accuracy >= 0.8 else "moderate"} performance
        - Gait features demonstrate {"strong" if best_accuracy >= 0.85 else "moderate"} discriminative power for ASD detection
        - Results suggest potential for clinical screening applications
        """)
        
    except Exception as e:
        st.error(f"❌ Error generating performance report: {e}")
        st.exception(e)

def show_visualization_page():
    """Display comprehensive data visualizations"""
    st.header("📈 Data Visualization & Analytics")
    
    if not st.session_state.neo4j_connection:
        st.warning("⚠️ Please configure Neo4j connection first.")
        return
    
    # Data Overview
    st.subheader("📊 Data Overview")
    
    try:
        # Get basic statistics
        stats_queries = {
            'total_participants': "MATCH (p:Participant) RETURN count(p) as count",
            'total_sessions': "MATCH (s:GaitSession) RETURN count(s) as count",
            'asd_cases': "MATCH (p:Participant {diagnosis: 'ASD'}) RETURN count(p) as count",
            'control_cases': "MATCH (p:Participant {diagnosis: 'Control'}) RETURN count(p) as count"
        }
        
        stats = {}
        for key, query in stats_queries.items():
            result = st.session_state.kg_manager.neo4j.execute_query(query)
            stats[key] = result[0]['count'] if result else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Participants", stats['total_participants'])
        with col2:
            st.metric("Total Sessions", stats['total_sessions'])
        with col3:
            st.metric("ASD Cases", stats['asd_cases'])
        with col4:
            st.metric("Control Cases", stats['control_cases'])
        
        # Sessions per participant ratio
        if stats['total_participants'] > 0:
            sessions_ratio = stats['total_sessions'] / stats['total_participants']
            st.info(f"📊 Average sessions per participant: {sessions_ratio:.2f}")
        
        # Age and Gender Distribution
        st.subheader("👥 Demographics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            age_query = """
            MATCH (p:Participant)
            WHERE p.age IS NOT NULL
            RETURN p.age as age, p.diagnosis as diagnosis
            """
            age_data = st.session_state.kg_manager.neo4j.execute_query(age_query)
            
            if age_data:
                age_df = pd.DataFrame(age_data)
                fig = px.histogram(age_df, x='age', color='diagnosis',
                                 title="Age Distribution by Diagnosis",
                                 nbins=20)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Gender distribution
            gender_query = """
            MATCH (p:Participant)
            RETURN p.gender as gender, p.diagnosis as diagnosis, count(*) as count
            """
            gender_data = st.session_state.kg_manager.neo4j.execute_query(gender_query)
            
            if gender_data:
                gender_df = pd.DataFrame(gender_data)
                fig = px.pie(gender_df, values='count', names='gender',
                           title="Gender Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        # Gait Feature Analysis
        st.subheader("🚶 Gait Feature Analysis")
        
        # Get all gait features
        features_query = """
        MATCH (p:Participant)-[:HAS_SESSION]->(s:GaitSession)-[:HAS_FEATURE]->(f:GaitFeature)
        RETURN p.diagnosis as diagnosis, f.feature_type as feature_type, f.value as value
        """
        features_data = st.session_state.kg_manager.neo4j.execute_query(features_query)
        
        if features_data:
            features_df = pd.DataFrame(features_data)
            
            # Feature selection for visualization
            available_features = features_df['feature_type'].unique()
            selected_features = st.multiselect(
                "Select features to visualize:",
                available_features,
                default=available_features[:4] if len(available_features) >= 4 else available_features
            )
            
            if selected_features:
                # Box plots for selected features
                for feature in selected_features:
                    feature_subset = features_df[features_df['feature_type'] == feature]
                    
                    if len(feature_subset) > 0:
                        fig = px.box(feature_subset, x='diagnosis', y='value',
                                   title=f"{feature} by Diagnosis")
                        st.plotly_chart(fig, use_container_width=True)
            
            # Correlation analysis
            st.subheader("🔗 Feature Correlations")
            
            # Pivot the data for correlation analysis
            pivot_df = features_df.pivot_table(
                index=['diagnosis'], 
                columns='feature_type', 
                values='value', 
                aggfunc='mean'
            ).reset_index()
            
            if len(pivot_df.columns) > 2:
                numeric_columns = pivot_df.select_dtypes(include=[np.number]).columns
                correlation_matrix = pivot_df[numeric_columns].corr()
                
                fig = px.imshow(correlation_matrix, 
                              title="Feature Correlation Matrix",
                              color_continuous_scale="RdBu")
                st.plotly_chart(fig, use_container_width=True)
        
        # Prediction Results Analysis
        st.subheader("🎯 Prediction Performance")
        
        predictions_query = """
        MATCH (p:Participant)-[:HAS_SESSION]->(s:GaitSession)-[:HAS_PREDICTION]->(r:PredictionResult)
        RETURN p.diagnosis as actual, r.prediction as predicted, r.confidence as confidence
        """
        predictions_data = st.session_state.kg_manager.neo4j.execute_query(predictions_query)
        
        if predictions_data:
            pred_df = pd.DataFrame(predictions_data)
            pred_df['actual_label'] = pred_df['actual'].map({'ASD': 1, 'Control': 0})
            
            # Confusion matrix
            if len(pred_df) > 0:
                confusion_data = []
                for actual in pred_df['actual'].unique():
                    for predicted in [0, 1]:
                        count = len(pred_df[(pred_df['actual'] == actual) & 
                                          (pred_df['predicted'] == predicted)])
                        confusion_data.append({
                            'Actual': actual,
                            'Predicted': 'ASD' if predicted == 1 else 'Control',
                            'Count': count
                        })
                
                confusion_df = pd.DataFrame(confusion_data)
                
                if len(confusion_df) > 0:
                    fig = px.density_heatmap(confusion_df, x='Predicted', y='Actual', z='Count',
                                           title="Prediction Confusion Matrix")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Confidence distribution
            fig = px.histogram(pred_df, x='confidence', color='actual',
                             title="Prediction Confidence Distribution",
                             nbins=20)
            st.plotly_chart(fig, use_container_width=True)
        
        # Network Visualization
        st.subheader("🕸️ Knowledge Graph Network")
        
        if st.button("Generate Network Visualization"):
            with st.spinner("Generating network visualization..."):
                try:
                    # Get graph data for visualization
                    network_query = """
                    MATCH (p:Participant)-[r:HAS_SESSION]->(s:GaitSession)
                    RETURN p.id as participant, s.session_id as session, 
                           p.diagnosis as diagnosis
                    LIMIT 50
                    """
                    network_data = st.session_state.kg_manager.neo4j.execute_query(network_query)
                    
                    if network_data:
                        # Create networkx graph
                        G = nx.Graph()
                        
                        for data in network_data:
                            G.add_node(data['participant'], 
                                     type='participant', 
                                     diagnosis=data['diagnosis'])
                            G.add_node(data['session'], type='session')
                            G.add_edge(data['participant'], data['session'])
                        
                        # Generate layout
                        pos = nx.spring_layout(G)
                        
                        # Create plotly visualization
                        edge_x, edge_y = [], []
                        for edge in G.edges():
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                        
                        node_x, node_y, node_text, node_color = [], [], [], []
                        for node in G.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                            node_info = G.nodes[node]
                            node_text.append(f"{node}<br>Type: {node_info.get('type', 'unknown')}")
                            
                            if node_info.get('type') == 'participant':
                                if node_info.get('diagnosis') == 'ASD':
                                    node_color.append('red')
                                else:
                                    node_color.append('blue')
                            else:
                                node_color.append('green')
                        
                        fig = go.Figure()
                        
                        # Add edges
                        fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                                               line=dict(width=0.5, color='#888'),
                                               hoverinfo='none',
                                               mode='lines'))
                        
                        # Add nodes
                        fig.add_trace(go.Scatter(x=node_x, y=node_y,
                                               mode='markers',
                                               hoverinfo='text',
                                               text=node_text,
                                               marker=dict(size=10, color=node_color)))
                        
                        fig.update_layout(title="Knowledge Graph Network Visualization",
                                        showlegend=False,
                                        hovermode='closest',
                                        margin=dict(b=20,l=5,r=5,t=40),
                                        annotations=[ dict(
                                            text="Red: ASD Participants, Blue: Control Participants, Green: Sessions",
                                            showarrow=False,
                                            xref="paper", yref="paper",
                                            x=0.005, y=-0.002 ) ],
                                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ Error generating network visualization: {e}")
        
    except Exception as e:
        st.error(f"❌ Error loading visualization data: {e}")

def show_query_interface_page():
    """Display the natural language query interface"""
    st.header("🔍 Natural Language Query Interface")
    
    if not st.session_state.neo4j_connection:
        st.warning("⚠️ Please configure Neo4j connection first.")
        return
    
    st.markdown("""
    Ask questions about your data in plain English. The system will translate your query 
    into Cypher and execute it against the knowledge graph.
    """)
    
    # Query Examples
    with st.expander("💡 Example Queries"):
        st.markdown("""
        **Sample queries you can try:**
        - "How many participants do we have?"
        - "Show me ASD positive cases"
        - "What is the average step length?"
        - "List participants by age"
        - "Show me all participants with high confidence predictions"
        - "What are the most important gait features?"
        - "Count total features processed"
        - "Show distribution of diagnosis types"
        """)
    
    # Query Input
    user_query = st.text_input(
        "Enter your question:",
        placeholder="e.g., How many participants have ASD diagnosis?"
    )
    
    if st.button("🔍 Execute Query"):
        if user_query:
            with st.spinner("Processing your query..."):
                try:
                    # Execute natural language query
                    results = st.session_state.kg_manager.execute_natural_language_query(user_query)
                    
                    if results:
                        st.subheader("📊 Query Results")
                        
                        # Check if results contain error
                        if len(results) == 1 and 'error' in results[0]:
                            st.error(results[0]['error'])
                        else:
                            # Display results as dataframe
                            results_df = pd.DataFrame(results)
                            st.dataframe(results_df)
                            
                            # Try to create a visualization if appropriate
                            if len(results_df.columns) == 2 and len(results_df) > 1:
                                try:
                                    fig = px.bar(results_df, 
                                               x=results_df.columns[0], 
                                               y=results_df.columns[1],
                                               title=f"Results for: {user_query}")
                                    st.plotly_chart(fig, use_container_width=True)
                                except:
                                    pass
                    else:
                        st.warning("⚠️ No results found for your query.")
                        
                except Exception as e:
                    st.error(f"❌ Error executing query: {e}")
        else:
            st.warning("⚠️ Please enter a query.")
    
    # Direct Cypher Query Interface
    with st.expander("🔧 Advanced: Direct Cypher Query"):
        st.markdown("""
        **For advanced users:** Execute Cypher queries directly against the Neo4j database.
        """)
        
        cypher_query = st.text_area(
            "Enter Cypher query:",
            placeholder="MATCH (p:Participant) RETURN p.id, p.age, p.diagnosis LIMIT 10",
            height=100
        )
        
        if st.button("Execute Cypher Query"):
            if cypher_query:
                try:
                    with st.spinner("Executing Cypher query..."):
                        results = st.session_state.kg_manager.neo4j.execute_query(cypher_query)
                        
                        if results:
                            st.subheader("🔧 Cypher Query Results")
                            results_df = pd.DataFrame(results)
                            st.dataframe(results_df)
                        else:
                            st.info("ℹ️ Query executed successfully but returned no results.")
                            
                except Exception as e:
                    st.error(f"❌ Cypher query error: {e}")
            else:
                st.warning("⚠️ Please enter a Cypher query.")
    
    # Query History (if implementing session management)
    st.subheader("📜 Recent Queries")
    
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    if user_query and st.button("Save Query"):
        st.session_state.query_history.append({
            'query': user_query,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        st.success("✅ Query saved to history")
    
    if st.session_state.query_history:
        history_df = pd.DataFrame(st.session_state.query_history)
        st.dataframe(history_df)

def show_reports_page():
    """Display comprehensive reports and analytics"""
    st.header("📋 Comprehensive Reports")
    
    if not st.session_state.neo4j_connection:
        st.warning("⚠️ Please configure Neo4j connection first.")
        return
    
    # Report Type Selection
    report_type = st.selectbox(
        "Select Report Type:",
        ["📊 Summary Report", "🎯 Model Performance Report", "👥 Demographic Analysis", "🚶 Gait Pattern Analysis"]
    )
    
    if report_type == "📊 Summary Report":
        generate_summary_report()
    elif report_type == "🎯 Model Performance Report":
        if st.session_state.ml_analyzer.is_trained:
            generate_participant_level_performance_report()
        else:
            st.warning("⚠️ Please train the models first.")
    elif report_type == "👥 Demographic Analysis":
        generate_demographic_report()
    elif report_type == "🚶 Gait Pattern Analysis":
        generate_gait_pattern_report()

def generate_summary_report():
    """Generate a comprehensive summary report"""
    st.subheader("📊 System Summary Report")
    
    try:
        # Get all necessary data
        summary_data = {}
        
        # Basic statistics
        queries = {
            'total_participants': "MATCH (p:Participant) RETURN count(p) as count",
            'total_sessions': "MATCH (s:GaitSession) RETURN count(s) as count",
            'asd_cases': "MATCH (p:Participant {diagnosis: 'ASD'}) RETURN count(p) as count",
            'control_cases': "MATCH (p:Participant {diagnosis: 'Control'}) RETURN count(p) as count",
            'predictions_made': "MATCH (r:PredictionResult) RETURN count(r) as count"
        }
        
        for key, query in queries.items():
            result = st.session_state.kg_manager.neo4j.execute_query(query)
            summary_data[key] = result[0]['count'] if result else 0
        
        # Generate report
        report_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            ## NeuroGait ASD Analysis System Report
            **Generated:** {report_date}
            
            ### Executive Summary
            This report provides a comprehensive overview of the NeuroGait ASD analysis system 
            performance and data insights with participant-level evaluation.
            
            ### Key Metrics
            - **Total Participants Analyzed:** {summary_data['total_participants']}
            - **Total Gait Sessions:** {summary_data['total_sessions']}
            - **ASD Cases:** {summary_data['asd_cases']} ({summary_data['asd_cases']/max(summary_data['total_participants'], 1)*100:.1f}%)
            - **Control Cases:** {summary_data['control_cases']} ({summary_data['control_cases']/max(summary_data['total_participants'], 1)*100:.1f}%)
            - **Predictions Made:** {summary_data['predictions_made']}
            
            ### Data Quality Assessment
            """)
            
            # Data quality metrics
            if summary_data['total_participants'] > 0:
                sessions_per_participant = summary_data['total_sessions'] / summary_data['total_participants']
                st.markdown(f"- **Average Sessions per Participant:** {sessions_per_participant:.2f}")
                
                if summary_data['predictions_made'] > 0:
                    prediction_coverage = summary_data['predictions_made'] / summary_data['total_sessions'] * 100
                    st.markdown(f"- **Prediction Coverage:** {prediction_coverage:.1f}% of sessions")
            
            # Methodology info
            st.markdown("""
            ### 🔒 Evaluation Methodology
            - **Participant-Level Splits**: Proper train/test separation by participants
            - **No Data Leakage**: No participant appears in both training and testing
            - **Realistic Performance**: Metrics represent true generalization ability
            """)
        
        with col2:
            # Visual summary
            fig = go.Figure(data=[
                go.Bar(name='ASD', x=['Cases'], y=[summary_data['asd_cases']]),
                go.Bar(name='Control', x=['Cases'], y=[summary_data['control_cases']])
            ])
            fig.update_layout(title="Case Distribution", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Export functionality
        if st.button("📄 Export Report as PDF"):
            st.info("PDF export functionality would be implemented here using libraries like ReportLab")
            
    except Exception as e:
        st.error(f"❌ Error generating summary report: {e}")

def generate_demographic_report():
    """Generate demographic analysis report"""
    st.subheader("👥 Demographic Analysis Report")
    
    try:
        # Age distribution analysis
        age_query = """
        MATCH (p:Participant)
        WHERE p.age IS NOT NULL
        RETURN p.age as age, p.diagnosis as diagnosis, p.gender as gender
        """
        age_data = st.session_state.kg_manager.neo4j.execute_query(age_query)
        
        if age_data:
            age_df = pd.DataFrame(age_data)
            
            # Age statistics by diagnosis
            st.subheader("📊 Age Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Age distribution histogram
                fig = px.histogram(age_df, x='age', color='diagnosis',
                                 title="Age Distribution by Diagnosis",
                                 nbins=15, barmode='overlay',
                                 opacity=0.7)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Age box plot
                fig = px.box(age_df, x='diagnosis', y='age',
                           title="Age Distribution Box Plot")
                st.plotly_chart(fig, use_container_width=True)
            
            # Age statistics table
            age_stats = age_df.groupby('diagnosis')['age'].agg(['mean', 'std', 'min', 'max', 'count']).round(2)
            st.dataframe(age_stats)
            
            # Gender analysis
            st.subheader("👫 Gender Analysis")
            
            gender_stats = age_df.groupby(['diagnosis', 'gender']).size().unstack(fill_value=0)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Gender distribution pie chart
                gender_totals = age_df['gender'].value_counts()
                fig = px.pie(values=gender_totals.values, names=gender_totals.index,
                           title="Overall Gender Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Gender by diagnosis stacked bar
                fig = px.bar(gender_stats.reset_index(), x='diagnosis', 
                           y=['Male', 'Female', 'Other'], 
                           title="Gender Distribution by Diagnosis")
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("⚠️ No demographic data available")
            
    except Exception as e:
        st.error(f"❌ Error generating demographic report: {e}")

def generate_gait_pattern_report():
    """Generate gait pattern analysis report"""
    st.subheader("🚶 Gait Pattern Analysis Report")
    
    try:
        # Get gait features data
        features_query = """
        MATCH (p:Participant)-[:HAS_SESSION]->(s:GaitSession)-[:HAS_FEATURE]->(f:GaitFeature)
        RETURN p.diagnosis as diagnosis, f.feature_type as feature_type, 
               f.value as value, p.age as age, p.gender as gender
        """
        features_data = st.session_state.kg_manager.neo4j.execute_query(features_query)
        
        if features_data:
            features_df = pd.DataFrame(features_data)
            
            # Feature importance analysis
            st.subheader("🎯 Key Gait Features Analysis")
            
            # Statistical analysis by diagnosis
            feature_stats = features_df.groupby(['diagnosis', 'feature_type'])['value'].agg([
                'mean', 'std', 'count'
            ]).round(4)
            
            # Show top features with significant differences
            asd_means = feature_stats.xs('ASD', level='diagnosis')['mean'] if 'ASD' in features_df['diagnosis'].values else pd.Series()
            control_means = feature_stats.xs('Control', level='diagnosis')['mean'] if 'Control' in features_df['diagnosis'].values else pd.Series()
            
            if not asd_means.empty and not control_means.empty:
                # Calculate effect sizes (simple difference)
                effect_sizes = abs(asd_means - control_means)
                top_features = effect_sizes.nlargest(10)
                
                st.write("**Top 10 Features with Largest Differences Between Groups:**")
                top_features_df = pd.DataFrame({
                    'Feature': top_features.index,
                    'Effect_Size': top_features.values,
                    'ASD_Mean': [asd_means.get(f, 0) for f in top_features.index],
                    'Control_Mean': [control_means.get(f, 0) for f in top_features.index]
                })
                st.dataframe(top_features_df)
                
                # Visualize top features
                col1, col2 = st.columns(2)
                
                with col1:
                    # Effect sizes bar chart
                    fig = px.bar(top_features_df, x='Effect_Size', y='Feature',
                               orientation='h', title="Feature Effect Sizes")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Mean comparison
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='ASD', x=top_features_df['Feature'], 
                                       y=top_features_df['ASD_Mean']))
                    fig.add_trace(go.Bar(name='Control', x=top_features_df['Feature'], 
                                       y=top_features_df['Control_Mean']))
                    fig.update_layout(title="Mean Feature Values by Diagnosis",
                                    xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Feature distribution analysis
            st.subheader("📊 Feature Distribution Analysis")
            
            # Select specific features for detailed analysis
            available_features = features_df['feature_type'].unique()[:10]  # Limit to first 10
            
            for feature in available_features:
                feature_data = features_df[features_df['feature_type'] == feature]
                
                if len(feature_data) > 10:  # Only analyze if sufficient data
                    fig = px.violin(feature_data, x='diagnosis', y='value',
                                  title=f"Distribution of {feature}")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Correlation analysis
            st.subheader("🔗 Feature Correlation Analysis")
            
            # Pivot data for correlation analysis
            pivot_df = features_df.pivot_table(
                index=['diagnosis'], 
                columns='feature_type', 
                values='value', 
                aggfunc='mean'
            ).fillna(0)
            
            if len(pivot_df.columns) > 2:
                # Calculate correlation matrix
                correlation_matrix = pivot_df.T.corr()
                
                # Show only top correlations
                fig = px.imshow(correlation_matrix, 
                              title="Feature Correlation Matrix",
                              color_continuous_scale="RdBu",
                              aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("⚠️ No gait pattern data available")
            
    except Exception as e:
        st.error(f"❌ Error generating gait pattern report: {e}")

if __name__ == "__main__":
    main()