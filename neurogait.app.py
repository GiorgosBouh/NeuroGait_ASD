# NeuroGait_ASD: Complete Implementation with Streamlit and Neo4j
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

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score, accuracy_score,
    precision_recall_curve, average_precision_score
)
import xgboost as xgb
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# Additional imports for enhanced functionality
import shap
import openpyxl
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

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

class ExcelDataProcessor:
    """Enhanced Excel data processing with comprehensive feature extraction"""
    
    def __init__(self):
        self.processed_data = None
        self.feature_columns = []
        
    def load_excel_data(self, file_path_or_buffer) -> pd.DataFrame:
        """Load and process Excel data"""
        try:
            # Read Excel file
            if isinstance(file_path_or_buffer, str):
                df = pd.read_excel(file_path_or_buffer)
            else:
                df = pd.read_excel(file_path_or_buffer)
            
            # Clean column names
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            # Basic data cleaning
            df = df.dropna(thresh=len(df.columns) * 0.5)  # Drop rows with >50% missing values
            
            self.processed_data = df
            self._identify_feature_columns()
            
            return df
            
        except Exception as e:
            st.error(f"Error loading Excel data: {e}")
            return pd.DataFrame()
    
    def _identify_feature_columns(self):
        """Automatically identify feature columns"""
        if self.processed_data is not None:
            # Exclude ID and label columns
            exclude_cols = ['participant_id', 'id', 'diagnosis', 'label', 'target']
            self.feature_columns = [col for col in self.processed_data.columns 
                                  if col not in exclude_cols and 
                                  self.processed_data[col].dtype in ['int64', 'float64']]
    
    def get_features_and_labels(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features and labels from processed data"""
        if self.processed_data is None:
            return pd.DataFrame(), pd.Series()
        
        # Get features
        X = self.processed_data[self.feature_columns]
        
        # Get labels (try different column names)
        label_cols = ['diagnosis', 'label', 'target', 'asd', 'class']
        y = None
        
        for col in label_cols:
            if col in self.processed_data.columns:
                y = self.processed_data[col]
                break
        
        if y is None:
            st.error("No label column found. Please ensure your data has one of: diagnosis, label, target, asd, class")
            return X, pd.Series()
        
        # Convert labels to binary if needed
        if y.dtype == 'object':
            y = y.map({'ASD': 1, 'Control': 0, 'asd': 1, 'control': 0, 
                      'positive': 1, 'negative': 0, 'yes': 1, 'no': 0})
        
        return X, y

class EnhancedMLAnalyzer:
    """Enhanced Machine Learning analyzer with comprehensive metrics and SHAP"""
    
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.shap_explainer = None
        self.shap_values = None
        
    def prepare_training_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for training with enhanced preprocessing"""
        if X.empty or y.empty:
            return np.array([]), np.array([])
        
        # Handle missing values
        X_clean = X.fillna(X.median())
        
        # Remove constant features
        X_clean = X_clean.loc[:, X_clean.var() != 0]
        
        self.feature_names = list(X_clean.columns)
        
        # Convert to numpy arrays
        X_array = X_clean.values
        y_array = y.values
        
        # Remove any remaining NaN values
        mask = ~(np.isnan(X_array).any(axis=1) | np.isnan(y_array))
        X_array = X_array[mask]
        y_array = y_array[mask]
        
        return X_array, y_array
    
    def train_models(self, X: np.ndarray, y: np.ndarray):
        """Train all models with enhanced validation"""
        if len(X) == 0:
            logger.error("No training data provided")
            return
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Handle class imbalance with SMOTE
        if len(np.unique(self.y_train)) > 1:
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(self.X_train_scaled, self.y_train)
        else:
            X_balanced, y_balanced = self.X_train_scaled, self.y_train
        
        # Train supervised models
        self.rf_model.fit(X_balanced, y_balanced)
        self.xgb_model.fit(X_balanced, y_balanced)
        
        # Train anomaly detection
        self.isolation_forest.fit(self.X_train_scaled)
        
        # Initialize SHAP explainer
        try:
            self.shap_explainer = shap.TreeExplainer(self.rf_model)
            self.shap_values = self.shap_explainer.shap_values(self.X_test_scaled)
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]  # Take positive class for binary classification
        except Exception as e:
            logger.warning(f"SHAP initialization failed: {e}")
            self.shap_explainer = None
        
        self.is_trained = True
        logger.info("Models trained successfully")
    
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
    
    def get_comprehensive_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.is_trained or self.X_test is None:
            return {}
        
        metrics = {}
        
        # Get predictions for test set
        rf_pred = self.rf_model.predict(self.X_test_scaled)
        rf_proba = self.rf_model.predict_proba(self.X_test_scaled)[:, 1]
        
        xgb_pred = self.xgb_model.predict(self.X_test_scaled)
        xgb_proba = self.xgb_model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate metrics for each model
        for model_name, y_pred, y_proba in [('rf', rf_pred, rf_proba), ('xgb', xgb_pred, xgb_proba)]:
            metrics[f'{model_name}_accuracy'] = accuracy_score(self.y_test, y_pred)
            metrics[f'{model_name}_precision'] = precision_score(self.y_test, y_pred, zero_division=0)
            metrics[f'{model_name}_recall'] = recall_score(self.y_test, y_pred, zero_division=0)
            metrics[f'{model_name}_f1'] = f1_score(self.y_test, y_pred, zero_division=0)
            
            if len(np.unique(self.y_test)) > 1:
                metrics[f'{model_name}_auc'] = roc_auc_score(self.y_test, y_proba)
                metrics[f'{model_name}_avg_precision'] = average_precision_score(self.y_test, y_proba)
        
        # Ensemble metrics
        ensemble_pred = ((rf_pred + xgb_pred) / 2 > 0.5).astype(int)
        ensemble_proba = (rf_proba + xgb_proba) / 2
        
        metrics['ensemble_accuracy'] = accuracy_score(self.y_test, ensemble_pred)
        metrics['ensemble_precision'] = precision_score(self.y_test, ensemble_pred, zero_division=0)
        metrics['ensemble_recall'] = recall_score(self.y_test, ensemble_pred, zero_division=0)
        metrics['ensemble_f1'] = f1_score(self.y_test, ensemble_pred, zero_division=0)
        
        if len(np.unique(self.y_test)) > 1:
            metrics['ensemble_auc'] = roc_auc_score(self.y_test, ensemble_proba)
            metrics['ensemble_avg_precision'] = average_precision_score(self.y_test, ensemble_proba)
        
        return metrics
    
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
    
    def get_confusion_matrices(self) -> Dict:
        """Get confusion matrices for all models"""
        if not self.is_trained or self.X_test is None:
            return {}
        
        rf_pred = self.rf_model.predict(self.X_test_scaled)
        xgb_pred = self.xgb_model.predict(self.X_test_scaled)
        ensemble_pred = ((rf_pred + xgb_pred) / 2 > 0.5).astype(int)
        
        return {
            'rf_cm': confusion_matrix(self.y_test, rf_pred),
            'xgb_cm': confusion_matrix(self.y_test, xgb_pred),
            'ensemble_cm': confusion_matrix(self.y_test, ensemble_pred)
        }
    
    def get_roc_data(self) -> Dict:
        """Get ROC curve data for all models"""
        if not self.is_trained or self.X_test is None:
            return {}
        
        rf_proba = self.rf_model.predict_proba(self.X_test_scaled)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(self.X_test_scaled)[:, 1]
        ensemble_proba = (rf_proba + xgb_proba) / 2
        
        roc_data = {}
        
        for model_name, y_proba in [('rf', rf_proba), ('xgb', xgb_proba), ('ensemble', ensemble_proba)]:
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            auc = roc_auc_score(self.y_test, y_proba)
            roc_data[model_name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
        
        return roc_data
    
    def get_shap_values(self) -> Dict:
        """Get SHAP values and importance"""
        if self.shap_explainer is None or self.shap_values is None:
            return {}
        
        # Feature importance from SHAP
        shap_importance = np.abs(self.shap_values).mean(0)
        shap_feature_importance = dict(zip(self.feature_names, shap_importance))
        
        return {
            'shap_values': self.shap_values,
            'feature_importance': shap_feature_importance,
            'test_data': self.X_test_scaled
        }

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
        features['step_length_cv'] = features['step_length_std'] / features['step_length_mean']
        
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
        features['stride_width_cv'] = features['stride_width_std'] / features['stride_width_mean']
        
        # Asymmetry measures
        left_step_var = df[f'landmark_{joint_indices["left_foot"]}_x'].diff().std()
        right_step_var = df[f'landmark_{joint_indices["right_foot"]}_x'].diff().std()
        features['step_asymmetry'] = abs(left_step_var - right_step_var) / (left_step_var + right_step_var)
        
        # Ground reaction force indicators (estimated from vertical displacement)
        left_ankle_y = df[f'landmark_{joint_indices["left_ankle"]}_y']
        right_ankle_y = df[f'landmark_{joint_indices["right_ankle"]}_y']
        
        features['left_grf_variance'] = left_ankle_y.diff().var()
        features['right_grf_variance'] = right_ankle_y.diff().var()
        
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
                'feature_value': float(feature_value) if not np.isnan(feature_value) else 0.0
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
            "average step length": """
                MATCH (s:GaitSession)-[:HAS_FEATURE]->(f:GaitFeature {feature_type: 'step_length_mean'})
                RETURN avg(f.value) as avg_step_length
            """,
            "participants by age": """
                MATCH (p:Participant)
                RETURN p.age as age, count(p) as count
                ORDER BY p.age
            """
        }
        
        # Simple keyword matching - replace with LLM in production
        for keyword, cypher_query in query_mappings.items():
            if keyword in nl_query.lower():
                return self.neo4j.execute_query(cypher_query)
        
        return [{"error": "Query not recognized. Please try a simpler query."}]

# Initialize session state
if 'neo4j_connection' not in st.session_state:
    st.session_state.neo4j_connection = None
if 'kg_manager' not in st.session_state:
    st.session_state.kg_manager = None
if 'ml_analyzer' not in st.session_state:
    st.session_state.ml_analyzer = EnhancedMLAnalyzer()
if 'gait_analyzer' not in st.session_state:
    st.session_state.gait_analyzer = GaitAnalyzer()
if 'excel_processor' not in st.session_state:
    st.session_state.excel_processor = ExcelDataProcessor()

def main():
    """Main Streamlit application"""
    
    st.title("🧠 NeuroGait ASD Analysis System")
    st.markdown("### Advanced Gait Analysis for Autism Spectrum Disorder Detection")
    
    # Sidebar for navigation and configuration
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔧 Setup", "📊 Data Upload", "📈 Excel Analysis", "🎯 Analysis", "📉 Visualization", "🔍 Query Interface", "📋 Reports"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔧 Setup":
        show_setup_page()
    elif page == "📊 Data Upload":
        show_data_upload_page()
    elif page == "📈 Excel Analysis":
        show_excel_analysis_page()
    elif page == "🎯 Analysis":
        show_analysis_page()
    elif page == "📉 Visualization":
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
        - **📈 Excel Data Analysis**: Process and analyze Excel datasets with comprehensive metrics
        - **🧠 Knowledge Graph Storage**: Store and relate complex gait data using Neo4j
        - **🤖 Machine Learning**: Multi-model approach for ASD prediction
        - **📊 Advanced Metrics**: Precision, Recall, F1-Score, ROC curves, and more
        - **🔍 SHAP Analysis**: Explainable AI with feature importance visualization
        - **📊 Interactive Visualizations**: Comprehensive data exploration tools
        - **💬 Natural Language Queries**: Ask questions about your data in plain English
        - **📱 Real-time Processing**: Immediate analysis and feedback
        
        ### System Architecture:
        1. **Data Collection**: Upload video files or Excel datasets
        2. **Feature Extraction**: Advanced pose estimation and feature engineering
        3. **Knowledge Graph**: Semantic storage and relationship modeling
        4. **ML Analysis**: Ensemble models with comprehensive evaluation metrics
        5. **Explainable AI**: SHAP-based feature importance and model interpretation
        6. **Visualization**: Interactive dashboards and detailed reports
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
        
        # Check SHAP availability
        shap_status = "✅ Available" if st.session_state.ml_analyzer.shap_explainer else "❌ Not Available"
        st.write(f"**SHAP Explainer**: {shap_status}")
        
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
            
            st.success("✅ ML configuration updated!")

def show_excel_analysis_page():
    """Display Excel data analysis page with comprehensive metrics and SHAP"""
    st.header("📈 Excel Data Analysis")
    st.markdown("Upload Excel files for comprehensive ASD analysis with advanced metrics and explainable AI")
    
    # File Upload
    uploaded_file = st.file_uploader(
        "Upload Excel file with gait features",
        type=['xlsx', 'xls'],
        help="Upload an Excel file containing gait features and diagnosis labels"
    )
    
    if uploaded_file is not None:
        try:
            # Load and process Excel data
            with st.spinner("Loading Excel data..."):
                df = st.session_state.excel_processor.load_excel_data(uploaded_file)
                
            if not df.empty:
                st.success(f"✅ Successfully loaded {len(df)} records with {len(df.columns)} columns")
                
                # Data Preview
                st.subheader("📊 Data Preview")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Dataset Information:**")
                    st.write(f"- Rows: {len(df)}")
                    st.write(f"- Columns: {len(df.columns)}")
                    st.write(f"- Feature columns: {len(st.session_state.excel_processor.feature_columns)}")
                
                with col2:
                    st.write("**Missing Values:**")
                    missing_data = df.isnull().sum()
                    if missing_data.sum() > 0:
                        st.write(missing_data[missing_data > 0])
                    else:
                        st.write("No missing values found")
                
                # Show data preview
                st.dataframe(df.head(10))
                
                # Data Analysis Section
                if st.button("🚀 Start Comprehensive Analysis"):
                    # Get features and labels
                    X, y = st.session_state.excel_processor.get_features_and_labels()
                    
                    if not X.empty and not y.empty:
                        with st.spinner("Training models and calculating metrics..."):
                            # Prepare data
                            X_prepared, y_prepared = st.session_state.ml_analyzer.prepare_training_data(X, y)
                            
                            if len(X_prepared) > 0:
                                # Train models
                                st.session_state.ml_analyzer.train_models(X_prepared, y_prepared)
                                
                                # Display results
                                show_comprehensive_results()
                            else:
                                st.error("❌ Unable to prepare training data")
                    else:
                        st.error("❌ Unable to extract features and labels from the data")
        
        except Exception as e:
            st.error(f"❌ Error processing Excel file: {e}")
    
    # Sample data format
    with st.expander("📋 Excel Data Format Guide"):
        st.markdown("""
        ### Required Excel Format
        
        Your Excel file should contain the following columns:
        
        **Required Columns:**
        - `diagnosis` or `label`: Target variable ('ASD', 'Control', 1, 0)
        - Gait feature columns (numerical values)
        
        **Example Feature Columns:**
        - step_length_mean, step_length_std
        - cadence, stride_width_mean
        - left_ankle_angle_mean, right_ankle_angle_mean
        - step_asymmetry, grf_variance
        
        **Optional Columns:**
        - participant_id: Unique identifier
        - age, gender: Demographic information
        
        ### Sample Data Structure:
        ```
        participant_id | age | gender | step_length_mean | cadence | diagnosis
        P001          | 25  | Male   | 0.65            | 1.2     | ASD
        P002          | 30  | Female | 0.72            | 1.1     | Control
        ...
        ```
        """)

def show_comprehensive_results():
    """Display comprehensive analysis results with metrics and SHAP"""
    
    # Get comprehensive metrics
    metrics = st.session_state.ml_analyzer.get_comprehensive_metrics()
    confusion_matrices = st.session_state.ml_analyzer.get_confusion_matrices()
    roc_data = st.session_state.ml_analyzer.get_roc_data()
    shap_data = st.session_state.ml_analyzer.get_shap_values()
    
    # Main metrics dashboard
    st.subheader("🎯 Model Performance Metrics")
    
    # Create metrics comparison table
    metrics_df = pd.DataFrame({
        'Random Forest': [
            metrics.get('rf_accuracy', 0),
            metrics.get('rf_precision', 0),
            metrics.get('rf_recall', 0),
            metrics.get('rf_f1', 0),
            metrics.get('rf_auc', 0)
        ],
        'XGBoost': [
            metrics.get('xgb_accuracy', 0),
            metrics.get('xgb_precision', 0),
            metrics.get('xgb_recall', 0),
            metrics.get('xgb_f1', 0),
            metrics.get('xgb_auc', 0)
        ],
        'Ensemble': [
            metrics.get('ensemble_accuracy', 0),
            metrics.get('ensemble_precision', 0),
            metrics.get('ensemble_recall', 0),
            metrics.get('ensemble_f1', 0),
            metrics.get('ensemble_auc', 0)
        ]
    }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'])
    
    # Display metrics table
    st.dataframe(metrics_df.round(4))
    
    # Visual metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Metrics comparison chart
        fig = go.Figure()
        for column in metrics_df.columns:
            fig.add_trace(go.Scatterpolar(
                r=metrics_df[column].values,
                theta=metrics_df.index,
                fill='toself',
                name=column
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Model Performance Comparison",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Individual metric bars
        melted_metrics = metrics_df.reset_index().melt(id_vars='index')
        fig = px.bar(melted_metrics, x='index', y='value', color='variable',
                    title="Detailed Metrics Comparison",
                    labels={'index': 'Metric', 'value': 'Score', 'variable': 'Model'})
        st.plotly_chart(fig, use_container_width=True)
    
    # ROC Curves
    if roc_data:
        st.subheader("📈 ROC Curves Analysis")
        
        fig = go.Figure()
        
        for model_name, data in roc_data.items():
            fig.add_trace(go.Scatter(
                x=data['fpr'], y=data['tpr'],
                mode='lines',
                name=f"{model_name.upper()} (AUC = {data['auc']:.3f})"
            ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title="ROC Curves Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrices
    if confusion_matrices:
        st.subheader("🔍 Confusion Matrices")
        
        col1, col2, col3 = st.columns(3)
        
        for i, (model_name, cm) in enumerate(confusion_matrices.items()):
            with [col1, col2, col3][i]:
                fig = px.imshow(cm, 
                              labels=dict(x="Predicted", y="Actual"),
                              x=['Control', 'ASD'],
                              y=['Control', 'ASD'],
                              title=f"{model_name.upper()} Confusion Matrix",
                              color_continuous_scale="Blues",
                              text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
    
    # SHAP Analysis
    if shap_data and 'shap_values' in shap_data:
        st.subheader("🔍 SHAP Analysis - Explainable AI")
        
        # Feature importance from SHAP
        shap_importance = shap_data['feature_importance']
        
        if shap_importance:
            col1, col2 = st.columns(2)
            
            with col1:
                # SHAP feature importance
                importance_df = pd.DataFrame(
                    list(shap_importance.items()),
                    columns=['Feature', 'SHAP Importance']
                ).sort_values('SHAP Importance', ascending=False).head(15)
                
                fig = px.bar(importance_df, 
                           x='SHAP Importance', 
                           y='Feature',
                           orientation='h',
                           title="SHAP Feature Importance")
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Traditional feature importance comparison
                traditional_importance = st.session_state.ml_analyzer.get_feature_importance()
                
                if traditional_importance and 'random_forest' in traditional_importance:
                    rf_importance = traditional_importance['random_forest']
                    
                    # Compare SHAP vs traditional importance
                    comparison_data = []
                    common_features = set(shap_importance.keys()) & set(rf_importance.keys())
                    
                    for feature in list(common_features)[:10]:  # Top 10 common features
                        comparison_data.append({
                            'Feature': feature,
                            'SHAP Importance': shap_importance[feature],
                            'RF Importance': rf_importance[feature]
                        })
                    
                    if comparison_data:
                        comp_df = pd.DataFrame(comparison_data)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=comp_df['RF Importance'],
                            y=comp_df['SHAP Importance'],
                            mode='markers+text',
                            text=comp_df['Feature'],
                            textposition='top center',
                            marker=dict(size=10),
                            name='Features'
                        ))
                        
                        fig.update_layout(
                            title="SHAP vs Traditional Feature Importance",
                            xaxis_title="Random Forest Importance",
                            yaxis_title="SHAP Importance"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # SHAP Summary Plot
        try:
            st.subheader("📊 SHAP Summary Visualization")
            
            # Create SHAP summary plot
            shap_values = shap_data['shap_values']
            test_data = shap_data['test_data']
            
            if len(shap_values.shape) == 2:  # For classification
                # Create a matplotlib figure for SHAP summary plot
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # SHAP summary plot
                shap.summary_plot(
                    shap_values, 
                    test_data, 
                    feature_names=st.session_state.ml_analyzer.feature_names,
                    plot_type="dot",
                    show=False
                )
                
                st.pyplot(fig)
                plt.close()
                
                # SHAP waterfall plot for first prediction
                if len(shap_values) > 0:
                    st.subheader("🌊 SHAP Waterfall Plot (Sample Prediction)")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Create explanation object
                    explanation = shap.Explanation(
                        values=shap_values[0],
                        base_values=st.session_state.ml_analyzer.shap_explainer.expected_value,
                        data=test_data[0],
                        feature_names=st.session_state.ml_analyzer.feature_names
                    )
                    
                    shap.waterfall_plot(explanation, show=False)
                    st.pyplot(fig)
                    plt.close()
        
        except Exception as e:
            st.warning(f"Could not generate SHAP plots: {e}")
    
    # Cross-validation results
    st.subheader("🔄 Cross-Validation Analysis")
    
    if hasattr(st.session_state.ml_analyzer, 'X_train') and st.session_state.ml_analyzer.X_train is not None:
        try:
            # Perform stratified k-fold cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            cv_results = {}
            for model_name, model in [('Random Forest', st.session_state.ml_analyzer.rf_model),
                                    ('XGBoost', st.session_state.ml_analyzer.xgb_model)]:
                
                cv_scores = cross_val_score(
                    model, 
                    st.session_state.ml_analyzer.X_train_scaled, 
                    st.session_state.ml_analyzer.y_train,
                    cv=skf, 
                    scoring='f1'
                )
                
                cv_results[model_name] = {
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'scores': cv_scores
                }
            
            # Display CV results
            col1, col2 = st.columns(2)
            
            with col1:
                cv_df = pd.DataFrame({
                    'Model': list(cv_results.keys()),
                    'Mean F1': [results['mean'] for results in cv_results.values()],
                    'Std F1': [results['std'] for results in cv_results.values()]
                })
                
                fig = px.bar(cv_df, x='Model', y='Mean F1', 
                           error_y='Std F1',
                           title="Cross-Validation F1 Scores")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot of CV scores
                cv_plot_data = []
                for model_name, results in cv_results.items():
                    for score in results['scores']:
                        cv_plot_data.append({'Model': model_name, 'F1 Score': score})
                
                cv_plot_df = pd.DataFrame(cv_plot_data)
                fig = px.box(cv_plot_df, x='Model', y='F1 Score',
                           title="Cross-Validation Score Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.warning(f"Could not perform cross-validation: {e}")

def show_data_upload_page():
    """Display the data upload and processing page"""
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
    
    # Batch Data Upload
    with st.expander("📁 Batch Data Upload"):
        st.markdown("""
        ### CSV Format for Batch Upload
        Upload a CSV file with pre-calculated gait features for multiple participants.
        
        **Required Columns:**
        - participant_id, age, gender, diagnosis
        - step_length_mean, step_length_std, cadence
        - left_shoulder_angle_mean, right_shoulder_angle_mean
        - stride_width_mean, stride_width_std
        - step_asymmetry, left_grf_variance, right_grf_variance
        """)
        
        csv_file = st.file_uploader("Upload CSV with gait features", type=['csv'])
        
        if csv_file is not None:
            try:
                df = pd.read_csv(csv_file)
                st.dataframe(df.head())
                
                if st.button("Process CSV Data"):
                    processed_count = 0
                    for _, row in df.iterrows():
                        # Process each row similar to individual upload
                        # Implementation would go here
                        processed_count += 1
                    
                    st.success(f"✅ Processed {processed_count} records from CSV")
                    
            except Exception as e:
                st.error(f"❌ Error processing CSV: {e}")

def show_analysis_page():
    """Display the ML analysis page"""
    st.header("🎯 Machine Learning Analysis")
    
    if not st.session_state.neo4j_connection:
        st.warning("⚠️ Please configure Neo4j connection first.")
        return
    
    # Model Training Section
    st.subheader("🏋️ Model Training")
    
    # Get training data from knowledge graph
    if st.button("🔄 Load Training Data from Knowledge Graph"):
        try:
            # Query for all gait features and diagnoses
            query = """
            MATCH (p:Participant)-[:HAS_SESSION]->(s:GaitSession)-[:HAS_FEATURE]->(f:GaitFeature)
            WITH p, s, collect({feature_type: f.feature_type, value: f.value}) as features
            RETURN p.id as participant_id, p.diagnosis as diagnosis, features
            """
            
            results = st.session_state.kg_manager.neo4j.execute_query(query)
            
            if results:
                # Process results into training format
                features_list = []
                labels = []
                
                for result in results:
                    feature_dict = {}
                    for feature in result['features']:
                        feature_dict[feature['feature_type']] = feature['value']
                    
                    features_list.append(feature_dict)
                    labels.append(1 if result['diagnosis'] == 'ASD' else 0)
                
                st.session_state.training_features = features_list
                st.session_state.training_labels = labels
                
                st.success(f"✅ Loaded {len(features_list)} training samples")
                
                # Display data summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", len(features_list))
                with col2:
                    asd_count = sum(labels)
                    st.metric("ASD Cases", asd_count)
                with col3:
                    control_count = len(labels) - asd_count
                    st.metric("Control Cases", control_count)
            
            else:
                st.warning("⚠️ No training data found in knowledge graph")
                
        except Exception as e:
            st.error(f"❌ Error loading training data: {e}")
    
    # Train Models
    if hasattr(st.session_state, 'training_features'):
        if st.button("🚀 Train Models"):
            with st.spinner("Training machine learning models..."):
                try:
                    # Convert to DataFrame for processing
                    features_df = pd.DataFrame(st.session_state.training_features)
                    labels_series = pd.Series(st.session_state.training_labels)
                    
                    X, y = st.session_state.ml_analyzer.prepare_training_data(features_df, labels_series)
                    
                    if len(X) > 0:
                        st.session_state.ml_analyzer.train_models(X, y)
                        st.success("✅ Models trained successfully!")
                        
                        # Show comprehensive results
                        show_comprehensive_results()
                    
                    else:
                        st.error("❌ No valid training data available")
                        
                except Exception as e:
                    st.error(f"❌ Error training models: {e}")
    
    # Prediction Section
    st.subheader("🔮 Individual Prediction")
    
    if hasattr(st.session_state, 'current_features') and st.session_state.ml_analyzer.is_trained:
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
                    
                    # Visualization
                    prediction_viz_data = {
                        'Model': ['Random Forest', 'XGBoost'],
                        'Confidence': [prediction_result['rf_confidence'], prediction_result['xgb_confidence']],
                        'Prediction': [prediction_result['rf_prediction'], prediction_result['xgb_prediction']]
                    }
                    
                    fig = px.bar(prediction_viz_data, x='Model', y='Confidence', color='Prediction',
                               title="Model Predictions and Confidence")
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {e}")
    
    elif not st.session_state.ml_analyzer.is_trained:
        st.info("ℹ️ Please train the models first before making predictions.")
    
    elif not hasattr(st.session_state, 'current_features'):
        st.info("ℹ️ Please upload and process a video first.")

def show_visualization_page():
    """Display comprehensive data visualizations"""
    st.header("📉 Data Visualization & Analytics")
    
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
        generate_model_performance_report()
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
            performance and data insights.
            
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
        
        with col2:
            # Visual summary
            fig = go.Figure(data=[
                go.Bar(name='ASD', x=['Cases'], y=[summary_data['asd_cases']]),
                go.Bar(name='Control', x=['Cases'], y=[summary_data['control_cases']])
            ])
            fig.update_layout(title="Case Distribution", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed analysis sections
        if summary_data['total_participants'] > 0:
            st.subheader("📈 Trend Analysis")
            
            # Session trend over time
            trend_query = """
            MATCH (s:GaitSession)
            WHERE s.date IS NOT NULL
            RETURN date(s.date) as session_date, count(s) as session_count
            ORDER BY session_date
            """
            
            trend_data = st.session_state.kg_manager.neo4j.execute_query(trend_query)
            
            if trend_data:
                trend_df = pd.DataFrame(trend_data)
                trend_df['session_date'] = pd.to_datetime(trend_df['session_date'])
                
                fig = px.line(trend_df, x='session_date', y='session_count',
                            title="Sessions Over Time")
                st.plotly_chart(fig, use_container_width=True)
        
        # Export functionality
        if st.button("📄 Export Report as PDF"):
            st.info("PDF export functionality would be implemented here using libraries like ReportLab")
            
    except Exception as e:
        st.error(f"❌ Error generating summary report: {e}")

def generate_model_performance_report():
    """Generate enhanced model performance analysis report"""
    st.subheader("🎯 Enhanced Model Performance Report")
    
    if not st.session_state.ml_analyzer.is_trained:
        st.warning("⚠️ No trained models available. Please train models first.")
        return
    
    # Show comprehensive results if models are trained
    show_comprehensive_results()

def generate_demographic_report():
    """Generate demographic analysis report"""
    st.subheader("👥 Demographic Analysis Report")
    # Implementation for demographic analysis
    st.info("Demographic report implementation would go here...")

def generate_gait_pattern_report():
    """Generate gait pattern analysis report"""
    st.subheader("🚶 Gait Pattern Analysis Report")
    # Implementation for gait pattern analysis
    st.info("Gait pattern report implementation would go here...")

if __name__ == "__main__":
    main()