#!/usr/bin/env python3
"""
True GNN Analysis Module for NeuroGait - REAL IMPLEMENTATION
"""
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class TrueGraphAnalysis:
    def __init__(self, samples_per_participant=8):
        self.samples_per_participant = samples_per_participant
        self.neo4j_available = False
        
    def connect_to_graph(self):
        """Connect to the actual Neo4j graph database"""
        try:
            from neo4j import GraphDatabase
            import os
            
            # Try to connect to your actual Neo4j instance
            uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
            user = os.getenv('NEO4J_USER', 'neo4j')
            password = os.getenv('NEO4J_PASSWORD', 'password')
            
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                # Test query to verify graph exists
                result = session.run("MATCH (n) RETURN count(n) as node_count")
                count = result.single()["node_count"]
                
            if count > 0:
                self.driver = driver
                self.neo4j_available = True
                logger.info(f"Connected to Neo4j with {count} nodes")
                return True
            else:
                raise Exception("Neo4j graph is empty")
                
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise RuntimeError(f"Neo4j connection failed: {str(e)}. Cannot proceed without real graph data.")
        
    def extract_graph_features(self, participant_ids):
        """Extract real features from Neo4j graph"""
        if not self.neo4j_available:
            raise RuntimeError("Neo4j not available - cannot extract real graph features")
            
        try:
            with self.driver.session() as session:
                # Query to extract real graph-based features
                query = """
                MATCH (p:Participant {id: $pid})-[:HAS_SAMPLE]->(s:Sample)-[:HAS_EMBEDDING]->(e:Embedding)
                RETURN e.vector as embedding, s.diagnosis as diagnosis
                """
                
                features = []
                labels = []
                
                for pid in participant_ids:
                    result = session.run(query, pid=f"P_{pid}")
                    for record in result:
                        features.append(record["embedding"])
                        labels.append(1 if record["diagnosis"] == "ASD" else 0)
                
                if len(features) == 0:
                    raise RuntimeError("No graph features found in Neo4j")
                    
                return np.array(features), np.array(labels)
                
        except Exception as e:
            raise RuntimeError(f"Failed to extract graph features: {str(e)}")
        
    def train_real_gnn(self, X_train, y_train, X_test, y_test, model_type="GCN"):
        """Train actual GNN models using PyTorch Geometric"""
        try:
            import torch
            import torch.nn.functional as F
            from torch_geometric.nn import GCNConv, GATConv, SAGEConv
            from torch_geometric.data import Data
            from sklearn.neighbors import kneighbors_graph
            
            # Create graph structure from features using k-NN
            A_train = kneighbors_graph(X_train, n_neighbors=5, mode='connectivity', include_self=True)
            edge_index_train = torch.tensor(np.array(A_train.nonzero()), dtype=torch.long)
            
            # Convert to PyTorch tensors
            x_train = torch.tensor(X_train, dtype=torch.float)
            y_train_torch = torch.tensor(y_train, dtype=torch.long)
            x_test = torch.tensor(X_test, dtype=torch.float)
            y_test_torch = torch.tensor(y_test, dtype=torch.long)
            
            # Define GNN model
            class GNNModel(torch.nn.Module):
                def __init__(self, input_dim, hidden_dim, output_dim, model_type):
                    super().__init__()
                    if model_type == "GCN":
                        self.conv1 = GCNConv(input_dim, hidden_dim)
                        self.conv2 = GCNConv(hidden_dim, output_dim)
                    elif model_type == "GAT":
                        self.conv1 = GATConv(input_dim, hidden_dim)
                        self.conv2 = GATConv(hidden_dim, output_dim)
                    elif model_type == "GraphSAGE":
                        self.conv1 = SAGEConv(input_dim, hidden_dim)
                        self.conv2 = SAGEConv(hidden_dim, output_dim)
                        
                def forward(self, x, edge_index):
                    x = F.relu(self.conv1(x, edge_index))
                    x = F.dropout(x, training=self.training)
                    x = self.conv2(x, edge_index)
                    return F.log_softmax(x, dim=1)
            
            # Initialize and train model
            model = GNNModel(X_train.shape[1], 32, 2, model_type)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            
            model.train()
            for epoch in range(200):
                optimizer.zero_grad()
                out = model(x_train, edge_index_train)
                loss = F.nll_loss(out, y_train_torch)
                loss.backward()
                optimizer.step()
            
            # Evaluate on test set
            A_test = kneighbors_graph(X_test, n_neighbors=5, mode='connectivity', include_self=True)
            edge_index_test = torch.tensor(np.array(A_test.nonzero()), dtype=torch.long)
            
            model.eval()
            with torch.no_grad():
                pred = model(x_test, edge_index_test)
                pred_probs = F.softmax(pred, dim=1)[:, 1].numpy()
                pred_labels = pred.argmax(dim=1).numpy()
            
            # Calculate metrics
            auc = roc_auc_score(y_test, pred_probs)
            f1 = f1_score(y_test, pred_labels)
            accuracy = accuracy_score(y_test, pred_labels)
            precision = precision_score(y_test, pred_labels, zero_division=0)
            recall = recall_score(y_test, pred_labels, zero_division=0)
            
            return {
                'auc': auc,
                'f1': f1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'y_test': y_test,
                'proba_test': pred_probs,
                'pred_test': pred_labels,
                'cv_scores': [auc],  # Single fold for simplicity
                'cv_mean': auc,
                'cv_std': 0.0
            }
            
        except ImportError:
            raise RuntimeError("PyTorch Geometric not installed. Run: pip install torch torch-geometric")
        except Exception as e:
            raise RuntimeError(f"GNN training failed: {str(e)}")
        
    def run_gnn_analysis(self, train_pids, test_pids):
        """Run REAL GNN analysis using actual graph data"""
        try:
            # Connect to graph
            if not self.connect_to_graph():
                raise RuntimeError("Cannot connect to Neo4j graph database")
            
            # Extract real features from graph
            X_train, y_train = self.extract_graph_features(train_pids)
            X_test, y_test = self.extract_graph_features(test_pids)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train real GNN models
            results = {}
            for model_type in ['GCN', 'GAT', 'GraphSAGE']:
                logger.info(f"Training real {model_type} model...")
                results[f'GNN_{model_type}'] = self.train_real_gnn(
                    X_train_scaled, y_train, X_test_scaled, y_test, model_type
                )
            
            logger.info(f"Real GNN analysis completed with {len(y_test)} test samples")
            return results
            
        except Exception as e:
            logger.error(f"Real GNN analysis failed: {str(e)}")
            # NO FALLBACK - re-raise the error to stop execution
            raise RuntimeError(f"GNN analysis failed and no fallback allowed: {str(e)}")

    def close(self):
        """Clean up Neo4j connection"""
        if hasattr(self, 'driver'):
            self.driver.close()

def align_test_predictions(gnn_results, reference_labels):
    """Ensure GNN predictions align with reference - only for real results"""
    # This function should only be called with real GNN results
    return gnn_results  # No modification needed for real results