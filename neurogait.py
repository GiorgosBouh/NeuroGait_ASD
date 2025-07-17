#!/usr/bin/env python3
"""
Complete NeuroGait Analysis - Raw Features + Graph Embeddings + Combined
Based on knowledge graph approach but with proper data leakage prevention
Tests 3 approaches: Raw features, Graph embeddings, Combined
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
import networkx as nx
from node2vec import Node2Vec
import warnings
import logging
import json
from datetime import datetime
import os

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class CompleteNeuroGaitAnalysis:
    def __init__(self):
        self.output_dir = f"complete_neurogait_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Body part mappings from knowledge graph
        self.body_parts = [
            'Head', 'Neck', 'SpineShoulder', 'ShoulderLeft', 'ShoulderRight',
            'ElbowLeft', 'ElbowRight', 'WristLeft', 'WristRight', 
            'ThumbLeft', 'ThumbRight', 'HandLeft', 'HandRight',
            'HandTipLeft', 'HandTipRight', 'SpineMid', 'SpineBase',
            'HipLeft', 'HipRight', 'KneeLeft', 'KneeRight',
            'AnkleLeft', 'AnkleRight', 'FootLeft', 'FootRight'
        ]
        
        # Anatomical connections for graph structure
        self.anatomical_connections = [
            ('Head', 'Neck'), ('Neck', 'SpineShoulder'),
            ('SpineShoulder', 'ShoulderLeft'), ('SpineShoulder', 'ShoulderRight'),
            ('ShoulderLeft', 'ElbowLeft'), ('ShoulderRight', 'ElbowRight'),
            ('ElbowLeft', 'WristLeft'), ('ElbowRight', 'WristRight'),
            ('WristLeft', 'HandLeft'), ('WristRight', 'HandRight'),
            ('WristLeft', 'ThumbLeft'), ('WristRight', 'ThumbRight'),
            ('HandLeft', 'HandTipLeft'), ('HandRight', 'HandTipRight'),
            ('SpineShoulder', 'SpineMid'), ('SpineMid', 'SpineBase'),
            ('SpineBase', 'HipLeft'), ('SpineBase', 'HipRight'),
            ('HipLeft', 'KneeLeft'), ('HipRight', 'KneeRight'),
            ('KneeLeft', 'AnkleLeft'), ('KneeRight', 'AnkleRight'),
            ('AnkleLeft', 'FootLeft'), ('AnkleRight', 'FootRight')
        ]
        
    def convert_to_float(self, value):
        """Convert comma decimal separator to float"""
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value).replace(',', '.'))
    
    def load_mean_features_only(self, csv_path='Final dataset.csv'):
        """Load data keeping only mean features (following knowledge graph approach)"""
        logger.info(f"\n📊 Loading data with mean features only...")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ Dataset not found: {csv_path}")
        
        # Load with European format
        df = pd.read_csv(csv_path, sep=';', decimal=',')
        logger.info(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
        
        # Convert numeric columns manually if needed
        numeric_columns = [col for col in df.columns if col != 'class']
        for col in numeric_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: self.convert_to_float(x) if pd.notna(x) else np.nan)
        
        # Convert target
        if 'class' in df.columns:
            df['class'] = df['class'].map({'A': 1, 'T': 0})
            df = df.rename(columns={'class': 'diagnosis'})
            logger.info("✅ Converted target: 'A'->1 (ASD), 'T'->0 (Typical)")
        
        # MEAN FEATURES ONLY - Following knowledge graph approach
        logger.info("\n🔧 Filtering to mean features only (eliminating redundancy)...")
        
        original_cols = len(df.columns)
        cols_to_keep = ['diagnosis']  # Always keep target
        
        for col in df.columns:
            col_clean = col.strip()
            
            # Keep mean coordinate features
            if col_clean.startswith('mean-') and any(coord in col_clean for coord in ['-x-', '-y-', '-z-']):
                cols_to_keep.append(col)
            
            # Keep mean angle features  
            elif col_clean.startswith('mean ') and len(col_clean.split()) >= 2:
                cols_to_keep.append(col)
            
            # Keep ROM features (no redundancy)
            elif col_clean.startswith('Rom'):
                cols_to_keep.append(col)
            
            # Keep gait parameters (no redundancy)
            elif col_clean in ['MaxStLe', 'MaxStWi', 'StrLe', 'GaCT', 'StaT', 'SwiT', 'Velocity']:
                cols_to_keep.append(col)
            
            # Keep other single features
            elif col_clean in ['HaTiLPos', 'HaTiRPos', 'MaxDBFE', 'MinDBFE', 'Threshold']:
                cols_to_keep.append(col)
        
        # Filter dataset
        df_filtered = df[cols_to_keep]
        
        # Basic cleaning
        df_filtered = df_filtered.dropna(axis=1, how='all')
        
        # Remove constant features
        for col in df_filtered.columns:
            if col != 'diagnosis' and df_filtered[col].nunique() <= 1:
                df_filtered = df_filtered.drop(columns=[col])
        
        logger.info(f"✅ Feature filtering results:")
        logger.info(f"   Original features: {original_cols}")
        logger.info(f"   Mean features kept: {len(df_filtered.columns)-1}")
        logger.info(f"   Redundancy eliminated: {original_cols - len(df_filtered.columns)} features")
        logger.info(f"   Data reduction: {((original_cols - len(df_filtered.columns)) / original_cols * 100):.1f}%")
        logger.info(f"📊 Class distribution: {df_filtered['diagnosis'].value_counts().to_dict()}")
        
        return df_filtered
    
    def normalize_body_part(self, body_part_str):
        """Normalize body part names from dataset to standard names"""
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
    
    def create_raw_feature_pipeline(self, n_features=30):
        """Conservative feature pipeline for raw features"""
        
        class RawFeatureSelector:
            def __init__(self, n_features):
                self.n_features = n_features
                self.scaler = StandardScaler()
                self.feature_selector = SelectKBest(f_classif, k=n_features)
                
            def fit(self, X, y):
                logger.info(f"\n🔧 Raw feature pipeline...")
                logger.info(f"   Input shape: {X.shape}")
                
                # Aggressive correlation removal
                corr_matrix = X.corr().abs()
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                to_drop = [column for column in upper_triangle.columns 
                          if any(upper_triangle[column] > 0.8)]
                
                logger.info(f"   Removing {len(to_drop)} highly correlated features")
                self.corr_features_to_drop = to_drop
                X_decorr = X.drop(columns=to_drop)
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X_decorr)
                
                # Select features
                actual_k = min(self.n_features, X_scaled.shape[1])
                self.feature_selector.set_params(k=actual_k)
                X_selected = self.feature_selector.fit_transform(X_scaled, y)
                
                selected_indices = self.feature_selector.get_support(indices=True)
                self.selected_features_ = X_decorr.columns[selected_indices].tolist()
                
                logger.info(f"   Selected {len(self.selected_features_)} raw features")
                return self
                
            def transform(self, X):
                X_decorr = X.drop(columns=self.corr_features_to_drop, errors='ignore')
                X_scaled = self.scaler.transform(X_decorr)
                X_selected = self.feature_selector.transform(X_scaled)
                return X_selected
                
            def fit_transform(self, X, y):
                return self.fit(X, y).transform(X)
        
        return RawFeatureSelector(n_features)
    
    def create_anatomical_graph(self):
        """Create anatomical graph structure"""
        logger.info("\n🧠 Creating anatomical graph structure...")
        
        G = nx.Graph()
        
        # Add body part nodes
        for body_part in self.body_parts:
            G.add_node(body_part, node_type='body_part')
        
        # Add anatomical connections
        for part1, part2 in self.anatomical_connections:
            if part1 in self.body_parts and part2 in self.body_parts:
                G.add_edge(part1, part2, connection_type='anatomical')
        
        logger.info(f"   Created anatomical graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def create_participant_graph(self, X_train, y_train, similarity_threshold=0.7):
        """Create participant similarity graph based on training data only"""
        logger.info(f"\n🔗 Creating participant similarity graph...")
        
        G = nx.Graph()
        
        # Add participant nodes with features
        for i in range(len(X_train)):
            participant_id = f"P_{i:04d}"
            # Add node with class label and some feature statistics
            feature_stats = {
                'label': int(y_train.iloc[i]),
                'mean_activity': float(X_train.iloc[i].mean()),
                'feature_std': float(X_train.iloc[i].std()),
                'node_type': 'participant'
            }
            G.add_node(participant_id, **feature_stats)
        
        # Add similarity edges using k-NN
        logger.info("   Computing participant similarities...")
        knn = NearestNeighbors(n_neighbors=min(8, len(X_train)//2), metric='cosine')
        knn.fit(X_train)
        
        distances, indices = knn.kneighbors(X_train)
        edge_count = 0
        
        for i, (neighbors, dists) in enumerate(zip(indices, distances)):
            participant_i = f"P_{i:04d}"
            for j, dist in zip(neighbors, dists):
                if i != j and (1 - dist) > similarity_threshold:  # Convert distance to similarity
                    participant_j = f"P_{j:04d}"
                    similarity = 1 - dist
                    G.add_edge(participant_i, participant_j, 
                             weight=similarity, 
                             connection_type='similarity')
                    edge_count += 1
        
        logger.info(f"   Added {edge_count} similarity edges")
        logger.info(f"   Participant graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def create_graph_embeddings(self, X_train, y_train, X_test, embedding_dim=32):
        """Create graph embeddings without data leakage"""
        logger.info(f"\n🧠 Creating graph embeddings (dim={embedding_dim})...")
        
        try:
            # Create participant similarity graph using training data only
            participant_graph = self.create_participant_graph(X_train, y_train)
            
            # If graph has edges, create embeddings
            if participant_graph.number_of_edges() > 0:
                logger.info("   Running Node2Vec on participant graph...")
                
                # Create Node2Vec model
                node2vec = Node2Vec(
                    participant_graph,
                    dimensions=embedding_dim,
                    walk_length=20,
                    num_walks=40,
                    p=1.0,
                    q=1.0,
                    workers=1,
                    quiet=True
                )
                
                # Train embeddings
                model = node2vec.fit(window=5, min_count=1, batch_words=4, epochs=10)
                
                # Get training embeddings
                train_embeddings = np.zeros((len(X_train), embedding_dim))
                for i in range(len(X_train)):
                    participant_id = f"P_{i:04d}"
                    if participant_id in model.wv:
                        train_embeddings[i] = model.wv[participant_id]
                    else:
                        train_embeddings[i] = np.random.normal(0, 0.01, embedding_dim)
                
                # For test set: project using k-NN from training embeddings
                logger.info("   Projecting test embeddings using k-NN...")
                knn = NearestNeighbors(n_neighbors=min(5, len(X_train)), metric='cosine')
                knn.fit(X_train)
                
                test_embeddings = np.zeros((len(X_test), embedding_dim))
                test_distances, test_indices = knn.kneighbors(X_test)
                
                for i, (neighbors, dists) in enumerate(zip(test_indices, test_distances)):
                    # Weight embeddings by inverse distance
                    weights = 1 / (dists + 1e-8)
                    weights = weights / weights.sum()
                    
                    # Weighted average of neighbor embeddings
                    test_embeddings[i] = np.average(train_embeddings[neighbors], axis=0, weights=weights)
                
                logger.info(f"✅ Created embeddings: train {train_embeddings.shape}, test {test_embeddings.shape}")
                
            else:
                logger.warning("   No edges in graph, using random embeddings")
                train_embeddings = np.random.normal(0, 0.01, (len(X_train), embedding_dim))
                test_embeddings = np.random.normal(0, 0.01, (len(X_test), embedding_dim))
            
            return train_embeddings, test_embeddings
            
        except Exception as e:
            logger.error(f"❌ Graph embedding failed: {str(e)}")
            logger.info("   Using random embeddings as fallback")
            train_embeddings = np.random.normal(0, 0.01, (len(X_train), embedding_dim))
            test_embeddings = np.random.normal(0, 0.01, (len(X_test), embedding_dim))
            return train_embeddings, test_embeddings
    
    def train_conservative_model(self, X_train, X_test, y_train, y_test, model_name="Model"):
        """Train conservative XGBoost model"""
        logger.info(f"\n🚀 Training {model_name}...")
        logger.info(f"   Training set: {X_train.shape}")
        logger.info(f"   Test set: {X_test.shape}")
        
        # Very conservative settings
        model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.02,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=3.0,
            reg_lambda=3.0,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Cross-validation on training data only
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        
        logger.info(f"   CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Train final model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'test_auc': roc_auc_score(y_test, y_pred_proba),
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred),
            'test_recall': recall_score(y_test, y_pred),
            'test_f1': f1_score(y_test, y_pred),
        }
        
        logger.info(f"\n📊 {model_name} Results:")
        logger.info(f"   CV AUC:      {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
        logger.info(f"   Test AUC:    {metrics['test_auc']:.4f}")
        logger.info(f"   Accuracy:    {metrics['test_accuracy']:.4f}")
        logger.info(f"   Precision:   {metrics['test_precision']:.4f}")
        logger.info(f"   Recall:      {metrics['test_recall']:.4f}")
        logger.info(f"   F1-score:    {metrics['test_f1']:.4f}")
        
        # Performance assessment
        if metrics['test_auc'] > 0.9:
            logger.warning("   ⚠️  Very high performance - possible remaining issues")
        elif metrics['test_auc'] > 0.8:
            logger.info("   ✅ Good performance")
        elif metrics['test_auc'] > 0.7:
            logger.info("   ✅ Realistic performance")
        else:
            logger.info("   ℹ️  Lower performance - may be more realistic")
        
        return metrics, model
    
    def run_complete_analysis(self):
        """Run complete analysis: Raw + Graph + Combined"""
        logger.info(f"\n🔍 Starting Complete NeuroGait Analysis - {datetime.now()}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        
        try:
            # 1. Load mean features only
            df = self.load_mean_features_only()
            
            # 2. Split data FIRST (critical for preventing leakage)
            X = df.drop('diagnosis', axis=1)
            y = df['diagnosis']
            
            logger.info(f"\n✂️  Splitting data (80% train, 20% test)...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"   Training: {len(X_train)} samples")
            logger.info(f"   Test: {len(X_test)} samples")
            logger.info(f"   Train class dist: {y_train.value_counts().to_dict()}")
            logger.info(f"   Test class dist: {y_test.value_counts().to_dict()}")
            
            # Results storage
            all_results = {}
            
            # 3. RAW FEATURES ANALYSIS
            logger.info(f"\n{'='*60}")
            logger.info("🔍 ANALYSIS 1: RAW FEATURES ONLY")
            logger.info(f"{'='*60}")
            
            raw_pipeline = self.create_raw_feature_pipeline(n_features=25)
            X_train_raw = raw_pipeline.fit_transform(X_train, y_train)
            X_test_raw = raw_pipeline.transform(X_test)
            
            raw_results, raw_model = self.train_conservative_model(
                X_train_raw, X_test_raw, y_train, y_test, "Raw Features"
            )
            all_results['raw_features'] = raw_results
            
            # 4. GRAPH EMBEDDINGS ANALYSIS
            logger.info(f"\n{'='*60}")
            logger.info("🧠 ANALYSIS 2: GRAPH EMBEDDINGS ONLY")
            logger.info(f"{'='*60}")
            
            train_embeddings, test_embeddings = self.create_graph_embeddings(
                X_train, y_train, X_test, embedding_dim=25
            )
            
            graph_results, graph_model = self.train_conservative_model(
                train_embeddings, test_embeddings, y_train, y_test, "Graph Embeddings"
            )
            all_results['graph_embeddings'] = graph_results
            
            # 5. COMBINED ANALYSIS
            logger.info(f"\n{'='*60}")
            logger.info("🔗 ANALYSIS 3: COMBINED (RAW + GRAPH)")
            logger.info(f"{'='*60}")
            
            # Combine features
            X_train_combined = np.hstack([X_train_raw, train_embeddings])
            X_test_combined = np.hstack([X_test_raw, test_embeddings])
            
            combined_results, combined_model = self.train_conservative_model(
                X_train_combined, X_test_combined, y_train, y_test, "Combined Features"
            )
            all_results['combined'] = combined_results
            
            # 6. Save results
            self.save_complete_results(all_results)
            
            # 7. Print final summary
            self.print_final_summary(all_results)
            
            logger.info(f"\n✅ Complete analysis finished!")
            logger.info(f"📁 Results saved to: {self.output_dir}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def save_complete_results(self, all_results):
        """Save comprehensive results"""
        # Convert numpy types for JSON serialization
        serializable_results = {}
        for approach, metrics in all_results.items():
            serializable_results[approach] = {}
            for key, value in metrics.items():
                if isinstance(value, (np.floating, np.integer)):
                    serializable_results[approach][key] = float(value)
                else:
                    serializable_results[approach][key] = value
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'Complete NeuroGait Analysis (Raw + Graph + Combined)',
            'approach': 'Mean features only (no variance/std)',
            'data_leakage_prevention': 'Strict train/test separation',
            'results': serializable_results
        }
        
        with open(f"{self.output_dir}/complete_analysis_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"💾 Complete results saved to: {self.output_dir}/complete_analysis_report.json")
    
    def print_final_summary(self, all_results):
        """Print comprehensive final summary"""
        logger.info("\n" + "="*70)
        logger.info("🏁 COMPLETE ANALYSIS SUMMARY")
        logger.info("="*70)
        
        # Find best approach
        best_approach = max(all_results.keys(), key=lambda x: all_results[x]['test_auc'])
        best_auc = all_results[best_approach]['test_auc']
        
        logger.info(f"\n🏆 Best Approach: {best_approach.upper().replace('_', ' ')}")
        logger.info(f"   Test AUC: {best_auc:.4f}")
        
        # Comparison table
        logger.info(f"\n📊 Performance Comparison:")
        logger.info(f"{'Approach':<20} {'CV AUC':<12} {'Test AUC':<10} {'Accuracy':<10} {'F1':<10}")
        logger.info("-" * 62)
        
        approach_names = {
            'raw_features': 'Raw Features',
            'graph_embeddings': 'Graph Embeddings', 
            'combined': 'Combined'
        }
        
        for approach, metrics in all_results.items():
            name = approach_names.get(approach, approach)
            cv_auc = f"{metrics['cv_auc_mean']:.3f}±{metrics['cv_auc_std']:.3f}"
            test_auc = f"{metrics['test_auc']:.3f}"
            accuracy = f"{metrics['test_accuracy']:.3f}"
            f1 = f"{metrics['test_f1']:.3f}"
            
            logger.info(f"{name:<20} {cv_auc:<12} {test_auc:<10} {accuracy:<10} {f1:<10}")
        
        # Insights
        logger.info(f"\n💡 Key Insights:")
        
        raw_auc = all_results['raw_features']['test_auc']
        graph_auc = all_results['graph_embeddings']['test_auc']
        combined_auc = all_results['combined']['test_auc']
        
        if combined_auc > max(raw_auc, graph_auc):
            logger.info("   ✅ Combined approach shows best performance")
        elif graph_auc > raw_auc:
            logger.info("   🧠 Graph embeddings outperform raw features")
        else:
            logger.info("   📊 Raw features perform best")
        
        if max(raw_auc, graph_auc, combined_auc) < 0.85:
            logger.info("   ✅ Realistic performance levels achieved")
        else:
            logger.info("   ⚠️  High performance - verify no remaining leakage")
        
        logger.info(f"\n📁 Complete results in: {os.path.abspath(self.output_dir)}")
        logger.info("\n✅ Complete analysis with graph embeddings finished!")


def main():
    """Main function to run complete analysis"""
    try:
        analyzer = CompleteNeuroGaitAnalysis()
        results = analyzer.run_complete_analysis()
        
        print("\n" + "="*50)
        print("🏁 COMPLETE ANALYSIS FINISHED")
        print("="*50)
        
        for approach, metrics in results.items():
            approach_name = approach.replace('_', ' ').title()
            print(f"\n{approach_name}:")
            print(f"  Test AUC: {metrics['test_auc']:.4f}")
            print(f"  Test F1:  {metrics['test_f1']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Main analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()