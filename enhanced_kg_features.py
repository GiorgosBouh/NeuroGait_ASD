#!/usr/bin/env python3
"""
NeuroGait Analysis with True Graph-Based Features
This shows how to integrate the graph-based features with your existing analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging
from enhanced_kg_features import GraphBasedKGFeatureBuilder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GraphEnhancedNeuroGaitAnalysis:
    def __init__(self, samples_per_participant=8):
        self.samples_per_participant = samples_per_participant
        self.graph_builder = GraphBasedKGFeatureBuilder(samples_per_participant)
        
    def load_data(self, filepath="Final dataset.csv"):
        """Load the dataset"""
        logger.info("📊 Loading dataset...")
        
        try:
            df = pd.read_csv(filepath, sep=';', decimal=',', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, sep=';', decimal=',', encoding='latin-1')
        
        # Convert numeric columns
        numeric_cols = [col for col in df.columns if col != 'class']
        for col in numeric_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.').astype(float)
        
        # Add participant info
        df['participant_id'] = df.index // self.samples_per_participant
        df['diagnosis'] = df['class'].map({'A': 'ASD', 'T': 'Typical'})
        
        return df
    
    def compare_approaches(self, df):
        """Compare three approaches: Original, Enhanced (no graph), and Graph-based"""
        
        # Define base features
        base_features = [
            'mean HESHL', 'mean HESHR', 'mean SPELL', 'mean SPELR',
            'mean SHWRL', 'mean SHWRR', 'mean ELHAL', 'mean ELHAR', 
            'mean THHAL', 'mean THHAR', 'mean SPKNL', 'mean SPKNR',
            'mean HIANL', 'mean HIANR', 'mean KNFOL', 'mean KNFOR',
            'GaCT', 'StaT', 'SwiT'
        ]
        
        # Filter available features
        available_features = [f for f in base_features if f in df.columns]
        
        # Get labels
        y = (df['diagnosis'] == 'ASD').astype(int)
        
        # Participant-level split
        participant_info = df.groupby('participant_id')['diagnosis'].first().reset_index()
        train_pids, test_pids = train_test_split(
            participant_info['participant_id'].values,
            test_size=0.2,
            stratify=participant_info['diagnosis'].values,
            random_state=42
        )
        
        train_mask = df['participant_id'].isin(train_pids)
        test_mask = df['participant_id'].isin(test_pids)
        
        results = {}
        
        # 1. Original Features Only
        logger.info("\n🔷 Approach 1: Original Features Only")
        X_original = df[available_features].fillna(0).values
        X_train_orig = X_original[train_mask]
        X_test_orig = X_original[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        # Standardize
        scaler = StandardScaler()
        X_train_orig_scaled = scaler.fit_transform(X_train_orig)
        X_test_orig_scaled = scaler.transform(X_test_orig)
        
        # Train and evaluate
        rf_orig = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_orig.fit(X_train_orig_scaled, y_train)
        
        acc_orig = accuracy_score(y_test, rf_orig.predict(X_test_orig_scaled))
        results['original'] = acc_orig
        logger.info(f"   Accuracy: {acc_orig:.3f}")
        
        # 2. Enhanced Features (Domain Knowledge, No Graph)
        logger.info("\n🔷 Approach 2: Enhanced Features (Domain Knowledge, No Graph)")
        from enhanced_kg_features_2 import EnhancedKGFeatureBuilder
        
        enhancer = EnhancedKGFeatureBuilder()
        X_enhanced, enhanced_names = enhancer.create_enhanced_kg_features(df, available_features)
        
        X_train_enh = X_enhanced[train_mask]
        X_test_enh = X_enhanced[test_mask]
        
        # Standardize
        scaler_enh = StandardScaler()
        X_train_enh_scaled = scaler_enh.fit_transform(X_train_enh)
        X_test_enh_scaled = scaler_enh.transform(X_test_enh)
        
        # Train and evaluate
        rf_enh = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_enh.fit(X_train_enh_scaled, y_train)
        
        acc_enh = accuracy_score(y_test, rf_enh.predict(X_test_enh_scaled))
        results['enhanced_no_graph'] = acc_enh
        logger.info(f"   Accuracy: {acc_enh:.3f}")
        logger.info(f"   Features: {X_enhanced.shape[1]} (added {X_enhanced.shape[1] - len(available_features)})")
        
        # 3. Graph-Based Features
        logger.info("\n🔷 Approach 3: Graph-Based Features")
        
        # Connect to graph
        if not self.graph_builder.connect():
            logger.error("Could not connect to Neo4j!")
            return results
        
        try:
            # Extract graph features for all samples
            logger.info("   Extracting graph features...")
            
            # We need to extract features for train and test separately to avoid leakage
            train_df = df[train_mask]
            test_df = df[test_mask]
            
            # Extract graph features for training data
            graph_features_train, graph_names, _ = self.graph_builder.extract_graph_features(
                participant_ids=train_pids,
                data_split='train'
            )
            
            # Extract graph features for test data
            graph_features_test, _, _ = self.graph_builder.extract_graph_features(
                participant_ids=test_pids,
                data_split='test'
            )
            
            # Combine with original features
            X_train_graph = np.hstack([X_train_orig, graph_features_train])
            X_test_graph = np.hstack([X_test_orig, graph_features_test])
            
            # Standardize
            scaler_graph = StandardScaler()
            X_train_graph_scaled = scaler_graph.fit_transform(X_train_graph)
            X_test_graph_scaled = scaler_graph.transform(X_test_graph)
            
            # Train and evaluate
            rf_graph = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_graph.fit(X_train_graph_scaled, y_train)
            
            acc_graph = accuracy_score(y_test, rf_graph.predict(X_test_graph_scaled))
            results['graph_based'] = acc_graph
            logger.info(f"   Accuracy: {acc_graph:.3f}")
            logger.info(f"   Features: {X_train_graph.shape[1]} (added {len(graph_names)} graph features)")
            
            # Feature importance analysis
            feature_names_all = [f"orig_{f}" for f in available_features] + graph_names
            importances = rf_graph.feature_importances_
            
            # Get top 10 most important features
            top_indices = np.argsort(importances)[-10:][::-1]
            logger.info("\n   Top 10 Most Important Features:")
            for i, idx in enumerate(top_indices, 1):
                logger.info(f"      {i:2d}. {feature_names_all[idx]}: {importances[idx]:.3f}")
            
            # Analyze feature categories
            categories = self.graph_builder.get_feature_importance_categories()
            logger.info("\n   Feature Importance by Category:")
            
            for category, cat_features in categories.items():
                cat_indices = [i for i, name in enumerate(feature_names_all) if name in cat_features]
                if cat_indices:
                    cat_importance = np.mean(importances[cat_indices])
                    logger.info(f"      {category}: {cat_importance:.3f} (avg of {len(cat_indices)} features)")
            
        finally:
            self.graph_builder.close()
        
        # Summary
        logger.info("\n📊 RESULTS SUMMARY:")
        logger.info("   " + "-"*50)
        logger.info(f"   Original Features:          {results['original']:.3f}")
        logger.info(f"   Enhanced (No Graph):        {results['enhanced_no_graph']:.3f}")
        logger.info(f"   Graph-Based Features:       {results.get('graph_based', 'N/A')}")
        logger.info("   " + "-"*50)
        
        if 'graph_based' in results:
            improvement = (results['graph_based'] - results['original']) / results['original'] * 100
            logger.info(f"   Graph improvement over original: {improvement:+.1f}%")
        
        return results


def main():
    logger.info("🧠 NeuroGait Analysis with Graph-Based Features")
    logger.info("="*60)
    
    # Create analyzer
    analyzer = GraphEnhancedNeuroGaitAnalysis()
    
    # Load data
    df = analyzer.load_data()
    
    # Compare approaches
    results = analyzer.compare_approaches(df)
    
    logger.info("\n✅ Analysis Complete!")
    
    # Additional insights
    if 'graph_based' in results and results['graph_based'] > max(results['original'], results['enhanced_no_graph']):
        logger.info("\n🎯 KEY INSIGHT: Graph-based features provide the best performance!")
        logger.info("   This suggests that the relationships captured in the knowledge graph")
        logger.info("   contain valuable information not present in the raw features alone.")
    else:
        logger.info("\n💡 The graph structure may need optimization or the current features")
        logger.info("   are already sufficiently informative for this classification task.")


if __name__ == "__main__":
    main()