# Enhanced Feature Analysis & ML Preparation Module
"""
Additional enhancements for the NeuroGait Knowledge Graph
Focuses on advanced feature analysis and ML preparation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedFeatureAnalyzer:
    """Advanced feature analysis for gait data"""
    
    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph
        self.feature_importance = {}
        self.correlations = {}
        
    def analyze_feature_distributions(self):
        """Analyze feature distributions by class"""
        logger.info("Analyzing feature distributions...")
        
        # Separate features by class
        asd_data = self.kg.data[self.kg.data['diagnosis'] == 'ASD']
        control_data = self.kg.data[self.kg.data['diagnosis'] == 'Control']
        
        significant_features = []
        
        for category, features in self.kg.feature_schema.items():
            if isinstance(features, list) and features:
                logger.info(f"Analyzing {category}: {len(features)} features")
                
                category_significant = []
                for feature in features[:50]:  # Analyze first 50 to avoid overwhelming
                    if feature in self.kg.data.columns:
                        asd_values = asd_data[feature].dropna()
                        control_values = control_data[feature].dropna()
                        
                        if len(asd_values) > 0 and len(control_values) > 0:
                            # Statistical test
                            from scipy.stats import ttest_ind
                            stat, p_value = ttest_ind(asd_values, control_values)
                            
                            if p_value < 0.05:  # Significant difference
                                effect_size = abs(asd_values.mean() - control_values.mean()) / np.sqrt((asd_values.var() + control_values.var()) / 2)
                                category_significant.append({
                                    'feature': feature,
                                    'p_value': p_value,
                                    'effect_size': effect_size,
                                    'asd_mean': asd_values.mean(),
                                    'control_mean': control_values.mean()
                                })
                
                # Sort by effect size
                category_significant.sort(key=lambda x: x['effect_size'], reverse=True)
                significant_features.extend(category_significant[:10])  # Top 10 per category
        
        self.feature_importance['statistical'] = significant_features
        return significant_features
    
    def create_ml_ready_dataset(self, output_path: str = None):
        """Create ML-ready dataset with proper preprocessing"""
        logger.info("Creating ML-ready dataset...")
        
        # Select features for ML (excluding metadata)
        exclude_cols = ['participant_id', 'class', 'diagnosis']
        feature_cols = [col for col in self.kg.data.columns if col not in exclude_cols]
        
        # Handle missing values
        X = self.kg.data[feature_cols].fillna(0)  # or use median/mean
        y = self.kg.data['diagnosis'].map({'ASD': 1, 'Control': 0})
        
        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=min(500, len(feature_cols)))
        X_selected = selector.fit_transform(X, y)
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        
        # Create final dataset
        ml_dataset = pd.DataFrame(X_selected, columns=selected_features)
        ml_dataset['target'] = y
        ml_dataset['participant_id'] = self.kg.data['participant_id']
        
        if output_path:
            ml_dataset.to_csv(output_path, index=False)
            logger.info(f"ML-ready dataset saved to {output_path}")
        
        logger.info(f"ML dataset created: {len(selected_features)} features, {len(ml_dataset)} samples")
        return ml_dataset, selected_features
    
    def analyze_feature_importance_by_bodypart(self):
        """Analyze which body parts are most discriminative"""
        bodypart_importance = {}
        
        for bodypart in self.kg.body_parts.keys():
            bodypart_features = [col for col in self.kg.data.columns if bodypart in col]
            if bodypart_features:
                # Calculate average importance for this body part
                X = self.kg.data[bodypart_features].fillna(0)
                y = self.kg.data['diagnosis'].map({'ASD': 1, 'Control': 0})
                
                selector = SelectKBest(score_func=f_classif, k=min(len(bodypart_features), 50))
                selector.fit(X, y)
                
                avg_score = np.mean(selector.scores_)
                bodypart_importance[bodypart] = {
                    'avg_score': avg_score,
                    'num_features': len(bodypart_features),
                    'top_features': [bodypart_features[i] for i in selector.get_support(indices=True)[:5]]
                }
        
        # Sort by importance
        sorted_bodyparts = sorted(bodypart_importance.items(), 
                                key=lambda x: x[1]['avg_score'], reverse=True)
        
        return sorted_bodyparts

class CrossValidationStrategy:
    """Proper cross-validation strategy for the dataset"""
    
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
    
    def create_folds(self, X, y, participant_ids):
        """Create stratified folds ensuring no participant leakage"""
        # Since each participant appears only once, standard StratifiedKFold is appropriate
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        folds = []
        for train_idx, test_idx in skf.split(X, y):
            folds.append({
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_participants': participant_ids.iloc[train_idx].tolist(),
                'test_participants': participant_ids.iloc[test_idx].tolist()
            })
        
        return folds
    
    def validate_no_leakage(self, folds):
        """Validate that there's no participant leakage between folds"""
        for i, fold in enumerate(folds):
            train_participants = set(fold['train_participants'])
            test_participants = set(fold['test_participants'])
            
            overlap = train_participants.intersection(test_participants)
            if overlap:
                logger.warning(f"Fold {i}: Found participant overlap: {overlap}")
                return False
        
        logger.info("✅ No participant leakage detected across folds")
        return True

# Additional utility functions for the knowledge graph

def add_statistical_relationships(kg_instance):
    """Add statistical relationships between features and outcomes"""
    with kg_instance.driver.session() as session:
        # Create statistical significance relationships
        session.run("""
            MATCH (f:GaitFeature)-[:HAS_FEATURE]-(s:GaitSession)-[:HAS_SESSION]-(p:Participant)
            MATCH (p)-[:CLASSIFIED_AS]->(c:Classification)
            WITH f.feature_type as feature_type, c.label as classification, 
                 avg(f.value) as avg_value, count(f.value) as sample_size
            WHERE sample_size > 10
            CREATE (fs:FeatureStatistic {
                feature_type: feature_type,
                classification: classification,
                mean_value: avg_value,
                sample_size: sample_size,
                calculated_at: datetime()
            })
        """)
        
        logger.info("Statistical relationships added to knowledge graph")

def create_feature_correlation_network(kg_instance, threshold=0.7):
    """Create a network of highly correlated features"""
    
    # Get numeric columns
    numeric_cols = kg_instance.data.select_dtypes(include=[np.number]).columns
    exclude_cols = ['participant_id'] 
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Calculate correlations
    corr_matrix = kg_instance.data[feature_cols].corr()
    
    with kg_instance.driver.session() as session:
        # Add high correlation relationships
        for i, feature1 in enumerate(feature_cols):
            for j, feature2 in enumerate(feature_cols[i+1:], i+1):
                correlation = corr_matrix.loc[feature1, feature2]
                
                if abs(correlation) > threshold:
                    session.run("""
                        MERGE (f1:FeatureNode {name: $feature1})
                        MERGE (f2:FeatureNode {name: $feature2})
                        MERGE (f1)-[r:CORRELATED_WITH]->(f2)
                        SET r.correlation = $correlation,
                            r.strength = CASE 
                                WHEN abs($correlation) > 0.9 THEN 'very_strong'
                                WHEN abs($correlation) > 0.7 THEN 'strong'
                                ELSE 'moderate'
                            END
                    """, feature1=feature1, feature2=feature2, correlation=float(correlation))
        
        logger.info(f"Feature correlation network created (threshold: {threshold})")

# Integration with the main knowledge graph
def enhance_knowledge_graph(kg_instance):
    """Add all enhancements to the knowledge graph"""
    
    # Add advanced feature analysis
    analyzer = AdvancedFeatureAnalyzer(kg_instance)
    significant_features = analyzer.analyze_feature_distributions()
    
    # Create ML-ready dataset
    ml_dataset, selected_features = analyzer.create_ml_ready_dataset("neurogait_ml_ready.csv")
    
    # Analyze body part importance
    bodypart_importance = analyzer.analyze_feature_importance_by_bodypart()
    
    # Add statistical relationships to graph
    add_statistical_relationships(kg_instance)
    
    # Create feature correlation network
    create_feature_correlation_network(kg_instance, threshold=0.8)
    
    logger.info("Knowledge graph enhanced with advanced analytics")
    
    return {
        'significant_features': significant_features,
        'ml_dataset': ml_dataset,
        'selected_features': selected_features,
        'bodypart_importance': bodypart_importance
    }