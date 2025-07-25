#!/usr/bin/env python3
"""
Enhanced Knowledge Graph Feature Engineering for NeuroGait Analysis
TIER 1 UPGRADE: Advanced graph-inspired features that work with classical ML

Key Improvements:
1. Biomechanical domain knowledge integration
2. Bilateral symmetry and coordination indices  
3. Temporal relationship modeling
4. Network-inspired feature interactions
5. Clinical gait pattern recognition

Compatible with existing RealisticAnalysis pipeline!
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)

class EnhancedKGFeatureBuilder:
    def __init__(self, samples_per_participant=8):
        self.samples_per_participant = samples_per_participant
        self.feature_names = []
        self.biomech_knowledge = self._load_biomechanical_knowledge()
        
    def _load_biomechanical_knowledge(self):
        """Load domain knowledge about gait biomechanics"""
        return {
            # Bilateral pairs for symmetry analysis
            'bilateral_pairs': [
                ('mean HESHL', 'mean HESHR'),  # Heel strike
                ('mean SPELR', 'mean SPELL'),  # Heel off (reversed intentionally)
                ('mean SHWRL', 'mean SHWRR'),  # Shank angular velocity
                ('mean ELHAL', 'mean ELHAR'),  # Elbow angle
                ('mean THHAL', 'mean THHAR'),  # Thigh angle
                ('mean SPKNL', 'mean SPKNR'),  # Spine knee
                ('mean HIANL', 'mean HIANR'),  # Hip angle
                ('mean KNFOL', 'mean KNFOR'),  # Knee flexion
            ],
            
            # Kinematic chains (anatomical connections)
            'kinematic_chains': [
                ['mean HIANL', 'mean SPKNL', 'mean KNFOL'],  # Left chain
                ['mean HIANR', 'mean SPKNR', 'mean KNFOR'],  # Right chain
                ['mean ELHAL', 'mean SHWRL', 'mean THHAL'],  # Left upper-lower
                ['mean ELHAR', 'mean SHWRR', 'mean THHAR'],  # Right upper-lower
            ],
            
            # Temporal features
            'temporal_features': ['GaCT', 'StaT', 'SwiT'],
            
            # Movement coordination features
            'coordination_pairs': [
                ('mean ELHAL', 'mean ELHAR'),  # Upper limb coordination
                ('mean THHAL', 'mean THHAR'),  # Thigh coordination
                ('mean SHWRL', 'mean SHWRR'),  # Shank coordination
            ]
        }
    
    def create_enhanced_kg_features(self, df, feature_list):
        """
        Create enhanced KG features using biomechanical domain knowledge
        
        Returns same participant-level split as input for fair comparison
        """
        logger.info("🧠 Creating Enhanced KG Features with Domain Knowledge...")
        
        # Start with original features
        available_features = [f for f in feature_list if f in df.columns]
        X_original = df[available_features].fillna(0).values
        
        logger.info(f"   📊 Input features: {len(available_features)}")
        
        # Create enhanced features
        enhanced_features = []
        feature_names = []
        
        # 1. Original features (baseline)
        enhanced_features.append(X_original)
        feature_names.extend([f"orig_{f}" for f in available_features])
        
        # 2. Bilateral symmetry features
        bilateral_features, bilateral_names = self._create_bilateral_features(df, available_features)
        if bilateral_features.size > 0:
            enhanced_features.append(bilateral_features)
            feature_names.extend(bilateral_names)
        
        # 3. Temporal coordination features
        temporal_features, temporal_names = self._create_temporal_features(df, available_features)
        if temporal_features.size > 0:
            enhanced_features.append(temporal_features)
            feature_names.extend(temporal_names)
        
        # 4. Kinematic chain features
        kinematic_features, kinematic_names = self._create_kinematic_features(df, available_features)
        if kinematic_features.size > 0:
            enhanced_features.append(kinematic_features)
            feature_names.extend(kinematic_names)
        
        # 5. Movement coordination features
        coordination_features, coordination_names = self._create_coordination_features(df, available_features)
        if coordination_features.size > 0:
            enhanced_features.append(coordination_features)
            feature_names.extend(coordination_names)
        
        # 6. Network-inspired features
        network_features, network_names = self._create_network_features(X_original, available_features)
        if network_features.size > 0:
            enhanced_features.append(network_features)
            feature_names.extend(network_names)
        
        # 7. Participant-level aggregation features
        participant_features, participant_names = self._create_participant_features(df, X_original, available_features)
        if participant_features.size > 0:
            enhanced_features.append(participant_features)
            feature_names.extend(participant_names)
        
        # Combine all features
        if len(enhanced_features) > 1:
            X_enhanced = np.hstack(enhanced_features)
        else:
            X_enhanced = enhanced_features[0]
        
        # Store feature names for analysis
        self.feature_names = feature_names
        
        logger.info(f"   ✅ Enhanced features created:")
        logger.info(f"      Original: {X_original.shape[1]} features")
        logger.info(f"      Enhanced: {X_enhanced.shape[1]} features")
        logger.info(f"      Added: {X_enhanced.shape[1] - X_original.shape[1]} new features")
        
        return X_enhanced, feature_names
    
    def _create_bilateral_features(self, df, available_features):
        """Create bilateral symmetry and coordination features"""
        features = []
        names = []
        
        for left_feature, right_feature in self.biomech_knowledge['bilateral_pairs']:
            if left_feature in available_features and right_feature in available_features:
                left_vals = df[left_feature].fillna(0).values
                right_vals = df[right_feature].fillna(0).values
                
                # Symmetry Index (SI) = |Left - Right| / (Left + Right)
                symmetry_index = np.abs(left_vals - right_vals) / (np.abs(left_vals) + np.abs(right_vals) + 1e-8)
                features.append(symmetry_index)
                names.append(f"symmetry_{left_feature.split()[-1]}")
                
                # Bilateral Coordination Index
                correlation = np.corrcoef(left_vals, right_vals)[0, 1]
                if not np.isnan(correlation):
                    coord_index = np.full(len(left_vals), correlation)
                    features.append(coord_index)
                    names.append(f"bilateral_coord_{left_feature.split()[-1]}")
                
                # Phase Difference (simplified)
                phase_diff = np.arctan2(right_vals - np.mean(right_vals), 
                                      left_vals - np.mean(left_vals))
                features.append(phase_diff)
                names.append(f"phase_diff_{left_feature.split()[-1]}")
        
        if features:
            return np.column_stack(features), names
        return np.array([]).reshape(len(df), 0), []
    
    def _create_temporal_features(self, df, available_features):
        """Create temporal relationship features"""
        features = []
        names = []
        
        temporal_cols = [f for f in self.biomech_knowledge['temporal_features'] if f in available_features]
        
        if len(temporal_cols) >= 2:
            for i, col1 in enumerate(temporal_cols):
                for col2 in temporal_cols[i+1:]:
                    vals1 = df[col1].fillna(0).values
                    vals2 = df[col2].fillna(0).values
                    
                    # Temporal ratio
                    ratio = vals1 / (vals2 + 1e-8)
                    features.append(ratio)
                    names.append(f"temp_ratio_{col1}_{col2}")
                    
                    # Temporal coupling strength
                    coupling = vals1 * vals2 / (np.std(vals1) * np.std(vals2) + 1e-8)
                    features.append(coupling)
                    names.append(f"temp_coupling_{col1}_{col2}")
        
        # Gait cycle analysis if available
        if 'GaCT' in available_features and 'StaT' in available_features and 'SwiT' in available_features:
            gact = df['GaCT'].fillna(0).values
            stat = df['StaT'].fillna(0).values
            swit = df['SwiT'].fillna(0).values
            
            # Stance/Swing ratio (clinical measure)
            stance_swing_ratio = stat / (swit + 1e-8)
            features.append(stance_swing_ratio)
            names.append("stance_swing_ratio")
            
            # Gait cycle regularity
            cycle_regularity = 1.0 / (np.std(gact) + 1e-8)
            features.append(np.full(len(gact), cycle_regularity))
            names.append("gait_regularity")
            
            # Double support time (estimated)
            double_support = np.maximum(0, stat + swit - gact)
            features.append(double_support)
            names.append("double_support_time")
        
        if features:
            return np.column_stack(features), names
        return np.array([]).reshape(len(df), 0), []
    
    def _create_kinematic_features(self, df, available_features):
        """Create kinematic chain coordination features"""
        features = []
        names = []
        
        for chain in self.biomech_knowledge['kinematic_chains']:
            chain_features = [f for f in chain if f in available_features]
            
            if len(chain_features) >= 2:
                chain_data = df[chain_features].fillna(0).values
                
                # Chain coordination (variance across chain)
                chain_variance = np.var(chain_data, axis=1)
                features.append(chain_variance)
                names.append(f"chain_coord_{len(chain_features)}joints")
                
                # Proximal-distal gradient
                if len(chain_features) >= 3:
                    proximal_distal = chain_data[:, 0] - chain_data[:, -1]
                    features.append(proximal_distal)
                    names.append(f"prox_dist_gradient_{len(chain_features)}joints")
                
                # Chain smoothness (second derivative approximation)
                if len(chain_features) >= 3:
                    smoothness = np.sum(np.diff(chain_data, n=2, axis=1)**2, axis=1)
                    features.append(smoothness)
                    names.append(f"chain_smoothness_{len(chain_features)}joints")
        
        if features:
            return np.column_stack(features), names
        return np.array([]).reshape(len(df), 0), []
    
    def _create_coordination_features(self, df, available_features):
        """Create movement coordination features"""
        features = []
        names = []
        
        # Upper-lower limb coordination
        upper_features = [f for f in available_features if 'ELH' in f]
        lower_features = [f for f in available_features if any(x in f for x in ['THH', 'SHW', 'KNF'])]
        
        if upper_features and lower_features:
            upper_data = df[upper_features].fillna(0).values
            lower_data = df[lower_features].fillna(0).values
            
            # Upper-lower coordination
            upper_mean = np.mean(upper_data, axis=1)
            lower_mean = np.mean(lower_data, axis=1)
            
            coordination = np.corrcoef(upper_mean, lower_mean)[0, 1]
            if not np.isnan(coordination):
                coord_feature = np.full(len(upper_mean), coordination)
                features.append(coord_feature)
                names.append("upper_lower_coordination")
        
        # Interlimb coordination variability
        for pair_name, (feat1, feat2) in zip(['elbow', 'thigh', 'shank'], 
                                           self.biomech_knowledge['coordination_pairs']):
            if feat1 in available_features and feat2 in available_features:
                vals1 = df[feat1].fillna(0).values
                vals2 = df[feat2].fillna(0).values
                
                # Coordination variability
                coord_var = np.abs(vals1 - vals2) / (np.abs(vals1) + np.abs(vals2) + 1e-8)
                features.append(coord_var)
                names.append(f"coord_var_{pair_name}")
        
        if features:
            return np.column_stack(features), names
        return np.array([]).reshape(len(df), 0), []
    
    def _create_network_features(self, X_original, feature_names):
        """Create network-inspired features from correlation structure"""
        features = []
        names = []
        
        if X_original.shape[1] < 3:
            return np.array([]).reshape(X_original.shape[0], 0), []
        
        # Feature correlation network
        try:
            correlation_matrix = np.corrcoef(X_original.T)
            correlation_matrix = np.nan_to_num(correlation_matrix)
            
            # Network metrics for each sample
            for i in range(X_original.shape[0]):
                sample = X_original[i, :]
                
                # Sample-specific network properties
                sample_network = np.outer(sample, sample)
                
                # Degree centrality proxy
                degree_centrality = np.sum(np.abs(sample_network), axis=1)
                avg_degree = np.mean(degree_centrality)
                features.append([avg_degree])
                
                # Network density proxy
                network_density = np.sum(np.abs(sample_network)) / (sample_network.shape[0]**2)
                features.append([network_density])
                
                # Clustering coefficient proxy
                clustering = np.sum(sample_network**3) / (np.sum(sample_network**2) + 1e-8)
                features.append([clustering])
            
            # Reshape and create names
            if features:
                network_array = np.array(features).T
                network_names = ['avg_degree_centrality', 'network_density', 'clustering_coefficient']
                return network_array, network_names
                
        except Exception as e:
            logger.warning(f"Network features creation failed: {e}")
        
        return np.array([]).reshape(X_original.shape[0], 0), []
    
    def _create_participant_features(self, df, X_original, feature_names):
        """Create participant-level aggregation features"""
        features = []
        names = []
        
        # Participant-level statistics
        participant_ids = df['participant_id'].values
        unique_participants = np.unique(participant_ids)
        
        participant_features = []
        
        for pid in participant_ids:
            participant_mask = participant_ids == pid
            participant_data = X_original[participant_mask]
            
            if len(participant_data) > 1:
                # Intra-participant variability
                intra_var = np.var(participant_data, axis=0)
                avg_intra_var = np.mean(intra_var)
                
                # Intra-participant consistency
                consistency = 1.0 / (np.std(np.std(participant_data, axis=1)) + 1e-8)
                
                # Movement complexity (entropy approximation)
                complexity = -np.sum(np.mean(participant_data, axis=0) * 
                                   np.log(np.abs(np.mean(participant_data, axis=0)) + 1e-8))
                
                participant_features.append([avg_intra_var, consistency, complexity])
            else:
                participant_features.append([0.0, 0.0, 0.0])
        
        if participant_features:
            participant_array = np.array(participant_features)
            participant_names = ['intra_participant_variability', 'movement_consistency', 'movement_complexity']
            return participant_array, participant_names
        
        return np.array([]).reshape(len(df), 0), []
    
    def get_feature_importance_categories(self):
        """Return categorized feature names for analysis"""
        categories = {
            'original': [name for name in self.feature_names if name.startswith('orig_')],
            'bilateral_symmetry': [name for name in self.feature_names if 'symmetry' in name or 'bilateral' in name],
            'temporal': [name for name in self.feature_names if 'temp_' in name or 'gait_' in name or 'stance_' in name],
            'kinematic': [name for name in self.feature_names if 'chain_' in name or 'prox_' in name],
            'coordination': [name for name in self.feature_names if 'coord' in name],
            'network': [name for name in self.feature_names if any(x in name for x in ['degree', 'density', 'clustering'])],
            'participant': [name for name in self.feature_names if 'intra_' in name or 'movement_' in name]
        }
        return categories


def integrate_enhanced_features_with_analysis():
    """
    Integration function to add enhanced KG features to RealisticAnalysis
    """
    
    integration_code = """
    # Add this to RealisticAnalysis class:
    
    def run_enhanced_kg_analysis(self):
        '''Run analysis with enhanced KG features'''
        
        # Phase 1-4: Same as before (data loading, preprocessing, split, feature selection)
        df, all_features = self.load_and_prepare_data()
        df_clean, clean_features = self.conservative_preprocessing(df, all_features)
        train_data, test_data, train_pids, test_pids = self.proper_train_test_split(df_clean)
        X_train, X_test, selected_features = self.conservative_feature_selection(
            train_data, test_data, clean_features
        )
        
        # Phase 5: Prepare data
        y_train = train_data['diagnosis']
        y_test = test_data['diagnosis']
        X_train_scaled, X_test_scaled = self.prepare_data_properly(X_train, X_test)
        
        # Phase 6: Raw features analysis (same as before)
        raw_results = self.train_conservative_models(
            X_train_scaled, X_test_scaled, y_train, y_test, train_pids, "Raw Features"
        )
        
        # Phase 7: Simple KG embeddings (same as before)  
        X_train_kg_simple, X_test_kg_simple = self.create_conservative_kg_embeddings(
            X_train_scaled, X_test_scaled
        )
        simple_kg_results = self.train_conservative_models(
            X_train_kg_simple, X_test_kg_simple, y_train, y_test, train_pids, "Simple KG"
        )
        
        # Phase 8: NEW - Enhanced KG features
        enhanced_builder = EnhancedKGFeatureBuilder()
        
        # Create enhanced features for train set
        X_train_enhanced, feature_names = enhanced_builder.create_enhanced_kg_features(
            train_data, selected_features
        )
        X_test_enhanced, _ = enhanced_builder.create_enhanced_kg_features(
            test_data, selected_features  
        )
        
        # Scale enhanced features (leakage-free)
        scaler_enhanced = StandardScaler()
        X_train_enhanced_scaled = scaler_enhanced.fit_transform(X_train_enhanced)
        X_test_enhanced_scaled = scaler_enhanced.transform(X_test_enhanced)
        
        enhanced_kg_results = self.train_conservative_models(
            X_train_enhanced_scaled, X_test_enhanced_scaled, y_train, y_test, 
            train_pids, "Enhanced KG"
        )
        
        # Phase 9: Three-way comparison
        comparison_results = self.compare_three_approaches(
            raw_results, simple_kg_results, enhanced_kg_results
        )
        
        # Phase 10: Enhanced results reporting
        self.print_enhanced_results(
            raw_results, simple_kg_results, enhanced_kg_results, 
            comparison_results, feature_names, enhanced_builder
        )
        
        return {
            'raw_results': raw_results,
            'simple_kg_results': simple_kg_results, 
            'enhanced_kg_results': enhanced_kg_results,
            'comparison_results': comparison_results,
            'feature_categories': enhanced_builder.get_feature_importance_categories()
        }
    """
    
    return integration_code


if __name__ == "__main__":
    # Example usage
    print("🧠 Enhanced KG Feature Builder")
    print("This creates advanced graph-inspired features for classical ML models")
    print("Compatible with existing RealisticAnalysis pipeline!")
    print("\nFeatures created:")
    print("✅ Bilateral symmetry indices")
    print("✅ Temporal coordination measures")  
    print("✅ Kinematic chain features")
    print("✅ Movement coordination indices")
    print("✅ Network-inspired features")
    print("✅ Participant-level aggregations")
    print("\nIntegration: Add to RealisticAnalysis.run_enhanced_kg_analysis()")
