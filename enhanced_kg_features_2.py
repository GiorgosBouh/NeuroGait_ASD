#!/usr/bin/env python3
"""
Enhanced feature engineering for NeuroGait analysis
"""

import numpy as np
import pandas as pd

class EnhancedKGFeatureBuilder:
    def create_enhanced_kg_features(self, df, base_features):
        """Create enhanced features using domain knowledge"""
        X_enhanced = df[base_features].copy()
        
        # Add some basic enhanced features
        for side in ['L', 'R']:
            if f'mean HESH{side}' in base_features and f'mean SPEL{side}' in base_features:
                X_enhanced[f'HESH_SPEL_ratio_{side}'] = (
                    df[f'mean HESH{side}'] / df[f'mean SPEL{side}']
                )
        
        # Add bilateral symmetry features
        for feature in base_features:
            if feature.endswith('L') and feature.replace('L', 'R') in base_features:
                base_name = feature[:-1]
                left_col = f'{base_name}L'
                right_col = f'{base_name}R'
                X_enhanced[f'{base_name}_symmetry'] = (
                    df[left_col] - df[right_col]
                ).abs()
        
        return X_enhanced.fillna(0).values, list(X_enhanced.columns)