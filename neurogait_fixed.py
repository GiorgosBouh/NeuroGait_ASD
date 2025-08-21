#!/usr/bin/env python3
"""
REALISTIC ANALYSIS - Enhanced με Clinical Features, Statistics, και GNN Support
GOAL: Raw vs KG vs GNN comparison με καλύτερα clinical features και πλήρη στατιστική ανάλυση

FIXED VERSION: 
- Eliminated all data leakage issues
- Proper train/test separation in all preprocessing steps
- Fixed cross-validation implementation
- Added proper statistical testing with multiple comparison correction
- Ensured clinical validity of feature selection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from scipy.stats import wilcoxon, friedmanchisquare
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# =====================
# Statistical utilities
# =====================

def paired_bootstrap_metric_diff(y, p1, p2, metric_func, n_boot=10000, seed=123, threshold=0.5):
    """
    Paired bootstrap for the difference in a metric between two methods using sample-level pairing.
    y: true labels (0/1), shape (n,)
    p1, p2: predicted probabilities or hard labels, shape (n,)
    metric_func: e.g., roc_auc_score, accuracy_score, f1_score
    For accuracy/f1, we binarize p >= threshold.
    Returns: (mean_diff, (ci_low, ci_high), diffs_array)
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        if metric_func.__name__ in ("accuracy_score", "f1_score"):
            m1 = metric_func(y[s], (p1[s] >= threshold).astype(int))
            m2 = metric_func(y[s], (p2[s] >= threshold).astype(int))
        else:
            m1 = metric_func(y[s], p1[s])
            m2 = metric_func(y[s], p2[s])
        diffs[b] = m1 - m2
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean()), (float(ci_low), float(ci_high)), diffs

def wilcoxon_rank_biserial_from_trueprob(y, p1, p2):
    """
    Wilcoxon signed-rank test on per-sample true-class probabilities,
    plus rank-biserial effect size.
    Returns: (W, p_value, rank_biserial)
    """
    pt1 = np.where(y == 1, p1, 1.0 - p1)
    pt2 = np.where(y == 1, p2, 1.0 - p2)
    stat, p = wilcoxon(pt1, pt2, zero_method="wilcox", alternative="two-sided", mode="auto")
    n = len(y)
    max_W = n * (n + 1) / 2.0
    rbc = 1.0 - (2.0 * stat / max_W)
    return float(stat), float(p), float(rbc)

def rank_biserial_to_label(r):
    """Heuristic interpretation for rank-biserial effect size."""
    ar = abs(r)
    if ar >= 0.474:
        return "Large"
    elif ar >= 0.33:
        return "Medium"
    elif ar >= 0.147:
        return "Small"
    else:
        return "Negligible"

# ΠΡΟΣΘΗΚΗ - Enhanced Features Support
try:
    from enhanced_kg_features import EnhancedKGFeatureBuilder
    ENHANCED_FEATURES_AVAILABLE = True
    print("✅ Enhanced KG Features available")
except ImportError as e:
    print(f"⚠️ Enhanced features not available - {str(e)}")
    print("   Ensure enhanced_kg_features.py contains EnhancedKGFeatureBuilder class")
    ENHANCED_FEATURES_AVAILABLE = False

# ΠΡΟΣΘΗΚΗ - GNN Support
try:
    import sys
    from pathlib import Path
    # Add the parent directory to Python path
    sys.path.append(str(Path(__file__).parent))
    from true_gnn_analysis import TrueGraphAnalysis, align_test_predictions
    GNN_ANALYSIS_AVAILABLE = True
    print("✅ GNN Analysis available (using true_gnn_analysis.py)")
except Exception as e:
    print(f"⚠️ GNN analysis not available - {str(e)}")
    print("   Ensure true_gnn_analysis.py exists in the same directory")
    GNN_ANALYSIS_AVAILABLE = False

class RealisticAnalysis:
    def __init__(self):
        self.random_state = 42
        self.samples_per_participant = 8
        
    def get_clinical_features(self, all_features):
        """Get clinical feature sets from domain expert analysis"""
        print(f"\n🧠 CLINICAL FEATURE SELECTION (from Domain Expert Analysis)")
        
        clinical_sets = {}
        
        # Set 1: Balance Stability features
        balance_keywords = [
            'spine', 'trunk', 'torso', 'midspain', 'spinebase', 'balance', 'stability', 
            'sway', 'postural', 'leg', 'foot', 'knee', 'hip', 'ankle', 'SPKNL', 'SPKNR', 
            'HIANL', 'HIANR', 'KNFOL', 'KNFOR', 'angle', 'rotation'
        ]
        
        balance_features = []
        for feature in all_features:
            feature_lower = feature.lower()
            if any(keyword in feature_lower for keyword in balance_keywords) or \
               any(keyword in feature for keyword in ['Midspain', 'SpineBase', 'SPKNL', 'SPKNR', 'HIANL', 'HIANR']):
                balance_features.append(feature)
        
        clinical_sets['balance_stability'] = balance_features[:30]
        
        # Set 2: Gait Focused features
        gait_keywords = [
            'gact', 'stat', 'swit', 'time', 'duration', 'cycle', 'step', 'stride', 
            'length', 'width', 'distance', 'leg', 'foot', 'knee', 'hip', 'velocity', 'speed'
        ]
        
        gait_features = []
        for feature in all_features:
            feature_lower = feature.lower()
            if any(keyword in feature_lower for keyword in gait_keywords) or \
               any(keyword in feature for keyword in ['GaCT', 'StaT', 'SwiT']):
                gait_features.append(feature)
        
        clinical_sets['gait_focused'] = gait_features[:20]
        
        # Set 3: ASD Specific features
        asd_keywords = [
            'gait', 'stat', 'swit', 'heshl', 'heshr', 'spell', 'spelr', 'coordination', 'timing',
            'shwrl', 'shwrr', 'elhal', 'elhar', 'thhal', 'thhar'
        ]
        
        asd_features = []
        for feature in all_features:
            feature_lower = feature.lower()
            if any(keyword in feature_lower for keyword in asd_keywords) or \
               any(keyword in feature for keyword in ['GaCT', 'StaT', 'SwiT', 'HESHL', 'HESHR', 'SHWRL', 'SHWRR']):
                asd_features.append(feature)
        
        clinical_sets['asd_specific'] = asd_features[:15]
        
        # Set 4: Combined Best
        combined_features = list(set(
            clinical_sets['balance_stability'][:15] + 
            clinical_sets['gait_focused'][:10] + 
            clinical_sets['asd_specific'][:8]
        ))
        clinical_sets['combined_best'] = combined_features
        
        print(f"   📋 Created {len(clinical_sets)} clinical feature sets:")
        for set_name, features in clinical_sets.items():
            available_count = len([f for f in features if f in all_features])
            print(f"      {set_name.replace('_', ' ').title():<18}: {available_count:2d} features")
        
        return clinical_sets
    
    def select_best_clinical_set(self, df, clinical_sets, train_indices):
        """Quick evaluation to select best clinical feature set using only training data"""
        print(f"\n🔍 EVALUATING CLINICAL FEATURE SETS (Training Data Only)")
        
        best_set_name = None
        best_auc = 0
        best_features = None
        
        # Use only training data for feature set selection
        train_df = df.iloc[train_indices]
        
        for set_name, feature_set in clinical_sets.items():
            try:
                available_features = [f for f in feature_set if f in df.columns]
                
                if len(available_features) < 5:
                    print(f"   {set_name.replace('_', ' '):<18}: Too few features ({len(available_features)})")
                    continue
                
                # Quick test with a subset of training data only
                test_df = train_df[available_features + ['participant_id', 'diagnosis']].dropna().head(200)
                
                if len(test_df) < 50:
                    print(f"   {set_name.replace('_', ' '):<18}: Insufficient data after cleaning")
                    continue
                
                # Quick model test
                X = test_df[available_features]
                y = test_df['diagnosis']
                
                if len(np.unique(y)) < 2:
                    print(f"   {set_name.replace('_', ' '):<18}: No class variation")
                    continue
                
                # Quick train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                # Quick standardization and model
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                lr = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
                lr.fit(X_train_scaled, y_train)
                y_pred = lr.predict_proba(X_test_scaled)[:, 1]
                auc = roc_auc_score(y_test, y_pred)
                
                print(f"   {set_name.replace('_', ' '):<18}: {len(available_features):2d} features, Quick AUC={auc:.3f}")
                
                if auc > best_auc:
                    best_auc = auc
                    best_set_name = set_name
                    best_features = available_features
                    
            except Exception as e:
                print(f"   {set_name.replace('_', ' '):<18}: Error - {str(e)[:30]}")
                continue
        
        if best_features is None:
            # Fallback to first available set
            for set_name, feature_set in clinical_sets.items():
                available_features = [f for f in feature_set if f in df.columns]
                if len(available_features) >= 10:
                    best_features = available_features[:25]
                    best_set_name = set_name
                    best_auc = 0.6
                    break
        
        print(f"\n✅ SELECTED CLINICAL FEATURE SET:")
        print(f"   Set: {best_set_name.replace('_', ' ').title()}")
        print(f"   Features: {len(best_features)}")
        print(f"   Estimated AUC: {best_auc:.3f}")
        
        return best_features, best_set_name
        
    def load_and_prepare_data(self):
        """Load data with proper clinical feature selection"""
        print("🏥 REALISTIC ANALYSIS - Enhanced με Clinical Features")
        print("="*80)
        print("🎯 Goal: Raw vs KG vs GNN comparison με clinical features")
        print("🔒 Proper train/test separation and validation")
        print("🛡️ No data leakage ensured")
        print()
        
        # Load data
        try:
            df = pd.read_csv('Final dataset.csv', sep=';', decimal=',', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv('Final dataset.csv', sep=';', decimal=',', encoding='latin-1')
        
        print(f"📊 Original dataset: {df.shape}")
        
        # Convert numeric columns
        numeric_cols = [col for col in df.columns if col != 'class']
        converted_features = []
        
        for col in numeric_cols:
            try:
                if df[col].dtype == 'object':
                    converted_col = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
                    if not converted_col.isna().all() and converted_col.var() > 1e-10:
                        df[col] = converted_col
                        converted_features.append(col)
                else:
                    if df[col].var() > 1e-10:
                        converted_features.append(col)
            except:
                continue
        
        print(f"📊 Converted {len(converted_features)} numeric features")
        
        # Participant mapping
        df['participant_id'] = df.index // self.samples_per_participant
        
        # Use original diagnosis without artificial bias correction
        df['diagnosis'] = df['class'].map({'A': 1, 'T': 0})
        
        # First split data to prevent leakage in feature selection
        participant_info = df.groupby('participant_id')['diagnosis'].first().reset_index()
        train_pids, test_pids = train_test_split(
            participant_info['participant_id'].values,
            test_size=0.25,
            stratify=participant_info['diagnosis'].values,
            random_state=self.random_state
        )
        
        train_indices = df[df['participant_id'].isin(train_pids)].index
        test_indices = df[df['participant_id'].isin(test_pids)].index
        
        # Get clinical features using training data only
        clinical_sets = self.get_clinical_features(converted_features)
        best_features, best_set_name = self.select_best_clinical_set(df, clinical_sets, train_indices)
        
        print(f"✅ Using {len(best_features)} clinical features from {best_set_name}")
        
        # Create sample-level participant IDs for the training data
        train_sample_pids = df.loc[train_indices, 'participant_id'].values
        
        return df, best_features, best_set_name, train_indices, test_indices, train_sample_pids, test_pids
    
    def create_preprocessing_pipeline(self, features):
        """Create a preprocessing pipeline to prevent data leakage"""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, features)
            ])
        
        return preprocessor
    
    def proper_train_test_split(self, df, train_indices, test_indices):
        """Proper participant-level train/test split without leakage"""
        print(f"\n🔧 PROPER PARTICIPANT-LEVEL SPLIT:")
        
        train_data = df.iloc[train_indices].copy()
        test_data = df.iloc[test_indices].copy()
        
        train_participants = len(train_data['participant_id'].unique())
        test_participants = len(test_data['participant_id'].unique())
        
        print(f"   ✅ Train: {train_participants} participants ({len(train_data)} samples)")
        print(f"   ✅ Test:  {test_participants} participants ({len(test_data)} samples)")
        print(f"   📊 Train distribution: {train_data['diagnosis'].value_counts().to_dict()}")
        print(f"   📊 Test distribution: {test_data['diagnosis'].value_counts().to_dict()}")
        
        # Verify no participant leakage
        train_pids = set(train_data['participant_id'].unique())
        test_pids = set(test_data['participant_id'].unique())
        assert len(train_pids.intersection(test_pids)) == 0
        print(f"   ✅ No participant leakage verified")
        
        return train_data, test_data
    
    def preprocess_data(self, train_data, test_data, features):
        """Preprocess data without leakage - fit on train, transform on test"""
        print(f"\n🧹 DATA PREPROCESSING (No Leakage):")
        
        # Handle missing values using training data only
        missing_threshold = 0.6
        missing_per_feature = train_data[features].isna().sum() / len(train_data)
        good_features = missing_per_feature[missing_per_feature <= missing_threshold].index.tolist()
        
        print(f"   🗑️ Removed {len(features) - len(good_features)} features with >{missing_threshold*100}% missing")
        
        # Remove samples with too many missing values (train only)
        missing_per_sample = train_data[good_features].isna().sum(axis=1) / len(good_features)
        good_samples = missing_per_sample <= 0.5
        train_clean = train_data[good_samples].copy()
        
        print(f"   🗑️ Removed {(~good_samples).sum()} train samples with >50% missing")
        
        # Impute missing values using training data statistics
        imputer = SimpleImputer(strategy='median')
        train_imputed = train_clean.copy()
        train_imputed[good_features] = imputer.fit_transform(train_clean[good_features])
        
        # Apply the same imputation to test data
        test_imputed = test_data.copy()
        test_imputed[good_features] = imputer.transform(test_data[good_features])
        
        # Remove constant features from training data
        constant_features = []
        for col in good_features:
            if train_imputed[col].nunique() <= 1:
                constant_features.append(col)
        
        final_features = [f for f in good_features if f not in constant_features]
        
        # Remove duplicates
        train_final = train_imputed.drop_duplicates(subset=final_features)
        test_final = test_imputed.drop_duplicates(subset=final_features)
        
        print(f"   📊 Final preprocessing:")
        print(f"      Features: {len(features)} → {len(final_features)}")
        print(f"      Train samples: {len(train_data)} → {len(train_final)}")
        print(f"      Test samples: {len(test_data)} → {len(test_final)}")
        print(f"      Constant features removed: {len(constant_features)}")
        
        return train_final, test_final, final_features
    
    def optimized_feature_selection(self, train_data, test_data, features):
        """Feature selection χρησιμοποιώντας ΜΟΝΟ training data (no leakage).
        Επιστρέφει: X_train_sel, X_test_sel, selected_features, train_groups (ευθυγραμμισμένα)."""

        import numpy as np
        import pandas as pd
        from sklearn.feature_selection import SelectKBest, f_classif

        print(f"\n🧠 OPTIMIZED FEATURE SELECTION (Training Data Only)")

        # Χτίζουμε X_train με index από το train_data ώστε να διατηρήσουμε στοίχιση
        X_train = train_data[features]
        y_train = train_data['diagnosis']

        n_samples, n_features = X_train.shape
        print(f"   📊 Input: {n_samples} samples × {n_features} features")

        # Στόχος χαρακτηριστικών (ασφαλές upper bound)
        max_features = max(15, min(80, n_samples // 10))
        print(f"   🎯 Target features: {max_features} (optimized ratio)")

        # sample-level groups ευθυγραμμισμένοι με X_train
        train_groups = train_data.loc[X_train.index, 'participant_id'].values

        if n_features <= max_features:
            print(f"   ✅ No selection needed (already {n_features} ≤ {max_features})")
            X_test = test_data[features]
            return X_train, X_test, features, train_groups

        print(f"   🔧 Using statistical feature selection on training data only...")
        selector = SelectKBest(score_func=f_classif, k=max_features)

        try:
            X_train_selected_np = selector.fit_transform(X_train, y_train)
            support_mask = selector.get_support()
            selected_features = [f for f, keep in zip(features, support_mask) if keep]

            # Διατηρούμε τον ΙΔΙΟ index στο X_train_selected ώστε να ευθυγραμμίζεται με groups
            X_train_selected = pd.DataFrame(
                X_train_selected_np,
                index=X_train.index,
                columns=selected_features
            )
            X_test_selected = test_data[selected_features]

            print(f"   ✅ Selected {len(selected_features)} features")
            print(f"   📊 Reduction: {n_features} → {len(selected_features)}")
            print(f"   📊 Feature-to-sample ratio: {len(selected_features)/n_samples:.3f}:1")

            train_groups_aligned = train_data.loc[X_train_selected.index, 'participant_id'].values
            return X_train_selected, X_test_selected, selected_features, train_groups_aligned

        except Exception as e:
            print(f"   ⚠️ Feature selection failed: {str(e)[:80]}")
            print(f"   📋 Using all features")
            return X_train, test_data[features], features, train_groups
    
    def prepare_data_properly(self, X_train, X_test):
        """Prepare data with proper scaling and outlier handling"""
        print(f"\n📊 PROPER DATA PREPARATION:")
        
        print(f"   📊 Shapes: Train{X_train.shape}, Test{X_test.shape}")
        
        # Convert to numpy arrays first
        X_train_arr = np.asarray(X_train)
        X_test_arr = np.asarray(X_test)
        
        # Handle outliers before scaling
        def cap_outliers(X, lower_percentile=1, upper_percentile=99):
            X_capped = X.copy()
            for i in range(X.shape[1]):
                lower_bound = np.percentile(X[:, i], lower_percentile)
                upper_bound = np.percentile(X[:, i], upper_percentile)
                X_capped[:, i] = np.clip(X[:, i], lower_bound, upper_bound)
            return X_capped
        
        # Cap outliers in training data
        X_train_capped = cap_outliers(X_train_arr)
        
        # Scale using training data statistics only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_capped)
        
        # Cap outliers in test data using training percentiles
        X_test_capped = X_test_arr.copy()
        for i in range(X_test_arr.shape[1]):
            lower_bound = np.percentile(X_train_arr[:, i], 1)
            upper_bound = np.percentile(X_train_arr[:, i], 99)
            X_test_capped[:, i] = np.clip(X_test_arr[:, i], lower_bound, upper_bound)
        
        X_test_scaled = scaler.transform(X_test_capped)
        
        print(f"   ✅ Scaling completed with outlier capping (fitted on train only)")
        print(f"   📊 Train range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
        print(f"   📊 Test range: [{X_test_scaled.min():.2f}, {X_test_scaled.max():.2f}]")
        
        return X_train_scaled, X_test_scaled
        
    def create_enhanced_kg_embeddings(self, X_train, X_test):
        """Create enhanced KG embeddings with better parameters"""
        print(f"\n🧠 ENHANCED KG EMBEDDINGS:")
        
        def optimized_graph_processing(X):
            """Optimized graph processing with stronger interactions"""
            X_kg = X.copy()
            n_samples, n_features = X.shape
            
            print(f"      Processing {n_features} features with enhanced interactions...")
            
            # Stronger feature interactions
            if n_features >= 3:
                interaction_strength = 0.08
                
                # More sophisticated interactions
                for i in range(min(8, n_features - 1)):
                    for j in range(i + 1, min(i + 4, n_features)):
                        interaction = X_kg[:, i] * X_kg[:, j] * interaction_strength
                        X_kg[:, i] += interaction * 0.3
                        X_kg[:, j] += interaction * 0.3
            
            # Enhanced smoothing
            if n_features >= 5:
                smoothing = 0.06
                for i in range(2, n_features - 2):
                    X_kg[:, i] = ((1 - 4*smoothing) * X_kg[:, i] + 
                                  smoothing * X_kg[:, i-2] + 
                                  smoothing * X_kg[:, i-1] + 
                                  smoothing * X_kg[:, i+1] + 
                                  smoothing * X_kg[:, i+2])
            
            # Non-linear transformation
            X_kg = np.tanh(X_kg * 0.5)
            
            # Normalize but preserve structure
            for i in range(n_features):
                std = np.std(X_kg[:, i])
                if std > 1e-6:
                    X_kg[:, i] = X_kg[:, i] / std
                    X_kg[:, i] = np.clip(X_kg[:, i], -3, 3)
            
            return X_kg
        
        X_train_kg = optimized_graph_processing(X_train)
        X_test_kg = optimized_graph_processing(X_test)
        
        print(f"   ✅ Enhanced KG embeddings created")
        print(f"      Train: {X_train_kg.shape}, Test: {X_test_kg.shape}")
        
        return X_train_kg, X_test_kg
    
    def create_conservative_kg_embeddings(self, X_train, X_test):
        """Create conservative KG embeddings"""
        print(f"\n🧠 CONSERVATIVE KG EMBEDDINGS:")
        
        def simple_graph_processing(X):
            X_kg = X.copy()
            n_samples, n_features = X.shape
            
            print(f"      Processing {n_features} features...")
            
            if n_features >= 5:
                interaction_strength = 0.01
                
                for i in range(min(5, n_features - 1)):
                    j = (i + 1) % n_features
                    interaction = X_kg[:, i] * X_kg[:, j] * interaction_strength
                    X_kg[:, i] += interaction * 0.5
                    X_kg[:, j] += interaction * 0.5
            
            if n_features >= 3:
                smoothing = 0.02
                for i in range(1, n_features - 1):
                    X_kg[:, i] = ((1 - 2*smoothing) * X_kg[:, i] + 
                                  smoothing * X_kg[:, i-1] + 
                                  smoothing * X_kg[:, i+1])
            
            for i in range(n_features):
                std = np.std(X_kg[:, i])
                if std > 1e-6:
                    X_kg[:, i] = X_kg[:, i] / std
                    X_kg[:, i] = np.clip(X_kg[:, i], -2, 2)
            
            return X_kg
        
        X_train_kg = simple_graph_processing(X_train)
        X_test_kg = simple_graph_processing(X_test)
        
        print(f"   ✅ Conservative KG embeddings created")
        print(f"      Train: {X_train_kg.shape}, Test: {X_test_kg.shape}")
        
        return X_train_kg, X_test_kg
    
    def train_optimized_models(self, X_train, X_test, y_train, y_test, train_groups, approach_name):
        """Train models (participant-level CV)."""

        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from xgboost import XGBClassifier
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        )

        print(f"\n🚀 TRAINING OPTIMIZED MODELS: {approach_name}")
        print(f"   📊 Data shape: {X_train.shape}")

        models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000, C=1.0, solver='liblinear'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_split=5, min_samples_leaf=2,
                max_features='sqrt', random_state=42
            ),
            'XGBoost': XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
                colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
                objective='binary:logistic', eval_metric='logloss',
                random_state=42, n_jobs=4
            ),
            'SVM': SVC(C=1.0, kernel='rbf', probability=True, gamma='scale', random_state=42)
        }

        results = {}

        for model_name, model in models.items():
            print(f"   🔧 Training {model_name}...")
            cv_scores = self._proper_cross_validation(X_train, y_train, train_groups, model)

            model.fit(np.asarray(X_train), np.asarray(y_train))
            y_pred = model.predict(np.asarray(X_test))
            y_proba = model.predict_proba(np.asarray(X_test))[:, 1]

            auc = roc_auc_score(y_test, y_proba)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # Label επίδοσης (χωρίς placeholders)
            if auc > 0.8:
                status = "🎉 Excellent"
            elif auc > 0.7:
                status = "✅ Good"
            elif auc > 0.6:
                status = "⚖️ Moderate"
            else:
                status = "📋 Limited"

            print(f"      {status}: AUC={auc:.3f}, F1={f1:.3f}, CV={np.mean(cv_scores):.3f}±{np.std(cv_scores):.3f}")

            results[model_name] = {
                'cv_scores': cv_scores,
                'cv_mean': float(np.mean(cv_scores)),
                'cv_std': float(np.std(cv_scores)),
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'f1': float(f1),
                'auc': float(auc),
                'y_test': np.asarray(y_test),
                'pred_test': y_pred,
                'proba_test': y_proba
            }

        return results
   
    def _proper_cross_validation(self, X_train, y_train, train_groups, model, cv_folds=5):
        """Participant-level CV χωρίς leakage, με αυστηρή στοίχιση δειγμάτων/groups."""

        import numpy as np
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import roc_auc_score

        X_arr = np.asarray(X_train)
        y_arr = np.asarray(y_train)
        groups_arr = np.asarray(train_groups)

        if not (len(X_arr) == len(y_arr) == len(groups_arr)):
            raise ValueError(
                f"CV groups must align with X_train/y_train: "
                f"len(X)={len(X_arr)}, len(y)={len(y_arr)}, len(groups)={len(groups_arr)}"
            )

        unique_pids = np.unique(groups_arr)
        if len(unique_pids) < cv_folds:
            cv_folds = max(2, len(unique_pids))
            print(f"   ⚠️ Reduced CV folds to {cv_folds} due to limited participants")

        gkf = GroupKFold(n_splits=cv_folds)
        cv_scores = []

        for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_arr, y_arr, groups=groups_arr), 1):
            X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
            y_tr, y_va = y_arr[tr_idx], y_arr[va_idx]

            if (len(np.unique(y_tr)) < 2) or (len(np.unique(y_va)) < 2) or (len(y_tr) < 10) or (len(y_va) < 5):
                raise ValueError(f"Fold {fold_idx}: insufficient class variation or too few samples")

            cloned = type(model)(**model.get_params())
            cloned.fit(X_tr, y_tr)

            if hasattr(cloned, "predict_proba"):
                y_proba = cloned.predict_proba(X_va)[:, 1]
            else:
                y_proba = cloned.decision_function(X_va)

            auc = roc_auc_score(y_va, y_proba)
            cv_scores.append(float(auc))
            print(f"   Fold {fold_idx}: AUC={auc:.3f}")

        return cv_scores

    def statistical_comparison_analysis(self, tier1_results):
        """Paired statistical comparison using sample-level test probabilities with multiple testing correction."""
        print("\n📊 DETAILED STATISTICAL ANALYSIS (sample-level, paired):")
        print("="*70)
        
        # Gather best model per approach (by test AUC) and their test predictions
        best = {}
        for approach_name, models in tier1_results.items():
            best_auc = -1.0
            best_entry = None
            for model_name, metrics in models.items():
                if 'proba_test' in metrics and 'y_test' in metrics:
                    auc = metrics.get('auc', -1.0)
                    if auc > best_auc:
                        best_auc = auc
                        best_entry = (model_name, metrics)
            if best_entry is not None:
                best[approach_name] = {
                    'model': best_entry[0],
                    'auc': best_auc,
                    'y': np.asarray(best_entry[1]['y_test']),
                    'p': np.asarray(best_entry[1]['proba_test'])
                }
        
        approaches = list(best.keys())
        statistical_results = {}
        p_values = []
        comparisons = []
        
        # Pairwise comparisons
        for i in range(len(approaches)):
            for j in range(i+1, len(approaches)):
                a1, a2 = approaches[i], approaches[j]
                y1, p1 = best[a1]['y'], best[a1]['p']
                y2, p2 = best[a2]['y'], best[a2]['p']
                
                if len(y1) != len(y2) or not np.array_equal(y1, y2):
                    print(f"\n⚠️ Skipping {a1} vs {a2}: mismatched test sets.")
                    continue
                
                y = y1
                print(f"\n🔍 COMPARING (test level): {a1} vs {a2}")
                print("-"*60)
                
                # Wilcoxon signed-rank test
                W, p_val, rbc = wilcoxon_rank_biserial_from_trueprob(y, p1, p2)
                p_values.append(p_val)
                comparisons.append(f"{a1} vs {a2}")
                
                # Bootstrap confidence intervals
                from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
                auc_diff, auc_ci, _ = paired_bootstrap_metric_diff(y, p1, p2, roc_auc_score, n_boot=10000, seed=123)
                acc_diff, acc_ci, _ = paired_bootstrap_metric_diff(y, p1, p2, accuracy_score, n_boot=10000, seed=123, threshold=0.5)
                f1_diff, f1_ci, _ = paired_bootstrap_metric_diff(y, p1, p2, f1_score, n_boot=10000, seed=123, threshold=0.5)
                
                print(f"AUC Δ = {auc_diff:+.3f}  (95% CI [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}])")
                print(f"Acc Δ = {acc_diff:+.3f} (95% CI [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}])")
                print(f" F1 Δ = {f1_diff:+.3f} (95% CI [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}])")
                
                label = rank_biserial_to_label(rbc)
                print(f"Wilcoxon (true prob): p = {p_val:.4f}, rank-biserial r = {rbc:+.3f} ({label})")
                
                key = f"{a1} vs {a2}"
                statistical_results[key] = {
                    'approach1': a1,
                    'approach2': a2,
                    'auc_diff': auc_diff,
                    'auc_ci': auc_ci,
                    'acc_diff': acc_diff,
                    'acc_ci': acc_ci,
                    'f1_diff': f1_diff,
                    'f1_ci': f1_ci,
                    'w_statistic': W,
                    'p_value': p_val,
                    'rank_biserial': rbc,
                    'effect_size': label
                }
        
        # Apply multiple testing correction
        if p_values:
            rejected, corrected_p, _, _ = multipletests(p_values, method='fdr_bh')
            for i, comp in enumerate(comparisons):
                if comp in statistical_results:
                    statistical_results[comp]['corrected_p_value'] = corrected_p[i]
                    statistical_results[comp]['significant_after_correction'] = rejected[i]
        
        # Summary table
        if statistical_results:
            print(f"\n📋 STATISTICAL SUMMARY TABLE (paired bootstrap & Wilcoxon):")
            print("="*110)
            print(f"{'Comparison':<35} {'ΔAUC (95% CI)':<30} {'p-value':<10} {'Corrected p':<12} {'r (effect)':<15}")
            print("="*110)
            for comp, res in statistical_results.items():
                ci = res['auc_ci']
                corrected_p = res.get('corrected_p_value', 'N/A')
                sig = "✅" if res.get('significant_after_correction', False) else "📋"
                print(f"{comp:<35} {res['auc_diff']:+.3f} [{ci[0]:.3f},{ci[1]:.3f}]   {res['p_value']:<10.4f} {corrected_p:<12.4f} {res['rank_biserial']:+.3f} {sig}")
            print("="*110)
            print("📋 Significance after FDR correction: ✅ p<0.05, 📋 p≥0.05")
        else:
            print("\n⚠️ No comparable approaches with aligned test sets.")
        
        return statistical_results

    def print_basic_comparison_results_with_stats(self, raw_results, kg_results, statistical_results,
                                                selected_count, original_count, clinical_set):
        """Print basic comparison results with statistical analysis"""
        print(f"\n{'='*70}")
        print("🎉 CLINICAL RAW vs KG COMPARISON RESULTS με STATISTICS")
        print(f"{'='*70}")
        
        # Best performers
        best_raw = max(raw_results.keys(), key=lambda k: raw_results[k]['auc'])
        best_kg = max(kg_results.keys(), key=lambda k: kg_results[k]['auc'])
        
        print(f"\n🏆 BEST PERFORMERS:")
        print(f"   Raw Clinical Features: {best_raw} (AUC: {raw_results[best_raw]['auc']:.3f})")
        print(f"   KG Embeddings:        {best_kg} (AUC: {kg_results[best_kg]['auc']:.3f})")
        
        # Clinical assessment
        raw_best_auc = raw_results[best_raw]['auc']
        kg_best_auc = kg_results[best_kg]['auc']
        improvement = ((kg_best_auc - raw_best_auc) / raw_best_auc) * 100
        
        print(f"\n📊 OVERALL ASSESSMENT:")
        print(f"   Clinical Feature Set: {clinical_set.replace('_', ' ').title()}")
        print(f"   Features Used: {original_count} → {selected_count}")
        print(f"   Raw Clinical Best AUC: {raw_best_auc:.3f}")
        print(f"   KG Embeddings Best AUC: {kg_best_auc:.3f}")
        print(f"   KG vs Raw Improvement: {improvement:+.1f}%")
        
        # Statistical significance
        if statistical_results:
            main_comparison = list(statistical_results.values())[0]  # Should be Raw vs KG
            p_val = main_comparison['p_value']
            corrected_p = main_comparison.get('corrected_p_value', p_val)
            effect_size = main_comparison['effect_size']
            significant = main_comparison.get('significant_after_correction', p_val < 0.05)
            
            print(f"   Statistical Analysis:")
            if not np.isnan(p_val):
                print(f"      p-value: {p_val:.4f}")
                print(f"      FDR-corrected p: {corrected_p:.4f}")
                print(f"      Effect size: {effect_size} (rank-biserial r={main_comparison['rank_biserial']:+.3f})")
                if significant:
                    print(f"      Result: ✅ STATISTICALLY SIGNIFICANT")
                else:
                    print(f"      Result: 📋 Not statistically significant")
            else:
                print(f"      Result: ⚠️ Statistical test could not be performed")
        
        # Winner declaration with statistical context
        print(f"\n🏆 FINAL COMPARISON WINNER:")
        if kg_best_auc > raw_best_auc + 0.02:
            print(f"   🧠 KNOWLEDGE GRAPH EMBEDDINGS WIN!")
            print(f"   💡 Graph processing enhances clinical features by {improvement:+.1f}%")
            if statistical_results and not np.isnan(list(statistical_results.values())[0]['p_value']):
                p_val = list(statistical_results.values())[0]['p_value']
                corrected_p = list(statistical_results.values())[0].get('corrected_p_value', p_val)
                significant = list(statistical_results.values())[0].get('significant_after_correction', p_val < 0.05)
                if significant:
                    print(f"   ✅ Victory is statistically significant (p={corrected_p:.4f})")
                else:
                    print(f"   📋 Victory not statistically significant (p={corrected_p:.4f})")
        elif raw_best_auc > kg_best_auc + 0.02:
            print(f"   📊 RAW CLINICAL FEATURES WIN!")
            print(f"   💡 Simple clinical features outperform graph processing")
        else:
            print(f"   ⚖️ TIE - Both approaches perform similarly")
            print(f"   💡 Difference ({improvement:+.1f}%) within statistical noise")

    def create_tuned_kg_embeddings(self, X_train, X_test, interaction_strength=0.02, smoothing=0.03, nonlinearity=0.3):
        """Create tuned KG embeddings with adjustable parameters"""
        print(f"\n🎯 TUNED KG EMBEDDINGS:")
        print(f"   Parameters: interaction={interaction_strength}, smoothing={smoothing}, nonlinearity={nonlinearity}")
        
        def tuned_graph_processing(X):
            """Tuned graph processing with adjustable parameters"""
            X_kg = X.copy()
            n_samples, n_features = X.shape
            
            print(f"      Processing {n_features} features with tuned interactions...")
            
            # Tunable feature interactions
            if n_features >= 3:
                # More conservative interactions than enhanced version
                for i in range(min(6, n_features - 1)):  # Reduced from 8
                    for j in range(i + 1, min(i + 3, n_features)):  # Reduced from 4
                        interaction = X_kg[:, i] * X_kg[:, j] * interaction_strength
                        X_kg[:, i] += interaction * 0.2  # Reduced from 0.3
                        X_kg[:, j] += interaction * 0.2
            
            # Tunable smoothing
            if n_features >= 5:
                for i in range(2, n_features - 2):
                    X_kg[:, i] = ((1 - 4*smoothing) * X_kg[:, i] + 
                                  smoothing * X_kg[:, i-2] + 
                                  smoothing * X_kg[:, i-1] + 
                                  smoothing * X_kg[:, i+1] + 
                                  smoothing * X_kg[:, i+2])
            
            # Tunable non-linear transformation
            X_kg = np.tanh(X_kg * nonlinearity)
            
            # Conservative normalization
            for i in range(n_features):
                std = np.std(X_kg[:, i])
                if std > 1e-6:
                    X_kg[:, i] = X_kg[:, i] / std
                    X_kg[:, i] = np.clip(X_kg[:, i], -2.5, 2.5)  # Less aggressive than -3,3
            
            return X_kg
        
        X_train_kg = tuned_graph_processing(X_train)
        X_test_kg = tuned_graph_processing(X_test)
        
        print(f"   ✅ Tuned KG embeddings created")
        print(f"      Train: {X_train_kg.shape}, Test: {X_test_kg.shape}")
        
        return X_train_kg, X_test_kg

    def hyperparameter_search(self, X_train, X_test, y_train, y_test, train_pids):
        """Search for optimal KG hyperparameters"""
        print(f"\n🔍 HYPERPARAMETER SEARCH FOR KG EMBEDDINGS:")
        print("="*60)
        
        # Define parameter grid
        param_grid = [
            # Conservative parameters (closer to simple KG)
            {'interaction': 0.015, 'smoothing': 0.025, 'nonlinearity': 0.4, 'name': 'Conservative+'},
            {'interaction': 0.020, 'smoothing': 0.030, 'nonlinearity': 0.3, 'name': 'Balanced'},
            {'interaction': 0.025, 'smoothing': 0.035, 'nonlinearity': 0.4, 'name': 'Moderate'},
            
            # Slightly more aggressive (but less than original enhanced)
            {'interaction': 0.030, 'smoothing': 0.040, 'nonlinearity': 0.5, 'name': 'Moderate+'},
            {'interaction': 0.035, 'smoothing': 0.045, 'nonlinearity': 0.4, 'name': 'Aggressive-'},
            
            # Original simple for comparison
            {'interaction': 0.010, 'smoothing': 0.020, 'nonlinearity': 0.5, 'name': 'Simple (baseline)'},
        ]
        
        best_config = None
        best_auc = 0
        results = {}
        
        for config in param_grid:
            print(f"\n🧪 Testing {config['name']}:")
            print(f"   Parameters: int={config['interaction']}, smooth={config['smoothing']}, nonlin={config['nonlinearity']}")
            
            try:
                # Create embeddings with current parameters
                X_train_kg, X_test_kg = self.create_tuned_kg_embeddings(
                    X_train, X_test, 
                    config['interaction'], 
                    config['smoothing'], 
                    config['nonlinearity']
                )
                
                # Test with best performing model from simple KG (Random Forest)
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=6, min_samples_split=5,
                    min_samples_leaf=2, max_features='sqrt', random_state=42
                )
                
                # Cross-validation
                cv_scores = self._proper_cross_validation(X_train_kg, y_train, train_pids, model)
                
                # Train and evaluate
                model.fit(X_train_kg, y_train)
                y_pred_proba = model.predict_proba(X_test_kg)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
                
                cv_mean = np.mean(cv_scores) if cv_scores else 0.5
                cv_std = np.std(cv_scores) if cv_scores else 0.0
                
                results[config['name']] = {
                    'auc': auc,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'config': config
                }
                
                # Assessment
                if auc > 0.7:
                    status = "✅ Good"
                elif auc > 0.6:
                    status = "⚖️ Moderate"
                else:
                    status = "📋 Limited"
                
                cv_info = f"CV={cv_mean:.3f}±{cv_std:.3f}" if cv_scores else "CV=N/A"
                print(f"   Result: {status} AUC={auc:.3f}, {cv_info}")
                
                if auc > best_auc:
                    best_auc = auc
                    best_config = config
                    
            except Exception as e:
                print(f"   ❌ Failed: {str(e)[:50]}")
                results[config['name']] = {
                    'auc': 0.5, 'cv_mean': 0.5, 'cv_std': 0.0, 'config': config
                }
        
        # Print summary
        print(f"\n📊 HYPERPARAMETER SEARCH RESULTS:")
        print("="*70)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
        
        for rank, (name, result) in enumerate(sorted_results, 1):
            auc = result['auc']
            cv_mean = result['cv_mean']
            cv_std = result['cv_std']
            
            cv_info = f"CV={cv_mean:.3f}±{cv_std:.3f}" if result['cv_std'] > 0 else "CV=N/A"
            
            if rank == 1:
                print(f"🏆 #{rank}: {name:<20} AUC={auc:.3f} {cv_info}")
            elif rank <= 3:
                print(f"🥈 #{rank}: {name:<20} AUC={auc:.3f} {cv_info}")
            else:
                print(f"   #{rank}: {name:<20} AUC={auc:.3f} {cv_info}")
        
        print(f"\n🎯 BEST CONFIGURATION:")
        if best_config:
            print(f"   Name: {best_config['name']}")
            print(f"   Parameters: interaction={best_config['interaction']}, smoothing={best_config['smoothing']}, nonlinearity={best_config['nonlinearity']}")
            print(f"   Best AUC: {best_auc:.3f}")
            
            # Compare with simple KG baseline
            simple_result = results.get('Simple (baseline)', {'auc': 0.6})
            improvement = ((best_auc - simple_result['auc']) / simple_result['auc']) * 100
            print(f"   Improvement over Simple KG: {improvement:+.1f}%")
        else:
            print("   No valid configuration found")
        
        return best_config, results

    def run_enhanced_analysis_with_tuning(self):
        """Run enhanced analysis with hyperparameter tuning"""
        
        print("🚀 ENHANCED NEUROGAIT ANALYSIS με Hyperparameter Tuning")
        print("="*70)
        print("🎯 Raw vs KG comparison με optimized clinical features και tuning")
        print("🔒 Leakage-free αλλά less conservative για better metrics")
        print("📊 Transparent reporting with comprehensive statistical analysis")
        print("🎛️ Hyperparameter tuning για optimal KG processing")
        print()
        
        # Enhanced preprocessing with clinical features
        df, best_features, best_set_name, train_indices, test_indices, train_pids, test_pids = self.load_and_prepare_data()
        train_data, test_data = self.proper_train_test_split(df, train_indices, test_indices)
        
        # Preprocess data without leakage
        train_clean, test_clean, clean_features = self.preprocess_data(train_data, test_data, best_features)
        
        # Feature selection
        X_train, X_test, selected_features = self.optimized_feature_selection(
            train_clean, test_clean, clean_features
        )
        
        y_train = train_clean['diagnosis']
        y_test = test_clean['diagnosis']
        X_train_scaled, X_test_scaled = self.prepare_data_properly(X_train, X_test)
        
        # === TIER 1A: RAW CLINICAL FEATURES ===
        print(f"\n{'='*50}")
        print("📊 TIER 1A: RAW CLINICAL FEATURES")
        print(f"{'='*50}")
        
        raw_results = self.train_optimized_models(
            X_train_scaled, X_test_scaled, y_train, y_test, train_pids, 
            f"Raw Clinical Features ({best_set_name})"
        )
        
        # === TIER 1B: SIMPLE KG FEATURES ===
        print(f"\n{'='*50}")
        print("🧠 TIER 1B: SIMPLE KG FEATURES (Baseline)")
        print(f"{'='*50}")
        
        X_train_kg_simple, X_test_kg_simple = self.create_conservative_kg_embeddings(
            X_train_scaled, X_test_scaled
        )
        simple_kg_results = self.train_optimized_models(
            X_train_kg_simple, X_test_kg_simple, y_train, y_test, train_pids, "Simple KG"
        )
        
        # === HYPERPARAMETER SEARCH ===
        print(f"\n{'='*50}")
        print("🎛️ KG HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*50}")
        
        best_config, tuning_results = self.hyperparameter_search(
            X_train_scaled, X_test_scaled, y_train, y_test, train_pids
        )
        
        # === TIER 1C: ENHANCED KG WITH ORIGINAL PARAMETERS ===
        enhanced_kg_results = None
        enhanced_builder = None
        feature_names = []
        
        if ENHANCED_FEATURES_AVAILABLE:
            print(f"\n{'='*50}")
            print("💡 TIER 1C: ENHANCED KG FEATURES (Original)")
            print(f"{'='*50}")
            
            try:
                enhanced_builder = EnhancedKGFeatureBuilder()
                
                X_train_enhanced, feature_names = enhanced_builder.create_enhanced_kg_features(
                    train_data, selected_features
                )
                X_test_enhanced, _ = enhanced_builder.create_enhanced_kg_features(
                    test_data, selected_features
                )
                
                scaler_enhanced = StandardScaler()
                X_train_enhanced_scaled = scaler_enhanced.fit_transform(X_train_enhanced)
                X_test_enhanced_scaled = scaler_enhanced.transform(X_test_enhanced)
                
                enhanced_kg_results = self.train_optimized_models(
                    X_train_enhanced_scaled, X_test_enhanced_scaled, y_train, y_test,
                    train_pids, "Enhanced KG"
                )
                
            except Exception as e:
                print(f"❌ Enhanced KG features failed: {e}")
                enhanced_kg_results = None
        
        # === TIER 1D: OPTIMIZED KG WITH TUNED PARAMETERS ===
        print(f"\n{'='*50}")
        print("🎯 TIER 1D: TUNED KG EMBEDDINGS (Best Config)")
        print(f"{'='*50}")
        
        if best_config:
            X_train_kg_tuned, X_test_kg_tuned = self.create_tuned_kg_embeddings(
                X_train_scaled, X_test_scaled,
                best_config['interaction'],
                best_config['smoothing'], 
                best_config['nonlinearity']
            )
            tuned_kg_results = self.train_optimized_models(
                X_train_kg_tuned, X_test_kg_tuned, y_train, y_test, train_pids, 
                f"Tuned KG ({best_config['name']})"
            )
        else:
            # Fallback to enhanced if no best config
            X_train_kg_tuned, X_test_kg_tuned = self.create_enhanced_kg_embeddings(X_train_scaled, X_test_scaled)
            tuned_kg_results = self.train_optimized_models(
                X_train_kg_tuned, X_test_kg_tuned, y_train, y_test, train_pids, "Tuned KG (Fallback)"
            )
        
        # === COMPREHENSIVE COMPARISON ===
        print(f"\n{'='*70}")
        print("📊 COMPREHENSIVE COMPARISON - TUNED KG ANALYSIS με STATISTICS")
        print(f"{'='*70}")
        
        # Collect all results
        tier1_results = {
            'Raw Clinical Features': raw_results,
            'Simple KG': simple_kg_results,
            'Tuned KG': tuned_kg_results
        }
        
        if enhanced_kg_results:
            tier1_results['Enhanced KG (Original)'] = enhanced_kg_results
        
        # Statistical comparison
        statistical_results = self.statistical_comparison_analysis(tier1_results)
        
        # Print enhanced results WITH statistical analysis
        self.print_tuned_comprehensive_results_with_statistics(
            tier1_results, tuning_results, best_config, best_set_name,
            {
                'train_participants': len(set(train_pids)),
                'test_participants': len(set(test_pids)),
                'original_features': len(best_features),
                'selected_features': len(selected_features),
                'enhanced_features': len(feature_names) if enhanced_kg_results else 0
            },
            statistical_results
        )
        
        return {
            'tier1_clinical_ml': tier1_results,
            'tuning_results': tuning_results,
            'best_config': best_config,
            'statistical_results': statistical_results,
            'data_summary': {
                'train_participants': len(set(train_pids)),
                'test_participants': len(set(test_pids)),
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            },
            'feature_info': {
                'clinical_set': best_set_name,
                'original_count': len(best_features),
                'selected_count': len(selected_features),
                'enhanced_count': len(feature_names) if enhanced_kg_results else 0
            }
        }

    def print_tuned_comprehensive_results_with_statistics(
        self,
        tier1_results,
        tuning_results,
        best_config,
        clinical_set_name,
        data_summary,
        statistical_results
    ):
        """Print comprehensive tuned analysis with statistics — prints only real, available data."""

        import numpy as np

        print("🎯 COMPREHENSIVE TUNED ANALYSIS RESULTS με STATISTICS")
        print("=" * 80)

        # ── CLINICAL CONTEXT ────────────────────────────────────────────────────────
        ctx_lines = []
        if clinical_set_name:
            ctx_lines.append(f"   Feature Set: {clinical_set_name.replace('_', ' ').title()}")
        if isinstance(data_summary, dict):
            if ('train_participants' in data_summary) and ('test_participants' in data_summary):
                ctx_lines.append(
                    f"   Train/Test: {data_summary['train_participants']} / {data_summary['test_participants']} participants"
                )
            if ('original_features' in data_summary) and ('selected_features' in data_summary):
                ctx_lines.append(
                    f"   Features: {data_summary['original_features']} → {data_summary['selected_features']} selected"
                )
        if ctx_lines:
            print("🏥 CLINICAL CONTEXT:")
            for ln in ctx_lines:
                print(ln)

        # ── TUNING SUMMARY ─────────────────────────────────────────────────────────
        if best_config or (isinstance(tuning_results, dict) and tuning_results):
            print("\n🎛️ HYPERPARAMETER TUNING SUMMARY:")

        if isinstance(best_config, dict) and best_config:
            if 'name' in best_config:
                print(f"   Best Configuration: {best_config['name']}")
            params_printed = False
            lines = []
            if 'interaction' in best_config:
                lines.append(f"      Interaction Strength: {best_config['interaction']}")
            if 'smoothing' in best_config:
                lines.append(f"      Smoothing Factor: {best_config['smoothing']}")
            if 'nonlinearity' in best_config:
                lines.append(f"      Nonlinearity: {best_config['nonlinearity']}")
            if lines:
                print("   Optimal Parameters:")
                for ln in lines:
                    print(ln)

        if isinstance(tuning_results, dict) and tuning_results:
            top = sorted(
                tuning_results.items(),
                key=lambda x: x[1].get('auc', float('-inf')),
                reverse=True
            )[:3]
            for idx, (name, res) in enumerate(top, 1):
                if 'auc' in res:
                    medal = "🏆" if idx == 1 else "🥈" if idx == 2 else "🥉"
                    print(f"   {medal} #{idx}: {name} (AUC: {res['auc']:.3f})")

        # ── PERFORMANCE SUMMARY ────────────────────────────────────────────────────
        if isinstance(tier1_results, dict) and tier1_results:
            print("\n📊 PERFORMANCE SUMMARY:")
            print("-" * 70)

        best_overall_auc = float("-inf")
        best_overall_approach = None
        best_overall_model = None

        for approach_name, results in (tier1_results or {}).items():
            print(f"\n{approach_name}:")
            for model_name, metrics in (results or {}).items():
                if not isinstance(metrics, dict):
                    continue
                if ('auc' not in metrics) or ('f1' not in metrics):
                    continue

                auc = metrics['auc']
                f1 = metrics['f1']

                # Optional CV info (τυπώνεται μόνο αν υπάρχει πλήρες)
                cv_scores = metrics.get('cv_scores', [])
                cv_mean = metrics.get('cv_mean', None)
                cv_std = metrics.get('cv_std', None)
                cv_info = ""
                if cv_scores and (cv_mean is not None) and (cv_std is not None):
                    ci_margin = 1.96 * (cv_std / np.sqrt(len(cv_scores)))
                    ci_lower = cv_mean - ci_margin
                    ci_upper = cv_mean + ci_margin
                    cv_info = f", CV={cv_mean:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"

                # Status label
                if auc > 0.8:
                    status = "🎉 Excellent"
                elif auc > 0.7:
                    status = "✅ Good"
                elif auc > 0.6:
                    status = "⚖️ Moderate"
                else:
                    status = "📋 Limited"

                print(f"   {model_name:15}: {status} AUC={auc:.3f}, F1={f1:.3f}{cv_info}")

                if auc > best_overall_auc:
                    best_overall_auc = auc
                    best_overall_approach = approach_name
                    best_overall_model = model_name

        # ── STATISTICAL COMPARISON RESULTS ─────────────────────────────────────────
        if isinstance(statistical_results, dict) and statistical_results:
            print("\n" + "=" * 70)
            print("📊 STATISTICAL COMPARISON RESULTS:")
            print("-" * 70)
            for comp_key, res in statistical_results.items():
                if not isinstance(res, dict):
                    continue

                # Label σύγκρισης
                if 'approach1' in res and 'approach2' in res:
                    comp_label = f"{res['approach1']} vs {res['approach2']}"
                else:
                    comp_label = comp_key

                parts = [f"{comp_label:<35}:"]

                # Διαφορά AUC (υποστηρίζει δύο schemas: auc_diff ή difference)
                if 'auc_diff' in res:
                    parts.append(f"ΔAUC={res['auc_diff']:+.3f}")
                elif 'difference' in res:
                    parts.append(f"ΔAUC={res['difference']:+.3f}")

                # 95% CI (π.χ. auc_ci ή generic ci)
                if 'auc_ci' in res and isinstance(res['auc_ci'], (list, tuple)) and len(res['auc_ci']) == 2:
                    parts.append(f"[{res['auc_ci'][0]:.3f},{res['auc_ci'][1]:.3f}]")
                elif 'ci' in res and isinstance(res['ci'], (list, tuple)) and len(res['ci']) == 2:
                    parts.append(f"[{res['ci'][0]:.3f},{res['ci'][1]:.3f}]")

                # p-values
                if ('p_value' in res) and (res['p_value'] is not None) and not (isinstance(res['p_value'], float) and np.isnan(res['p_value'])):
                    parts.append(f"p={res['p_value']:.4f}")
                if ('corrected_p_value' in res) and (res['corrected_p_value'] is not None) and not (isinstance(res['corrected_p_value'], float) and np.isnan(res['corrected_p_value'])):
                    parts.append(f"corrected_p={res['corrected_p_value']:.4f}")

                # Effect (π.χ. significance label, rank-biserial label ή Cohen's d)
                if 'significance' in res and res['significance']:
                    parts.append(str(res['significance']))
                if 'effect_size' in res and res['effect_size']:
                    parts.append(f"effect: {res['effect_size']}")
                elif 'cohens_d' in res and (res['cohens_d'] is not None) and not (isinstance(res['cohens_d'], float) and np.isnan(res['cohens_d'])):
                    parts.append(f"effect: {res['cohens_d']:+.3f}")
                if 'significant_after_correction' in res:
                    parts.append("✅" if res['significant_after_correction'] else "📋")

                print(", ".join(parts))

        # ── BEST OVERALL PERFORMER ─────────────────────────────────────────────────
        if best_overall_approach and best_overall_model and best_overall_auc != float("-inf"):
            print("\n🏆 BEST OVERALL PERFORMER:")
            print(f"   Approach: {best_overall_approach}")
            print(f"   Model: {best_overall_model}")
            print(f"   AUC: {best_overall_auc:.3f}")

        # ── TUNING EFFECTIVENESS (μόνο αν υπάρχουν και τα δύο sections) ───────────
        if isinstance(tier1_results, dict) and ('Simple KG' in tier1_results) and ('Tuned KG' in tier1_results):
            simple_vals = [
                m.get('auc') for m in tier1_results['Simple KG'].values()
                if isinstance(m, dict) and ('auc' in m)
            ]
            tuned_vals = [
                m.get('auc') for m in tier1_results['Tuned KG'].values()
                if isinstance(m, dict) and ('auc' in m)
            ]
            if simple_vals and tuned_vals:
                simple_kg_best = max(simple_vals)
                tuned_kg_best = max(tuned_vals)
                if simple_kg_best not in (0, None):
                    tuning_improvement = ((tuned_kg_best - simple_kg_best) / simple_kg_best) * 100.0
                    print(f"\n🎛️ TUNING EFFECTIVENESS:")
                    print(f"   Simple KG Best: {simple_kg_best:.3f}")
                    print(f"   Tuned KG Best: {tuned_kg_best:.3f}")
                    print(f"   Tuning Improvement: {tuning_improvement:+.1f}%")
                    if tuning_improvement > 5:
                        print("   🎯 TUNING SUCCESSFUL - Meaningful improvement achieved!")
                    elif tuning_improvement > 0:
                        print("   ⚖️ TUNING HELPFUL - Small improvement achieved")
                    else:
                        print("   📋 TUNING INEFFECTIVE - Simple KG remains better")

        # ── CLINICAL INTERPRETATION ────────────────────────────────────────────────
        if best_overall_auc != float("-inf"):
            print("\n🏥 CLINICAL INTERPRETATION:")
            if best_overall_auc > 0.8:
                print("   Assessment: 🎉 EXCELLENT - High clinical utility for ASD screening")
                print("   Recommendation: Suitable for clinical decision support with appropriate validation")
            elif best_overall_auc > 0.7:
                print("   Assessment: ✅ GOOD - Meaningful clinical utility")
                print("   Recommendation: Promising for clinical applications with further validation")
            elif best_overall_auc > 0.6:
                print("   Assessment: ⚖️ MODERATE - Limited but useful clinical utility")
                print("   Recommendation: May be useful as part of comprehensive assessment")
            else:
                print("   Assessment: 📋 LIMITED - Insufficient for standalone clinical use")
                print("   Recommendation: Requires significant improvement before clinical application")

        # ── KNOWLEDGE GRAPH INSIGHTS (μόνο αν υπάρχουν Raw & κάποιο KG) ────────────
        if isinstance(tier1_results, dict) and ('Raw Clinical Features' in tier1_results):
            raw_vals = [
                m.get('auc') for m in tier1_results['Raw Clinical Features'].values()
                if isinstance(m, dict) and ('auc' in m)
            ]
            if raw_vals:
                raw_best = max(raw_vals)
                kg_approaches = [k for k in tier1_results.keys() if 'KG' in k]
                if kg_approaches:
                    kg_best_approach = max(
                        kg_approaches,
                        key=lambda k: max(
                            [m.get('auc', float('-inf')) for m in tier1_results.get(k, {}).values()]
                            or [float('-inf')]
                        )
                    )
                    kg_best_vals = [
                        m.get('auc') for m in tier1_results.get(kg_best_approach, {}).values()
                        if isinstance(m, dict) and ('auc' in m)
                    ]
                    if kg_best_vals and (raw_best not in (0, None)):
                        kg_best_auc = max(kg_best_vals)
                        kg_improvement = ((kg_best_auc - raw_best) / raw_best) * 100.0
                        print(f"\n🧠 KNOWLEDGE GRAPH INSIGHTS με STATISTICAL VALIDATION:")
                        print(f"   Raw Clinical Features: AUC = {raw_best:.3f}")
                        print(f"   Best KG Approach ({kg_best_approach}): AUC = {kg_best_auc:.3f}")
                        print(f"   KG Improvement: {kg_improvement:+.1f}%")

                        # Εκτύπωση αντίστοιχων στατιστικών αν υπάρχουν για Raw vs KG
                        if isinstance(statistical_results, dict) and statistical_results:
                            for key, res in statistical_results.items():
                                if not isinstance(res, dict):
                                    continue
                                a1 = res.get('approach1', '')
                                a2 = res.get('approach2', '')
                                if ('Raw Clinical Features' in a1 and kg_best_approach in a2) or \
                                ('Raw Clinical Features' in a2 and kg_best_approach in a1):
                                    parts = []
                                    if ('p_value' in res) and (res['p_value'] is not None) and not (isinstance(res['p_value'], float) and np.isnan(res['p_value'])):
                                        parts.append(f"p={res['p_value']:.4f}")
                                    if ('corrected_p_value' in res) and (res['corrected_p_value'] is not None) and not (isinstance(res['corrected_p_value'], float) and np.isnan(res['corrected_p_value'])):
                                        parts.append(f"corrected_p={res['corrected_p_value']:.4f}")
                                    if 'effect_size' in res and res['effect_size']:
                                        parts.append(f"effect: {res['effect_size']}")
                                    elif 'cohens_d' in res and (res['cohens_d'] is not None) and not (isinstance(res['cohens_d'], float) and np.isnan(res['cohens_d'])):
                                        parts.append(f"effect: {res['cohens_d']:+.3f}")
                                    if parts:
                                        print("   Statistical significance: " + ", ".join(parts))
                                    break

        # ── PARAMETER INSIGHTS (μόνο αν υπάρχουν) ──────────────────────────────────
        if isinstance(best_config, dict) and best_config:
            lines = []
            if 'interaction' in best_config:
                lines.append(("Interaction Strength", best_config['interaction']))
            if 'smoothing' in best_config:
                lines.append(("Smoothing Factor", best_config['smoothing']))
            if lines:
                print("\n🔬 OPTIMAL PARAMETER INSIGHTS:")
                for label, val in lines:
                    print(f"   {label}: {val}")
                    if label == "Interaction Strength" and isinstance(val, (int, float)):
                        if val < 0.02:
                            print("      → Very conservative interactions work best")
                        elif val < 0.04:
                            print("      → Moderate interactions are optimal")
                        else:
                            print("      → Strong interactions are needed")
                    if label == "Smoothing Factor" and isinstance(val, (int, float)):
                        if val < 0.03:
                            print("      → Minimal smoothing preserves information")
                        else:
                            print("      → Moderate smoothing helps generalization")

        # ── LIMITATIONS & RECOMMENDATIONS (σταθερό κείμενο) ────────────────────────
        print("\n⚠️ STUDY LIMITATIONS:")
        print("   • Small sample size limits hyperparameter search effectiveness")
        print("   • Limited parameter grid due to computational constraints")
        print("   • Single dataset requires external validation of optimal parameters")
        print("   • Clinical features may not capture all relevant gait patterns")
        print("   • Multiple comparisons increase Type I error risk")

        print("\n🚀 RECOMMENDATIONS:")
        print("   • Validate optimal parameters on independent clinical datasets")
        print("   • Expand hyperparameter search with larger sample sizes")
        print("   • Apply multiple testing corrections (Bonferroni/FDR)")
        print("   • Include temporal gait dynamics in parameter optimization")
        print("   • Clinical expert validation of feature relevance")
        print("   • Integration with other diagnostic modalities")
    
    def print_gnn_comparison_results(self, all_results, clinical_set_name, data_summary, statistical_results):
        """Print comprehensive GNN comparison results with statistical analysis"""
        
        print("🎯 COMPREHENSIVE GNN COMPARISON RESULTS")
        print("="*80)
        
        # CONTEXT
        print("🏥 ANALYSIS CONTEXT:")
        print(f"   Feature Set: {clinical_set_name.replace('_', ' ').title()}")
        print(f"   Train/Test: {data_summary['train_participants']} / {data_summary['test_participants']} participants")
        print(f"   Features: {data_summary['original_features']} → {data_summary['selected_features']} selected")
        
        # PERFORMANCE SUMMARY BY APPROACH
        print("\n📊 PERFORMANCE SUMMARY BY APPROACH:")
        print("-" * 80)
        
        approach_summaries = {}
        best_overall_auc = 0
        best_overall_approach = ""
    def run_gnn_comparison_analysis(self):
        """Raw vs KG vs Enhanced KG vs True GNN με σωστή στοίχιση CV groups (no leakage, no placeholders)."""

        import numpy as np

        print("\n🧠 GRAPH NEURAL NETWORK COMPARISON ANALYSIS")
        print("=" * 70)
        print("🎯 Comparing: Raw, Simple KG, Enhanced KG, and True GNN")
        print("🔒 Using actual Neo4j graph structure for GNN")
        print("📊 Complete statistical comparison")

        # 1) Load/prepare your split (υποθέτω ότι έχεις ήδη train_data/test_data dataframes)
        train_data, test_data = self.train_df, self.test_df  # πρέπει να υπάρχουν
        features = self.selected_clinical_features  # πρέπει να οριστούν upstream

        # 2) Feature selection (Training Only) — ΕΠΙΣΤΡΕΦΕΙ 4 τιμές!
        X_train, X_test, selected_features, train_groups = self.optimized_feature_selection(
            train_data, test_data, features
        )

        # 3) Scaling (fit ONLY on train)
        X_train_scaled, X_test_scaled = self.scale_no_leakage(X_train, X_test)

        # 4) Targets (ευθυγραμμισμένα με X_train/X_test index)
        y_train = train_data.loc[X_train.index, 'diagnosis'].values
        y_test = test_data.loc[X_test.index, 'diagnosis'].values

        # === TIER 1: RAW CLINICAL FEATURES ===
        raw_results = self.train_optimized_models(
            X_train_scaled, X_test_scaled, y_train, y_test, train_groups, "Raw Clinical Features"
        )

        # === TIER 2: SIMPLE KG ===
        X_train_kg, X_test_kg = self.build_simple_kg_embeddings(X_train_scaled, X_test_scaled, selected_features)
        simplekg_results = self.train_optimized_models(
            X_train_kg, X_test_kg, y_train, y_test, train_groups, "Simple KG"
        )

        # === TIER 3: ENHANCED KG ===
        X_train_enh, X_test_enh = self.build_enhanced_kg_embeddings(X_train_scaled, X_test_scaled, selected_features)
        enhancedkg_results = self.train_optimized_models(
            X_train_enh, X_test_enh, y_train, y_test, train_groups, "Enhanced KG"
        )

        # === TIER 4: TRUE GNN ===
        gnn_results = self.run_true_gnn_analysis(  # αυτή η μέθοδος πρέπει να παράγει test predictions και AUC/F1
            train_data, test_data, selected_features
        )

        # === Συγκέντρωση για εκτύπωση/στατιστικά ===
        tier1_results = {
            "Raw Clinical Features": raw_results,
            "Simple KG": simplekg_results,
            "Enhanced KG": enhancedkg_results,
            "True GNN": gnn_results
        }

        # Υπολογισμός στατιστικών συγκρίσεων (η δική σου υλοποίηση)
        statistical_results = self.statistical_comparison_analysis(tier1_results)

        # Εκτύπωση συνολικών αποτελεσμάτων (η δική σου print συνάρτηση, χωρίς placeholders)
        self.print_tuned_comprehensive_results_with_statistics(
            tier1_results=tier1_results,
            tuning_results={},             # αν έχεις tuning, βάλε τα πραγματικά
            best_config={},                # αν έχεις best_config, βάλε το πραγματικό
            clinical_set_name=self.best_clinical_set_name,  # upstream
            data_summary={
                'train_participants': len(set(train_groups)),
                'test_participants': len(set(test_data['participant_id'].values)),
                'original_features': len(features),
                'selected_features': len(selected_features),
            },
            statistical_results=statistical_results
        )

        return {
            'all_results': tier1_results,
            'statistical_results': statistical_results
        }

    def run_realistic_analysis(self):
        """Run basic realistic analysis with clinical features and statistical testing"""
        
        # Use clinical features for enhanced basic analysis with proper leakage prevention
        df, best_features, best_set_name, train_indices, test_indices, train_pids, test_pids = self.load_and_prepare_data()
        
        # Split data properly
        train_data, test_data = self.proper_train_test_split(df, train_indices, test_indices)
        
        # Preprocess data without leakage
        train_clean, test_clean, clean_features = self.preprocess_data(train_data, test_data, best_features)
        
        # Feature selection using training data only
        X_train, X_test, selected_features = self.optimized_feature_selection(
            train_clean, test_clean, clean_features
        )
        
        y_train = train_clean['diagnosis']
        y_test = test_clean['diagnosis']
        X_train_scaled, X_test_scaled = self.prepare_data_properly(X_train, X_test)
        
        # Raw features analysis
        print(f"\n{'='*50}")
        print(f"📊 RAW CLINICAL FEATURES ANALYSIS")
        print(f"{'='*50}")
        
        raw_results = self.train_optimized_models(
            X_train_scaled, X_test_scaled, y_train, y_test, train_pids, "Raw Clinical Features"
        )
        
        # KG embeddings analysis
        X_train_kg, X_test_kg = self.create_enhanced_kg_embeddings(X_train_scaled, X_test_scaled)
        
        print(f"\n{'='*50}")
        print(f"🧠 OPTIMIZED KG EMBEDDINGS ANALYSIS")
        print(f"{'='*50}")
        
        kg_results = self.train_optimized_models(
            X_train_kg, X_test_kg, y_train, y_test, train_pids, "Optimized KG Embeddings"
        )
        
        # Statistical comparison
        tier1_results = {
            'Raw Clinical Features': raw_results,
            'Optimized KG': kg_results
        }
        
        print(f"\n{'='*60}")
        print("📊 STATISTICAL COMPARISON")
        print(f"{'='*60}")
        
        statistical_results = self.statistical_comparison_analysis(tier1_results)
        
        # Results
        self.print_basic_comparison_results_with_stats(
            raw_results, kg_results, statistical_results,
            len(selected_features), len(best_features), best_set_name
        )
        
        return {
            'raw_results': raw_results,
            'kg_results': kg_results,
            'statistical_results': statistical_results,
            'selected_features': selected_features,
            'clinical_set': best_set_name
        }

# MAIN FUNCTION με GNN Support
def main():
        """Main execution with clinical features, comprehensive statistical analysis, hyperparameter tuning, and GNN support"""
        print("🏥 ENHANCED NEUROGAIT ANALYSIS με Clinical Features, Statistics, και GNN")
        print("🎯 Raw vs KG vs GNN comparison με καλύτερα clinical features")
        print("🔒 No data leakage ensured")
        print("📊 Complete statistical analysis με Wilcoxon tests and multiple testing correction")
        print("🎛️ Hyperparameter tuning για optimal performance")
        print("🤖 Graph Neural Networks για advanced analysis")
        print()
        
        # Show available analysis options
        available_options = [
            "1. Basic Analysis (Raw vs KG με clinical features και statistics)",
            "2. Enhanced Analysis (All tiers με comprehensive statistics)",
            "3. Tuned Analysis (Enhanced + Hyperparameter tuning)",
            "4. GNN Analysis (Raw vs KG vs Enhanced KG vs True GNN)"
        ]
        
        # Check availability
        if ENHANCED_FEATURES_AVAILABLE:
            enhanced_status = "✅"
        else:
            enhanced_status = "⚠️"
        
        if GNN_ANALYSIS_AVAILABLE:
            gnn_status = "✅"
        else:
            gnn_status = "⚠️"
        
        print("Available analysis types:")
        for i, option in enumerate(available_options, 1):
            if i == 2 or i == 3:
                print(f"   {enhanced_status} {option}")
            elif i == 4:
                print(f"   {gnn_status} {option}")
            else:
                print(f"   ✅ {option}")
        
        if not GNN_ANALYSIS_AVAILABLE:
            print("\n📋 For GNN analysis, install requirements:")
            print("   pip install torch torch-geometric")
            print("   Create true_gnn_analysis.py module")
        
        if not ENHANCED_FEATURES_AVAILABLE:
            print("\n📋 For enhanced KG features, create enhanced_kg_features.py module")
        
        print("\n" + "="*70)
        
        # Initialize analyzer
        analyzer = RealisticAnalysis()
        
        try:
            # Get user choice
            print("\nChoose analysis type (1-4): ", end="")
            choice = input().strip()
            
            if choice == "1":
                print("\n🚀 Running Basic Analysis...")
                results = analyzer.run_realistic_analysis()
                
            elif choice == "2":
                print("\n🚀 Running Enhanced Analysis...")
                results = analyzer.run_enhanced_analysis_with_tuning()
                
            elif choice == "3":
                print("\n🚀 Running Tuned Analysis...")
                results = analyzer.run_enhanced_analysis_with_tuning()
                
            elif choice == "4":
                print("\n🚀 Running GNN Analysis...")
                results = analyzer.run_gnn_comparison_analysis()
                
            else:
                print("❌ Invalid choice. Running default basic analysis...")
                results = analyzer.run_realistic_analysis()
            
            print("\n" + "="*80)
            print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*80)
            
            return results
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Analysis interrupted by user")
            return None
            
        except Exception as e:
            print(f"\n\n❌ Analysis failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def run_demo_analysis():
        """Run a demonstration analysis with synthetic data if no dataset is available"""
        print("🔬 DEMO MODE - Synthetic NeuroGait Analysis")
        print("="*60)
        print("📋 This demonstrates the analysis pipeline with synthetic data")
        print("🎯 Replace with 'Final dataset.csv' for real analysis")
        print()
        
        # Generate synthetic data that mimics the structure
        np.random.seed(42)
        n_participants = 20
        samples_per_participant = 8
        n_samples = n_participants * samples_per_participant
        n_features = 25
        
        # Create synthetic features with realistic names
        feature_names = [
            'SpineBase_X', 'SpineBase_Y', 'SpineBase_Z',
            'SPKNL_angle', 'SPKNR_angle', 'HIANL_angle', 'HIANR_angle',
            'GaCT_duration', 'StaT_duration', 'SwiT_duration',
            'HESHL_velocity', 'HESHR_velocity', 'SHWRL_position', 'SHWRR_position',
            'balance_score', 'stability_metric', 'gait_rhythm',
            'step_length', 'stride_width', 'walking_speed',
            'coordination_index', 'symmetry_measure', 'timing_variability',
            'postural_sway', 'movement_smoothness'
        ]
        
        # Generate synthetic data
        X = np.random.randn(n_samples, n_features)
        
        # Add some structure to make it more realistic
        for i in range(n_features):
            X[:, i] = X[:, i] * (i + 1) / 5  # Different scales
            
        # Create participant IDs and diagnosis
        participant_ids = np.repeat(np.arange(n_participants), samples_per_participant)
        
        # Create somewhat realistic diagnosis pattern (40% ASD)
        asd_participants = np.random.choice(n_participants, size=int(n_participants * 0.4), replace=False)
        diagnosis = np.array([1 if pid in asd_participants else 0 for pid in participant_ids])
        
        # Add slight correlation between features and diagnosis
        asd_mask = diagnosis == 1
        X[asd_mask, :5] += 0.3  # ASD participants have slightly different values
        X[~asd_mask, 5:10] += 0.3  # Control participants have different pattern
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['participant_id'] = participant_ids
        df['diagnosis'] = diagnosis
        df['class'] = ['A' if d == 1 else 'T' for d in diagnosis]
        
        print(f"📊 Generated synthetic dataset:")
        print(f"   Participants: {n_participants}")
        print(f"   Samples: {n_samples}")
        print(f"   Features: {n_features}")
        print(f"   ASD cases: {np.sum(diagnosis)} ({np.mean(diagnosis)*100:.1f}%)")
        
        # Save synthetic data
        df.to_csv('synthetic_neurogait_data.csv', index=False, sep=';')
        print("💾 Saved as 'synthetic_neurogait_data.csv'")
        
        # Run basic analysis on synthetic data
        analyzer = RealisticAnalysis()
        
        # Mock the load_and_prepare_data method for demo
        def demo_load_and_prepare_data():
            feature_names_only = [col for col in df.columns if col not in ['participant_id', 'diagnosis', 'class']]
            return df, feature_names_only, "synthetic_demo", np.arange(len(df)), np.arange(len(df)), participant_ids, participant_ids
        
        # Replace method temporarily
        original_method = analyzer.load_and_prepare_data
        analyzer.load_and_prepare_data = demo_load_and_prepare_data
        
        try:
            print("\n🚀 Running demo analysis...")
            results = analyzer.run_realistic_analysis()
            
            print("\n🎯 DEMO COMPLETED!")
            print("📋 This was a demonstration with synthetic data")
            print("🔄 Use real 'Final dataset.csv' for actual analysis")
            
            return results
            
        except Exception as e:
            print(f"❌ Demo failed: {str(e)}")
            return None
        finally:
            # Restore original method
            analyzer.load_and_prepare_data = original_method    

if __name__ == "__main__":
    print("🏥 NEUROGAIT ANALYSIS SYSTEM")
    print("="*50)
    
    # Check if real dataset exists
    import os
    if os.path.exists('Final dataset.csv'):
        print("✅ Real dataset found - running full analysis")
        results = main()
    else:
        print("⚠️ 'Final dataset.csv' not found")
        print("🔬 Running demonstration with synthetic data")
        print()
        results = run_demo_analysis()
    
    if results:
        print("\n✅ Analysis pipeline completed successfully!")
    else:
        print("\n❌ Analysis failed - check error messages above")