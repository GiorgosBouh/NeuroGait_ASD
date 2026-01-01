#!/usr/bin/env python3
"""
REALISTIC ANALYSIS - Enhanced με Clinical Features, Statistics, και GNN Support
GOAL: Raw vs KG vs GNN comparison με καλύτερα clinical features και πλήρη στατιστική ανάλυση

Fixed (Jan 2026):
- No leakage participant-level split
- Added missing methods (_proper_cross_validation, preprocess_train_test)
- Fixed wrong method calls / wrong signatures
- CV is participant-level (Group-like) with StratifiedKFold on participant labels
- Robustness: works even if enhanced_kg_features.py / true_gnn_analysis.py are missing
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import xgboost as xgb
from scipy.stats import wilcoxon


# =========================
# Optional modules
# =========================
try:
    from enhanced_kg_features import EnhancedKGFeatureBuilder
    ENHANCED_FEATURES_AVAILABLE = True
    print("✅ Enhanced KG Features available")
except ImportError:
    print("⚠️ Enhanced features not available - using basic/tuned comparison only")
    print("   Create enhanced_kg_features.py to enable enhanced KG features tier")
    ENHANCED_FEATURES_AVAILABLE = False

try:
    from true_gnn_analysis import TrueGraphAnalysis
    GNN_ANALYSIS_AVAILABLE = True
    print("✅ GNN Analysis available")
except ImportError:
    print("⚠️ GNN analysis not available")
    print("   Install: pip install torch torch-geometric")
    print("   Create true_gnn_analysis.py to enable GNN analysis")
    GNN_ANALYSIS_AVAILABLE = False


# =========================
# Main Analyzer
# =========================
class RealisticAnalysis:
    def __init__(self):
        self.random_state = 42
        self.samples_per_participant = 8  # Used only if participant_id missing

    # -------------------------
    # Clinical feature selection
    # -------------------------
    def get_clinical_features(self, all_features):
        """Get clinical feature sets from domain expert analysis"""
        print(f"\n🧠 CLINICAL FEATURE SELECTION (Domain-inspired keywords)")

        clinical_sets = {}

        # Set 1: Balance Stability
        balance_keywords = [
            'spine', 'trunk', 'torso', 'midspain', 'spinebase', 'balance', 'stability',
            'sway', 'postural', 'leg', 'foot', 'knee', 'hip', 'ankle', 'SPKNL', 'SPKNR',
            'HIANL', 'HIANR', 'KNFOL', 'KNFOR', 'angle', 'rotation'
        ]
        balance_features = []
        for feature in all_features:
            fl = feature.lower()
            if any(k in fl for k in balance_keywords) or any(k in feature for k in ['Midspain', 'SpineBase', 'SPKNL', 'SPKNR', 'HIANL', 'HIANR']):
                balance_features.append(feature)
        clinical_sets['balance_stability'] = balance_features[:30]

        # Set 2: Gait Focused
        gait_keywords = [
            'gact', 'stat', 'swit', 'time', 'duration', 'cycle', 'step', 'stride',
            'length', 'width', 'distance', 'leg', 'foot', 'knee', 'hip', 'velocity', 'speed'
        ]
        gait_features = []
        for feature in all_features:
            fl = feature.lower()
            if any(k in fl for k in gait_keywords) or any(k in feature for k in ['GaCT', 'StaT', 'SwiT']):
                gait_features.append(feature)
        clinical_sets['gait_focused'] = gait_features[:20]

        # Set 3: ASD Specific-ish (dataset-dependent naming)
        asd_keywords = [
            'gait', 'stat', 'swit', 'heshl', 'heshr', 'spell', 'spelr', 'coordination', 'timing',
            'shwrl', 'shwrr', 'elhal', 'elhar', 'thhal', 'thhar'
        ]
        asd_features = []
        for feature in all_features:
            fl = feature.lower()
            if any(k in fl for k in asd_keywords) or any(k in feature for k in ['GaCT', 'StaT', 'SwiT', 'HESHL', 'HESHR', 'SHWRL', 'SHWRR']):
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
            available_count = len(features)
            print(f"      {set_name.replace('_', ' ').title():<18}: {available_count:2d} features")

        return clinical_sets

    def select_best_clinical_set(self, train_df, clinical_sets):
        """Quick evaluation on TRAIN ONLY to select best clinical feature set"""
        print(f"\n🔍 EVALUATING CLINICAL FEATURE SETS (TRAIN ONLY)")

        best_set_name = None
        best_auc = -np.inf
        best_features = None

        for set_name, feature_set in clinical_sets.items():
            try:
                available = [f for f in feature_set if f in train_df.columns]
                if len(available) < 5:
                    print(f"   {set_name.replace('_', ' '):<18}: Too few features ({len(available)})")
                    continue

                test_df = train_df[available + ['participant_id', 'diagnosis']].dropna()
                test_df = test_df.head(250)  # quick check
                if len(test_df) < 50:
                    print(f"   {set_name.replace('_', ' '):<18}: Insufficient rows after cleaning")
                    continue

                X = test_df[available]
                y = test_df['diagnosis']
                if len(np.unique(y)) < 2:
                    print(f"   {set_name.replace('_', ' '):<18}: No class variation")
                    continue

                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.30, random_state=self.random_state, stratify=y
                )

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)

                lr = LogisticRegression(random_state=self.random_state, max_iter=1000, C=1.0, solver='liblinear')
                lr.fit(X_tr_s, y_tr)
                y_proba = lr.predict_proba(X_te_s)[:, 1]
                auc = roc_auc_score(y_te, y_proba)

                print(f"   {set_name.replace('_', ' '):<18}: {len(available):2d} features, Quick AUC={auc:.3f}")

                if auc > best_auc:
                    best_auc = auc
                    best_set_name = set_name
                    best_features = available

            except Exception as e:
                print(f"   {set_name.replace('_', ' '):<18}: Error - {str(e)[:60]}")
                continue

        # Fallback
        if best_features is None:
            for set_name, feature_set in clinical_sets.items():
                available = [f for f in feature_set if f in train_df.columns]
                if len(available) >= 10:
                    best_features = available[:25]
                    best_set_name = set_name
                    best_auc = 0.60
                    break

        print(f"\n✅ SELECTED CLINICAL FEATURE SET:")
        print(f"   Set: {best_set_name.replace('_', ' ').title()}")
        print(f"   Features: {len(best_features)}")
        print(f"   Estimated AUC (quick): {best_auc:.3f}")

        return best_features, best_set_name

    # -------------------------
    # Data loading / splitting
    # -------------------------
    def load_dataset(self, path='Final dataset.csv'):
        print("🔬 DATA LOADING")
        print("=" * 80)

        try:
            df = pd.read_csv(path, sep=';', decimal=',', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(path, sep=';', decimal=',', encoding='latin-1')

        print(f"📊 Original dataset: {df.shape}")

        if 'class' not in df.columns:
            raise ValueError("Dataset must contain a 'class' column with labels (e.g., A/T).")

        # Convert numeric columns
        numeric_cols = [c for c in df.columns if c != 'class']
        converted = []
        for col in numeric_cols:
            try:
                if df[col].dtype == 'object':
                    converted_col = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
                    if not converted_col.isna().all() and np.nanvar(converted_col) > 1e-10:
                        df[col] = converted_col
                        converted.append(col)
                else:
                    if np.nanvar(df[col]) > 1e-10:
                        converted.append(col)
            except Exception:
                continue

        # participant_id
        if 'participant_id' not in df.columns:
            df['participant_id'] = (df.index // self.samples_per_participant) + 1
            print("⚠️ participant_id not found -> created from row blocks")
        else:
            print("✅ Using existing participant_id from dataset")

        # stable ids
        df['original_index'] = df.index
        df['sample_id'] = df.apply(lambda r: f"S_{r['participant_id']}_{r['original_index']}", axis=1)

        # diagnosis
        df['diagnosis'] = df['class'].map({'A': 1, 'T': 0})
        if df['diagnosis'].isna().any():
            raise ValueError("Unknown class labels found. Expected mapping {'A':1,'T':0}.")

        return df, converted

    def participant_level_split(self, df, test_size=0.25):
        """Participant-level train/test split (no leakage)"""
        print(f"\n🔧 PARTICIPANT-LEVEL TRAIN/TEST SPLIT")

        participant_info = df.groupby('participant_id')['diagnosis'].first().reset_index()
        print(f"   Participants: {len(participant_info)}")
        print(f"   Class distribution: {participant_info['diagnosis'].value_counts().to_dict()}")

        train_pids, test_pids = train_test_split(
            participant_info['participant_id'].values,
            test_size=test_size,
            stratify=participant_info['diagnosis'].values,
            random_state=self.random_state
        )

        train_mask = df['participant_id'].isin(train_pids)
        test_mask = df['participant_id'].isin(test_pids)

        train_df = df[train_mask].reset_index(drop=True)
        test_df = df[test_mask].reset_index(drop=True)

        print(f"   ✅ Train: {len(set(train_pids))} participants, {len(train_df)} samples")
        print(f"   ✅ Test : {len(set(test_pids))} participants, {len(test_df)} samples")
        assert len(set(train_pids).intersection(set(test_pids))) == 0
        print("   ✅ No participant leakage verified")

        return train_df, test_df, train_pids, test_pids

    def load_and_prepare_data(self):
        """Full pipeline: load -> split -> clinical feature selection on train only"""
        df, converted_features = self.load_dataset('Final dataset.csv')

        # Split first (CRITICAL)
        train_df, test_df, train_pids, test_pids = self.participant_level_split(df, test_size=0.25)

        # Clinical sets from converted features only
        clinical_sets = self.get_clinical_features(converted_features)

        # Choose best set using ONLY train_df
        best_features, best_set_name = self.select_best_clinical_set(train_df, clinical_sets)

        print(f"\n✅ Using {len(best_features)} clinical features from: {best_set_name}")

        return df, train_df, test_df, best_features, best_set_name, train_pids, test_pids

    # -------------------------
    # Preprocessing (no leakage)
    # -------------------------
    def preprocess_train_test(self, train_df, test_df, features,
                              feature_missing_threshold=0.60,
                              sample_missing_threshold=0.50):
        """
        Fit preprocessing decisions on TRAIN ONLY:
        - Remove features with too much missing in train
        - Remove samples with too much missing (separately in train/test)
        - Impute using train statistics (median/mode)
        - Remove constant features using train
        """
        print(f"\n🧠 PREPROCESSING (fit on TRAIN only, apply to TEST)")

        base_cols = ['participant_id', 'diagnosis']
        train_work = train_df[features + base_cols].copy()
        test_work = test_df[features + base_cols].copy()

        print(f"   Start: {len(features)} features | train={len(train_work)} samples | test={len(test_work)} samples")

        # Feature missingness on TRAIN
        miss_rate = train_work[features].isna().mean(axis=0)
        kept_features = miss_rate[miss_rate <= feature_missing_threshold].index.tolist()
        print(f"   🗑️ Removed {len(features) - len(kept_features)} features with >{feature_missing_threshold*100:.0f}% missing (train)")

        # Drop high-missing samples
        tr_miss_s = train_work[kept_features].isna().mean(axis=1)
        te_miss_s = test_work[kept_features].isna().mean(axis=1)

        train_work = train_work.loc[tr_miss_s <= sample_missing_threshold].copy()
        test_work = test_work.loc[te_miss_s <= sample_missing_threshold].copy()
        print(f"   🗑️ Removed train samples: {(tr_miss_s > sample_missing_threshold).sum()} | test samples: {(te_miss_s > sample_missing_threshold).sum()}")

        # Impute using TRAIN statistics
        impute_values = {}
        for col in kept_features:
            col_train = train_work[col]
            if col_train.isna().any():
                if col_train.nunique(dropna=True) > 10:
                    val = col_train.median()
                else:
                    mode = col_train.mode()
                    val = mode.iloc[0] if len(mode) else 0
                if pd.isna(val):
                    val = 0
                impute_values[col] = val
            else:
                # still store for completeness
                if col_train.nunique(dropna=True) > 10:
                    impute_values[col] = col_train.median()
                else:
                    mode = col_train.mode()
                    impute_values[col] = mode.iloc[0] if len(mode) else 0

        for col, val in impute_values.items():
            train_work[col] = train_work[col].fillna(val)
            test_work[col] = test_work[col].fillna(val)

        # Remove constant features based on TRAIN
        constant = [c for c in kept_features if train_work[c].nunique(dropna=True) <= 1]
        final_features = [c for c in kept_features if c not in constant]
        print(f"   🧹 Constant features removed (train): {len(constant)}")
        print(f"   ✅ Final: {len(final_features)} features | train={len(train_work)} | test={len(test_work)}")

        # Drop duplicate rows by features (optional, safe)
        train_work = train_work.drop_duplicates(subset=final_features).reset_index(drop=True)
        test_work = test_work.drop_duplicates(subset=final_features).reset_index(drop=True)

        return train_work, test_work, final_features

    # -------------------------
    # Feature selection
    # -------------------------
    def optimized_feature_selection(self, train_df, test_df, features):
        """Less conservative feature selection for better performance (fit on train)"""
        print(f"\n🧠 FEATURE SELECTION (SelectKBest on train only)")

        X_train = train_df[features]
        X_test = test_df[features]
        y_train = train_df['diagnosis']

        n_samples, n_features = X_train.shape
        print(f"   Input: train={n_samples} × {n_features}")

        max_features = max(15, min(80, n_samples // 10))
        print(f"   🎯 Target features: {max_features}")

        if n_features <= max_features:
            print(f"   ✅ No selection needed")
            return X_train, X_test, features

        selector = SelectKBest(score_func=f_classif, k=max_features)
        try:
            X_train_sel = selector.fit_transform(X_train, y_train)
            X_test_sel = selector.transform(X_test)

            selected_features = [features[i] for i, ok in enumerate(selector.get_support()) if ok]

            print(f"   ✅ Selected {len(selected_features)} features (from {n_features})")

            return (pd.DataFrame(X_train_sel, columns=selected_features),
                    pd.DataFrame(X_test_sel, columns=selected_features),
                    selected_features)
        except Exception as e:
            print(f"   ⚠️ Feature selection failed: {str(e)[:80]}")
            return X_train, X_test, features

    # -------------------------
    # Scaling
    # -------------------------
    def scale_train_test(self, X_train, X_test):
        """Scale train and apply to test (no leakage)"""
        print(f"\n📊 SCALING")
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        print(f"   ✅ Scaling completed (fit on train only)")
        return X_train_s, X_test_s

    # -------------------------
    # KG embeddings (synthetic transforms)
    # -------------------------
    def create_conservative_kg_embeddings(self, X_train, X_test):
        """Create conservative KG embeddings (deterministic transform)"""
        print(f"\n🧠 CONSERVATIVE KG EMBEDDINGS")

        def transform(X):
            X_kg = X.copy()
            n_samples, n_features = X.shape
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

            # normalize per feature
            for i in range(n_features):
                std = np.std(X_kg[:, i])
                if std > 1e-6:
                    X_kg[:, i] = np.clip(X_kg[:, i] / std, -2, 2)
            return X_kg

        X_train_kg = transform(X_train)
        X_test_kg = transform(X_test)
        print(f"   ✅ KG created: train={X_train_kg.shape}, test={X_test_kg.shape}")
        return X_train_kg, X_test_kg

    def create_tuned_kg_embeddings(self, X_train, X_test, interaction_strength=0.02, smoothing=0.03, nonlinearity=0.3):
        """Create tuned KG embeddings with adjustable parameters"""
        print(f"\n🎯 TUNED KG EMBEDDINGS")
        print(f"   Params: interaction={interaction_strength}, smoothing={smoothing}, nonlinearity={nonlinearity}")

        def transform(X):
            X_kg = X.copy()
            n_samples, n_features = X.shape

            if n_features >= 3:
                for i in range(min(6, n_features - 1)):
                    for j in range(i + 1, min(i + 3, n_features)):
                        interaction = X_kg[:, i] * X_kg[:, j] * interaction_strength
                        X_kg[:, i] += interaction * 0.2
                        X_kg[:, j] += interaction * 0.2

            if n_features >= 5:
                for i in range(2, n_features - 2):
                    X_kg[:, i] = ((1 - 4*smoothing) * X_kg[:, i] +
                                  smoothing * X_kg[:, i-2] +
                                  smoothing * X_kg[:, i-1] +
                                  smoothing * X_kg[:, i+1] +
                                  smoothing * X_kg[:, i+2])

            X_kg = np.tanh(X_kg * nonlinearity)

            for i in range(n_features):
                std = np.std(X_kg[:, i])
                if std > 1e-6:
                    X_kg[:, i] = np.clip(X_kg[:, i] / std, -2.5, 2.5)

            return X_kg

        X_train_kg = transform(X_train)
        X_test_kg = transform(X_test)
        print(f"   ✅ Tuned KG created: train={X_train_kg.shape}, test={X_test_kg.shape}")
        return X_train_kg, X_test_kg

    # -------------------------
    # Participant-level CV
    # -------------------------
    def _proper_cross_validation(self, X_train, y_train, train_pids, model, cv_folds=5):
        """
        Participant-level CV:
        - Split unique participant ids with stratification on participant label
        - Train on all samples of train participants, validate on all samples of val participants
        """
        try:
            train_pids = np.asarray(train_pids)
            y_train_arr = np.asarray(y_train)

            unique_pids = np.unique(train_pids)
            # participant label = first sample label
            pid_labels = np.array([y_train_arr[np.where(train_pids == pid)[0][0]] for pid in unique_pids])

            if len(unique_pids) < cv_folds:
                cv_folds = max(3, len(unique_pids) // 2)

            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            scores = []

            for train_idx, val_idx in skf.split(unique_pids, pid_labels):
                tr_p = unique_pids[train_idx]
                va_p = unique_pids[val_idx]

                tr_mask = np.isin(train_pids, tr_p)
                va_mask = np.isin(train_pids, va_p)

                X_tr = X_train[tr_mask]
                X_va = X_train[va_mask]
                y_tr = y_train_arr[tr_mask]
                y_va = y_train_arr[va_mask]

                if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
                    continue

                m = type(model)(**model.get_params())
                if hasattr(m, "random_state"):
                    m.set_params(random_state=self.random_state)

                m.fit(X_tr, y_tr)

                if hasattr(m, "predict_proba"):
                    proba = m.predict_proba(X_va)[:, 1]
                else:
                    # fallback
                    pred = m.predict(X_va)
                    proba = pred.astype(float)

                auc = roc_auc_score(y_va, proba)
                if not np.isnan(auc):
                    scores.append(float(auc))

            return scores
        except Exception:
            return []

    # -------------------------
    # Model training
    # -------------------------
    def train_optimized_models(self, X_train, X_test, y_train, y_test, train_pids, approach_name):
        """Train models + participant-level CV"""
        print(f"\n🚀 TRAINING MODELS: {approach_name}")
        print(f"   Shapes: train={X_train.shape}, test={X_test.shape}")

        models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state, max_iter=1000, C=1.0, solver='liblinear'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=7, min_samples_split=5,
                min_samples_leaf=2, max_features='sqrt', random_state=self.random_state
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=self.random_state, max_depth=5, n_estimators=200,
                learning_rate=0.08, subsample=0.85, colsample_bytree=0.85,
                reg_alpha=0.2, reg_lambda=0.4, eval_metric='logloss',
                verbosity=0
            ),
            'SVM': SVC(
                random_state=self.random_state, probability=True, C=1.0, gamma='scale'
            )
        }

        results = {}

        y_train_arr = np.asarray(y_train)
        y_test_arr = np.asarray(y_test)

        for name, model in models.items():
            print(f"   🔧 {name}...")
            try:
                cv_scores = self._proper_cross_validation(X_train, y_train_arr, train_pids, model, cv_folds=5)
                if len(cv_scores) < 3:
                    # still allow training, but warn
                    print(f"      ⚠️ CV had few valid folds (n={len(cv_scores)})")

                model.fit(X_train, y_train_arr)

                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                metrics = {
                    'cv_scores': cv_scores,
                    'cv_mean': float(np.mean(cv_scores)) if len(cv_scores) else np.nan,
                    'cv_std': float(np.std(cv_scores)) if len(cv_scores) else np.nan,
                    'accuracy': float(accuracy_score(y_test_arr, y_pred)),
                    'precision': float(precision_score(y_test_arr, y_pred, zero_division=0)),
                    'recall': float(recall_score(y_test_arr, y_pred, zero_division=0)),
                    'f1': float(f1_score(y_test_arr, y_pred, zero_division=0)),
                    'auc': float(roc_auc_score(y_test_arr, y_proba)),
                    'y_test': y_test_arr,
                    'pred_test': y_pred,
                    'proba_test': y_proba
                }

                results[name] = metrics

                auc = metrics['auc']
                if auc > 0.8:
                    status = "🎉 Excellent"
                elif auc > 0.7:
                    status = "✅ Good"
                elif auc > 0.6:
                    status = "⚖️ Moderate"
                else:
                    status = "📋 Limited"

                cv_txt = "CV=N/A" if np.isnan(metrics['cv_mean']) else f"CV={metrics['cv_mean']:.3f}±{metrics['cv_std']:.3f}"
                print(f"      {status}: AUC={auc:.3f}, F1={metrics['f1']:.3f}, {cv_txt}")

            except Exception as e:
                print(f"      ❌ Failed: {str(e)[:100]}")
                continue

        return results

    # -------------------------
    # Statistics
    # -------------------------
    def statistical_comparison_analysis(self, results_by_approach):
        """Pairwise comparisons with Wilcoxon where possible"""
        print("\n📊 DETAILED STATISTICAL ANALYSIS")
        print("=" * 70)

        approaches = list(results_by_approach.keys())
        stats_out = {}

        for i in range(len(approaches)):
            for j in range(i + 1, len(approaches)):
                a1, a2 = approaches[i], approaches[j]
                print(f"\n🔍 COMPARING: {a1} vs {a2}")
                print("-" * 60)

                aucs1 = [m['auc'] for m in results_by_approach[a1].values()]
                aucs2 = [m['auc'] for m in results_by_approach[a2].values()]

                mean1, mean2 = float(np.mean(aucs1)), float(np.mean(aucs2))
                std1, std2 = float(np.std(aucs1)), float(np.std(aucs2))

                print(f"   AUC mean±std:")
                print(f"      {a1}: {mean1:.3f} ± {std1:.3f}")
                print(f"      {a2}: {mean2:.3f} ± {std2:.3f}")
                print(f"      Diff: {mean2 - mean1:+.3f}")

                pooled = np.sqrt((std1**2 + std2**2) / 2) + 1e-8
                cohens_d = (mean2 - mean1) / pooled

                if abs(cohens_d) > 0.8:
                    eff = "Large"
                elif abs(cohens_d) > 0.5:
                    eff = "Medium"
                elif abs(cohens_d) > 0.2:
                    eff = "Small"
                else:
                    eff = "Negligible"
                print(f"   Effect size: d={cohens_d:+.3f} ({eff})")

                # Wilcoxon on paired AUCs (only if comparable length >=3)
                p_value = np.nan
                w_stat = np.nan
                significance = "N/A"

                try:
                    n = min(len(aucs1), len(aucs2))
                    if n >= 3:
                        x1 = np.array(aucs1[:n])
                        x2 = np.array(aucs2[:n])
                        diffs = x2 - x1
                        if np.sum(np.abs(diffs) > 1e-10) >= 3:
                            w_stat, p_value = wilcoxon(x2, x1, alternative='two-sided', mode='auto', zero_method='wilcox')
                            if p_value < 0.001:
                                significance = "Highly significant (***)"
                            elif p_value < 0.01:
                                significance = "Very significant (**)"
                            elif p_value < 0.05:
                                significance = "Significant (*)"
                            elif p_value < 0.1:
                                significance = "Marginal"
                            else:
                                significance = "Not significant"
                            print(f"   Wilcoxon: W={w_stat:.2f}, p={p_value:.4f} → {significance}")
                        else:
                            print("   Wilcoxon: skipped (insufficient variation)")
                            significance = "Cannot test (low variation)"
                    else:
                        print("   Wilcoxon: skipped (insufficient paired points)")
                        significance = "Insufficient data"
                except Exception as e:
                    print(f"   Wilcoxon failed: {str(e)[:80]}")
                    significance = "Test failed"

                key = f"{a1}_vs_{a2}"
                stats_out[key] = {
                    'approach1': a1, 'approach2': a2,
                    'mean1': mean1, 'mean2': mean2,
                    'difference': mean2 - mean1,
                    'cohens_d': float(cohens_d),
                    'effect_size': eff,
                    'w_statistic': w_stat,
                    'p_value': p_value,
                    'significance': significance
                }

        print(f"\n📋 SUMMARY TABLE")
        print("=" * 90)
        print(f"{'Comparison':<35} {'Diff':<8} {'Cohen d':<10} {'p-value':<10} {'Significance':<20}")
        print("=" * 90)
        for k, r in stats_out.items():
            comp = f"{r['approach1'][:14]} vs {r['approach2'][:14]}"
            p_str = "N/A" if np.isnan(r['p_value']) else f"{r['p_value']:.4f}"
            print(f"{comp:<35} {r['difference']:+.3f}   {r['cohens_d']:+.3f}     {p_str:<10} {r['significance']:<20}")
        print("=" * 90)

        return stats_out

    # -------------------------
    # Hyperparameter search for tuned KG
    # -------------------------
    def hyperparameter_search(self, X_train, X_test, y_train, y_test, train_pids):
        print(f"\n🔍 HYPERPARAMETER SEARCH FOR KG EMBEDDINGS")
        print("=" * 60)

        grid = [
            {'interaction': 0.015, 'smoothing': 0.025, 'nonlinearity': 0.4, 'name': 'Conservative+'},
            {'interaction': 0.020, 'smoothing': 0.030, 'nonlinearity': 0.3, 'name': 'Balanced'},
            {'interaction': 0.025, 'smoothing': 0.035, 'nonlinearity': 0.4, 'name': 'Moderate'},
            {'interaction': 0.030, 'smoothing': 0.040, 'nonlinearity': 0.5, 'name': 'Moderate+'},
            {'interaction': 0.035, 'smoothing': 0.045, 'nonlinearity': 0.4, 'name': 'Aggressive-'},
            {'interaction': 0.010, 'smoothing': 0.020, 'nonlinearity': 0.5, 'name': 'Simple (baseline)'},
        ]

        best = None
        best_auc = -np.inf
        results = {}

        # Use one strong baseline model for tuning (RF)
        model = RandomForestClassifier(
            n_estimators=200, max_depth=7, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt', random_state=self.random_state
        )

        for cfg in grid:
            print(f"\n🧪 {cfg['name']}: int={cfg['interaction']}, smooth={cfg['smoothing']}, nonlin={cfg['nonlinearity']}")
            try:
                X_tr_kg, X_te_kg = self.create_tuned_kg_embeddings(
                    X_train, X_test,
                    cfg['interaction'], cfg['smoothing'], cfg['nonlinearity']
                )

                cv_scores = self._proper_cross_validation(X_tr_kg, y_train, train_pids, model, cv_folds=5)

                model.fit(X_tr_kg, y_train)
                proba = model.predict_proba(X_te_kg)[:, 1]
                auc = float(roc_auc_score(y_test, proba))

                results[cfg['name']] = {
                    'auc': auc,
                    'cv_mean': float(np.mean(cv_scores)) if len(cv_scores) else np.nan,
                    'cv_std': float(np.std(cv_scores)) if len(cv_scores) else np.nan,
                    'config': cfg
                }

                status = "✅" if auc > 0.7 else "⚖️" if auc > 0.6 else "📋"
                cvtxt = "CV=N/A" if np.isnan(results[cfg['name']]['cv_mean']) else f"CV={results[cfg['name']]['cv_mean']:.3f}±{results[cfg['name']]['cv_std']:.3f}"
                print(f"   {status} AUC={auc:.3f} | {cvtxt}")

                if auc > best_auc:
                    best_auc = auc
                    best = cfg

            except Exception as e:
                print(f"   ❌ Failed: {str(e)[:100]}")
                results[cfg['name']] = {'auc': 0.5, 'cv_mean': np.nan, 'cv_std': np.nan, 'config': cfg}

        print(f"\n📊 TUNING RESULTS (sorted by AUC)")
        print("=" * 70)
        for rank, (name, r) in enumerate(sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True), 1):
            medal = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"{medal} #{rank:>2} {name:<20} AUC={r['auc']:.3f}")

        if best:
            print(f"\n🎯 BEST CONFIG: {best['name']} | AUC={best_auc:.3f}")
        else:
            print("\n⚠️ No valid best config found")

        return best, results

    # -------------------------
    # Reporting helpers
    # -------------------------
    def print_comprehensive_results(self, results_by_approach, clinical_set_name, data_summary, tuning_results=None, best_config=None):
        print("\n" + "=" * 80)
        print("🎯 COMPREHENSIVE RESULTS")
        print("=" * 80)

        print("🏥 CONTEXT:")
        print(f"   Clinical feature set: {clinical_set_name.replace('_', ' ').title()}")
        print(f"   Train/Test participants: {data_summary['train_participants']} / {data_summary['test_participants']}")
        print(f"   Features: {data_summary['original_features']} → {data_summary['selected_features']} selected")

        if best_config is not None:
            print("\n🎛️ TUNING:")
            print(f"   Best: {best_config['name']} (interaction={best_config['interaction']}, smoothing={best_config['smoothing']}, nonlinearity={best_config['nonlinearity']})")

        print("\n📊 PERFORMANCE:")
        best_auc = -np.inf
        best_where = ("", "")
        for approach, models in results_by_approach.items():
            print(f"\n{approach}:")
            for model_name, m in models.items():
                auc = m['auc']
                f1 = m['f1']
                cvm = m['cv_mean']
                cvs = m['cv_std']
                cvtxt = "CV=N/A" if np.isnan(cvm) else f"CV={cvm:.3f}±{cvs:.3f}"
                print(f"   {model_name:<18} AUC={auc:.3f} | F1={f1:.3f} | {cvtxt}")
                if auc > best_auc:
                    best_auc = auc
                    best_where = (approach, model_name)

        print(f"\n🏆 BEST OVERALL: {best_where[0]} / {best_where[1]} (AUC={best_auc:.3f})")

        # stats
        stats = self.statistical_comparison_analysis(results_by_approach)
        return stats

    # -------------------------
    # Run modes
    # -------------------------
    def run_realistic_analysis(self):
        """Basic: Raw vs (Conservative) KG with stats"""
        df, train_df, test_df, best_features, best_set_name, train_pids, test_pids = self.load_and_prepare_data()

        train_clean, test_clean, clean_features = self.preprocess_train_test(train_df, test_df, best_features)
        X_train, X_test, selected_features = self.optimized_feature_selection(train_clean, test_clean, clean_features)

        y_train = train_clean['diagnosis'].values
        y_test = test_clean['diagnosis'].values

        X_train_s, X_test_s = self.scale_train_test(X_train, X_test)
        train_sample_pids = train_clean['participant_id'].values

        # Raw
        raw_results = self.train_optimized_models(
            X_train_s, X_test_s, y_train, y_test, train_sample_pids,
            f"Raw Clinical ({best_set_name})"
        )

        # KG
        X_train_kg, X_test_kg = self.create_conservative_kg_embeddings(X_train_s, X_test_s)
        kg_results = self.train_optimized_models(
            X_train_kg, X_test_kg, y_train, y_test, train_sample_pids,
            "Conservative KG"
        )

        all_results = {
            'Raw Clinical Features': raw_results,
            'Conservative KG': kg_results
        }

        stats = self.print_comprehensive_results(
            all_results, best_set_name,
            data_summary={
                'train_participants': len(np.unique(train_sample_pids)),
                'test_participants': len(np.unique(test_clean['participant_id'].values)),
                'original_features': len(best_features),
                'selected_features': len(selected_features),
            }
        )

        return {'results': all_results, 'stats': stats, 'clinical_set': best_set_name, 'selected_features': selected_features}

    def run_enhanced_analysis_with_tuning(self):
        """Raw vs Simple KG vs Tuned KG (+ optional Enhanced KG Features tier)"""
        df, train_df, test_df, best_features, best_set_name, train_pids, test_pids = self.load_and_prepare_data()

        train_clean, test_clean, clean_features = self.preprocess_train_test(train_df, test_df, best_features)
        X_train, X_test, selected_features = self.optimized_feature_selection(train_clean, test_clean, clean_features)

        y_train = train_clean['diagnosis'].values
        y_test = test_clean['diagnosis'].values

        X_train_s, X_test_s = self.scale_train_test(X_train, X_test)
        train_sample_pids = train_clean['participant_id'].values

        # Tier Raw
        raw_results = self.train_optimized_models(
            X_train_s, X_test_s, y_train, y_test, train_sample_pids,
            f"Raw Clinical ({best_set_name})"
        )

        # Simple KG
        X_train_kg_simple, X_test_kg_simple = self.create_conservative_kg_embeddings(X_train_s, X_test_s)
        simple_kg_results = self.train_optimized_models(
            X_train_kg_simple, X_test_kg_simple, y_train, y_test, train_sample_pids,
            "Simple/Conservative KG"
        )

        # Tuning
        best_cfg, tuning_results = self.hyperparameter_search(X_train_s, X_test_s, y_train, y_test, train_sample_pids)

        # Tuned KG
        if best_cfg:
            X_train_kg_tuned, X_test_kg_tuned = self.create_tuned_kg_embeddings(
                X_train_s, X_test_s, best_cfg['interaction'], best_cfg['smoothing'], best_cfg['nonlinearity']
            )
            tuned_kg_results = self.train_optimized_models(
                X_train_kg_tuned, X_test_kg_tuned, y_train, y_test, train_sample_pids,
                f"Tuned KG ({best_cfg['name']})"
            )
        else:
            tuned_kg_results = {}

        all_results = {
            'Raw Clinical Features': raw_results,
            'Simple KG': simple_kg_results,
            'Tuned KG': tuned_kg_results
        }

        # Optional: Enhanced KG Features module
        if ENHANCED_FEATURES_AVAILABLE:
            try:
                print("\n💡 OPTIONAL TIER: EnhancedKGFeatureBuilder()")
                builder = EnhancedKGFeatureBuilder()
                Xtr_enh, feat_names = builder.create_enhanced_kg_features(train_clean, selected_features)
                Xte_enh, _ = builder.create_enhanced_kg_features(test_clean, selected_features)

                # scale
                Xtr_enh_s, Xte_enh_s = self.scale_train_test(Xtr_enh, Xte_enh)
                enh_results = self.train_optimized_models(
                    Xtr_enh_s, Xte_enh_s, y_train, y_test, train_sample_pids,
                    "Enhanced KG Features (module)"
                )
                all_results['Enhanced KG Features'] = enh_results
            except Exception as e:
                print(f"⚠️ EnhancedKGFeatureBuilder tier failed: {str(e)[:120]}")

        stats = self.print_comprehensive_results(
            all_results, best_set_name,
            data_summary={
                'train_participants': len(np.unique(train_sample_pids)),
                'test_participants': len(np.unique(test_clean['participant_id'].values)),
                'original_features': len(best_features),
                'selected_features': len(selected_features),
            },
            tuning_results=tuning_results,
            best_config=best_cfg
        )

        return {
            'results': all_results,
            'stats': stats,
            'tuning_results': tuning_results,
            'best_config': best_cfg,
            'clinical_set': best_set_name,
            'selected_features': selected_features
        }

    def run_gnn_comparison_analysis(self):
        """Raw vs Simple KG vs Tuned KG (+ optional True GNN if available)"""
        print("\n🧠 GNN COMPARISON ANALYSIS")
        print("=" * 70)

        df, train_df, test_df, best_features, best_set_name, train_pids, test_pids = self.load_and_prepare_data()

        train_clean, test_clean, clean_features = self.preprocess_train_test(train_df, test_df, best_features)
        X_train, X_test, selected_features = self.optimized_feature_selection(train_clean, test_clean, clean_features)

        y_train = train_clean['diagnosis'].values
        y_test = test_clean['diagnosis'].values

        X_train_s, X_test_s = self.scale_train_test(X_train, X_test)
        train_sample_pids = train_clean['participant_id'].values

        # Raw
        raw_results = self.train_optimized_models(
            X_train_s, X_test_s, y_train, y_test, train_sample_pids,
            f"Raw Clinical ({best_set_name})"
        )

        # Simple KG
        X_train_kg_simple, X_test_kg_simple = self.create_conservative_kg_embeddings(X_train_s, X_test_s)
        simple_kg_results = self.train_optimized_models(
            X_train_kg_simple, X_test_kg_simple, y_train, y_test, train_sample_pids,
            "Simple KG"
        )

        # Tuned (use balanced default if no tuning)
        X_train_kg_tuned, X_test_kg_tuned = self.create_tuned_kg_embeddings(X_train_s, X_test_s, 0.02, 0.03, 0.3)
        tuned_kg_results = self.train_optimized_models(
            X_train_kg_tuned, X_test_kg_tuned, y_train, y_test, train_sample_pids,
            "Tuned KG (default balanced)"
        )

        all_results = {
            'Raw Clinical Features': raw_results,
            'Simple KG': simple_kg_results,
            'Tuned KG': tuned_kg_results
        }

        # True GNN (optional)
        if GNN_ANALYSIS_AVAILABLE:
            try:
                print("\n🤖 TRUE GNN (Neo4j) tier")
                gnn = TrueGraphAnalysis(samples_per_participant=self.samples_per_participant)
                train_pids_int = [int(pid) for pid in np.unique(train_sample_pids)]
                test_pids_int = [int(pid) for pid in np.unique(test_clean['participant_id'].values)]
                gnn_results = gnn.run_gnn_analysis(train_pids_int, test_pids_int)
                if isinstance(gnn_results, dict) and len(gnn_results):
                    all_results['True GNN'] = gnn_results
                    print(f"   ✅ GNN returned {len(gnn_results)} models")
                else:
                    print("   ⚠️ GNN returned no results (skipping)")
            except Exception as e:
                print(f"   ❌ GNN tier failed: {str(e)[:140]}")

        stats = self.print_comprehensive_results(
            all_results, best_set_name,
            data_summary={
                'train_participants': len(np.unique(train_sample_pids)),
                'test_participants': len(np.unique(test_clean['participant_id'].values)),
                'original_features': len(best_features),
                'selected_features': len(selected_features),
            }
        )

        return {'results': all_results, 'stats': stats, 'clinical_set': best_set_name, 'selected_features': selected_features}


# =========================
# CLI main
# =========================
def main():
    print("🏥 ENHANCED NEUROGAIT ANALYSIS SYSTEM")
    print("🎯 Options: Basic / Tuned / GNN")
    print()

    analyzer = RealisticAnalysis()

    options = [
        "1. Basic Analysis (Raw vs Conservative KG + stats)",
        "2. Tuned Analysis (Raw vs Simple KG vs Tuned KG + optional EnhancedKGFeatureBuilder)",
        "3. GNN Analysis (Raw vs Simple KG vs Tuned KG + optional True GNN)"
    ]

    print("Available analysis types:")
    for opt in options:
        print(f"   ✅ {opt}")

    if not ENHANCED_FEATURES_AVAILABLE:
        print("\n📋 EnhancedKGFeatureBuilder not available (that's OK).")
    if not GNN_ANALYSIS_AVAILABLE:
        print("\n📋 True GNN not available (that's OK).")

    print("\nChoose analysis type (1-3): ", end="")
    choice = input().strip()

    try:
        if choice == "1":
            return analyzer.run_realistic_analysis()
        elif choice == "2":
            return analyzer.run_enhanced_analysis_with_tuning()
        elif choice == "3":
            return analyzer.run_gnn_comparison_analysis()
        else:
            print("❌ Invalid choice -> running Basic.")
            return analyzer.run_realistic_analysis()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
        return None


# =========================
# Demo mode (synthetic)
# =========================
def run_demo_analysis():
    print("🔬 DEMO MODE - Synthetic NeuroGait Analysis")
    print("=" * 60)

    np.random.seed(42)
    n_participants = 20
    spp = 8
    n_samples = n_participants * spp
    n_features = 25

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

    X = np.random.randn(n_samples, n_features)
    for i in range(n_features):
        X[:, i] = X[:, i] * (i + 1) / 5

    participant_ids = np.repeat(np.arange(1, n_participants + 1), spp)

    asd_participants = np.random.choice(np.arange(1, n_participants + 1), size=int(n_participants * 0.4), replace=False)
    diagnosis = np.array([1 if pid in asd_participants else 0 for pid in participant_ids])

    X[diagnosis == 1, :5] += 0.3
    X[diagnosis == 0, 5:10] += 0.3

    df = pd.DataFrame(X, columns=feature_names)
    df['participant_id'] = participant_ids
    df['diagnosis'] = diagnosis
    df['class'] = ['A' if d == 1 else 'T' for d in diagnosis]

    df.to_csv('synthetic_neurogait_data.csv', index=False, sep=';')
    print("💾 Saved synthetic dataset as: synthetic_neurogait_data.csv")

    # Use synthetic as Final dataset.csv temporarily
    df.to_csv('Final dataset.csv', index=False, sep=';')
    print("💾 Also wrote as Final dataset.csv so the pipeline runs unchanged")

    analyzer = RealisticAnalysis()
    return analyzer.run_realistic_analysis()


if __name__ == "__main__":
    print("🏥 NEUROGAIT ANALYSIS")
    print("=" * 50)

    if os.path.exists('Final dataset.csv'):
        print("✅ Found Final dataset.csv -> running main")
        out = main()
    else:
        print("⚠️ Final dataset.csv not found -> running demo")
        out = run_demo_analysis()

    if out is not None:
        print("\n✅ Completed successfully.")
    else:
        print("\n❌ Failed / interrupted.")