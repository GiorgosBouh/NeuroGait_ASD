
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import warnings
import logging
from datetime import datetime

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_and_prepare_data():
    """Load and prepare data with participant structure"""
    logger.info("📊 Loading data...")
    
    # Load data
    df = pd.read_csv('Final dataset.csv', sep=';', decimal=',')
    logger.info(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Convert target
    df['class'] = df['class'].map({'A': 1, 'T': 0})
    df = df.rename(columns={'class': 'diagnosis'})
    
    # Keep only mean features
    cols_to_keep = ['diagnosis']
    for col in df.columns:
        col_clean = col.strip()
        if (col_clean.startswith('mean-') and any(coord in col_clean for coord in ['-x-', '-y-', '-z-'])) or \
           (col_clean.startswith('mean ') and len(col_clean.split()) >= 2) or \
           col_clean.startswith('Rom') or \
           col_clean in ['MaxStLe', 'MaxStWi', 'StrLe', 'GaCT', 'StaT', 'SwiT', 'Velocity', 'HaTiLPos', 'HaTiRPos', 'MaxDBFE', 'MinDBFE', 'Threshold']:
            cols_to_keep.append(col)
    
    df_filtered = df[cols_to_keep]
    
    # Remove constant features
    for col in df_filtered.columns:
        if col != 'diagnosis' and df_filtered[col].nunique() <= 1:
            df_filtered = df_filtered.drop(columns=[col])
    
    logger.info(f"✅ Kept {len(df_filtered.columns)-1} features")
    return df_filtered

def participant_level_split(X, y, test_size=0.2, samples_per_participant=8):
    """Split at participant level to prevent leakage"""
    logger.info("🔧 Performing participant-level split...")
    
    n_samples = len(X)
    n_participants = n_samples // samples_per_participant
    
    # Create participant IDs
    participant_ids = np.repeat(range(n_participants), samples_per_participant)
    
    # Get one label per participant (they're all the same for each participant)
    participant_labels = y[::samples_per_participant].values
    
    # Split participants
    train_pids, test_pids = train_test_split(
        range(n_participants), 
        test_size=test_size, 
        stratify=participant_labels, 
        random_state=42
    )
    
    # Get sample indices
    train_mask = np.isin(participant_ids, train_pids)
    test_mask = np.isin(participant_ids, test_pids)
    
    X_train = X[train_mask].reset_index(drop=True)
    X_test = X[test_mask].reset_index(drop=True)
    y_train = y[train_mask].reset_index(drop=True)
    y_test = y[test_mask].reset_index(drop=True)
    
    logger.info(f"✅ Split: {len(train_pids)} train participants ({len(X_train)} samples)")
    logger.info(f"         {len(test_pids)} test participants ({len(X_test)} samples)")
    
    return X_train, X_test, y_train, y_test, train_pids

def simple_cv_with_participants(X_train, y_train, train_pids, samples_per_participant=8):
    """Simple participant-level CV"""
    logger.info("🔄 Creating participant-level CV...")
    
    # Get participant labels
    n_train_participants = len(train_pids)
    participant_labels = y_train[::samples_per_participant].values
    
    # Create CV splits
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_p_idx, val_p_idx) in enumerate(skf.split(range(n_train_participants), participant_labels)):
        # Get sample indices for this fold
        train_sample_indices = []
        val_sample_indices = []
        
        for p_idx in train_p_idx:
            start_idx = p_idx * samples_per_participant
            end_idx = start_idx + samples_per_participant
            train_sample_indices.extend(range(start_idx, end_idx))
        
        for p_idx in val_p_idx:
            start_idx = p_idx * samples_per_participant
            end_idx = start_idx + samples_per_participant
            val_sample_indices.extend(range(start_idx, end_idx))
        
        # Get fold data
        X_fold_train = X_train.iloc[train_sample_indices]
        X_fold_val = X_train.iloc[val_sample_indices]
        y_fold_train = y_train.iloc[train_sample_indices]
        y_fold_val = y_train.iloc[val_sample_indices]
        
        # Train fold model
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=2.0, reg_lambda=2.0,
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        )
        model.fit(X_fold_train, y_fold_train)
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
        fold_auc = roc_auc_score(y_fold_val, y_pred_proba)
        cv_scores.append(fold_auc)
        
        logger.info(f"   Fold {fold+1}: {len(train_p_idx)} train participants, {len(val_p_idx)} val participants, AUC: {fold_auc:.4f}")
    
    return cv_scores

def train_final_model(X_train, X_test, y_train, y_test):
    """Train final model"""
    logger.info("🚀 Training final model...")
    
    # Feature selection and scaling
    # Remove highly correlated features
    corr_matrix = X_train.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]
    
    X_train_decorr = X_train.drop(columns=to_drop)
    X_test_decorr = X_test.drop(columns=to_drop)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_decorr)
    X_test_scaled = scaler.transform(X_test_decorr)
    
    # Select features
    selector = SelectKBest(f_classif, k=min(25, X_train_scaled.shape[1]))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    logger.info(f"   Selected {X_train_selected.shape[1]} features after processing")
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=2.0, reg_lambda=2.0,
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    )
    model.fit(X_train_selected, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_selected)
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
    
    metrics = {
        'test_auc': roc_auc_score(y_test, y_pred_proba),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred),
        'test_recall': recall_score(y_test, y_pred),
        'test_f1': f1_score(y_test, y_pred),
    }
    
    return metrics

def main():
    """Main analysis"""
    logger.info("🔍 Starting SIMPLE Fixed NeuroGait Analysis")
    
    try:
        # Load data
        df = load_and_prepare_data()
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']
        
        # Participant-level split
        X_train, X_test, y_train, y_test, train_pids = participant_level_split(X, y)
        
        # Participant-level CV
        cv_scores = simple_cv_with_participants(X_train, y_train, train_pids)
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        logger.info(f"   CV AUC: {cv_mean:.4f} ± {cv_std:.4f}")
        
        # Train final model
        metrics = train_final_model(X_train, X_test, y_train, y_test)
        
        # Results
        logger.info("\n📊 FINAL RESULTS (NO PARTICIPANT LEAKAGE):")
        logger.info(f"   CV AUC:      {cv_mean:.4f} ± {cv_std:.4f}")
        logger.info(f"   Test AUC:    {metrics['test_auc']:.4f}")
        logger.info(f"   Accuracy:    {metrics['test_accuracy']:.4f}")
        logger.info(f"   Precision:   {metrics['test_precision']:.4f}")
        logger.info(f"   Recall:      {metrics['test_recall']:.4f}")
        logger.info(f"   F1-score:    {metrics['test_f1']:.4f}")
        
        if metrics['test_auc'] < 0.85:
            logger.info("✅ Realistic performance achieved!")
        else:
            logger.warning("⚠️  Still high - may have other issues")
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
