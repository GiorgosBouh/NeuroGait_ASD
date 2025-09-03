# enhanced_kg_features.py
import os
import logging
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EnhancedKGFeatureBuilder:
    """
    Lightweight helper για φόρτωση & προεπεξεργασία χαρακτηριστικών
    που χρησιμοποιούνται από τον NeuroGait KG builder ή/και το analysis script.
    Στόχος: να μην σπάει σε διαφορετικά ονόματα στηλών, encodings κλπ.
    """

    # Συγχρονισμένα χαρακτηριστικά με το KG builder (όσα υπάρχουν στο CSV)
    _SYNC_FEATURE_CANDIDATES: List[str] = [
        "mean HESHL", "mean SPELR", "mean SHWRL", "mean SHWRR", "mean ELHAL",
        "mean THHAR", "mean SPKNL", "mean SPKNR", "mean HIANR",
        "GaCT", "StaT", "SwiT",
        "mean-x-Midspain", "mean-y-Midspain", "mean-z-Midspain",
        "mean-x-SpineBase", "mean-y-SpineBase", "mean-z-SpineBase",
        "Velocity",
    ]

    def __init__(self, samples_per_participant: int = 8):
        self.samples_per_participant = samples_per_participant

    # --------------------------- IO & Column resolution ---------------------------

    def load_data(self, filepath: str = "Final dataset.csv") -> pd.DataFrame:
        """Load the dataset from CSV with robust encoding/decimal handling."""
        logger.info("📊 Loading dataset...")

        # 1) Προσπάθησε UTF-8 / αν όχι, δοκίμασε latin-1
        try:
            df = pd.read_csv(filepath, sep=";", decimal=",", encoding="utf-8")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(filepath, sep=";", decimal=",", encoding="latin-1")
            except Exception as e:
                logger.error(f"❌ Failed to read CSV '{filepath}': {e}")
                raise

        # 2) Μετατροπή αριθμητικών στηλών (εκτός της class)
        numeric_cols = [c for c in df.columns if c != "class"]
        for col in numeric_cols:
            if df[col].dtype == "object":
                try:
                    # Αφαίρεσε κενά, αντικατάστησε κόμμα με τελεία, ρίξε σε float
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.strip()
                        .str.replace(",", ".", regex=False)
                        .replace({"": np.nan})
                        .astype(float)
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Could not convert column '{col}' to float: {e}")

        # 3) Πρόσθεσε participant & diagnosis αν λείπουν
        # participant_id: βάσει index//samples_per_participant αν δεν υπάρχει ήδη
        if "participant_id" not in df.columns:
            try:
                df["participant_id"] = df.index // self.samples_per_participant
            except Exception as e:
                logger.error(f"❌ Failed while creating participant_id: {e}")
                raise

        # diagnosis από class: A->ASD, T->Typical (μόνο αν υπάρχει 'class')
        if "diagnosis" not in df.columns and "class" in df.columns:
            try:
                df["diagnosis"] = df["class"].map({"A": "ASD", "T": "Typical"})
            except Exception as e:
                logger.warning(f"⚠️ Could not map diagnosis from 'class': {e}")

        return df

    def _resolve_columns(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """
        Προσπαθεί να βρει βασικές στήλες με διάφορα πιθανά ονόματα.
        Επιστρέφει dict από λογικό όνομα -> πραγματικό όνομα στήλης ή None.
        """
        candidates = {
            "participant_id": [
                "participant_id", "ParticipantID", "participant", "PID", "pid",
                "subject_id", "SubjectID", "subject",
            ],
            "sample_id":      ["sample_id", "SampleID", "sample", "Sample", "id", "ID"],
            "diagnosis":      ["diagnosis", "Diagnosis", "label", "Label", "class", "Class", "group"],
            "data_split":     ["data_split", "split", "Split", "set", "Set", "split_group"],
            "augmentation":   ["augmentation_type", "augmentation", "Augmentation", "aug", "Aug"],
            "velocity":       ["Velocity", "velocity"],
        }
        cols = {}
        lower_map = {c.lower(): c for c in df.columns}
        for key, names in candidates.items():
            found = None
            for name in names:
                if name in df.columns:
                    found = name
                    break
                if name.lower() in lower_map:
                    found = lower_map[name.lower()]
                    break
            cols[key] = found
        return cols

    # --------------------------- Feature selection helpers ---------------------------

    def get_available_sync_features(self, df: pd.DataFrame) -> List[str]:
        """Επιστρέφει όσα από τα συγχρονισμένα χαρακτηριστικά υπάρχουν πραγματικά στο df."""
        present = [f for f in self._SYNC_FEATURE_CANDIDATES if f in df.columns]
        if not present:
            logger.warning("⚠️ No synchronized features found in DataFrame columns.")
        return present

    def select_features(self, df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        """Παίρνει subset από το df μόνο με όσα χαρακτηριστικά υπάρχουν."""
        existing = [c for c in feature_names if c in df.columns]
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            logger.info(f"ℹ️ Missing features skipped: {missing}")
        return df[existing].copy()

    # --------------------------- Public API used by analysis/KG ---------------------------

    def preprocess_features(
        self,
        df: pd.DataFrame,
        min_variance: float = 0.0,
    ) -> pd.DataFrame:
        """
        Βασικό preprocessing:
        - Κρατά μόνο τα διαθέσιμα συγχρονισμένα features
        - (προαιρετικά) πετά πολύ-χαμηλής διακύμανσης columns
        """
        features = self.get_available_sync_features(df)
        if not features:
            # Αν δεν υπάρχουν οι “γνωστές” στήλες, κράτα αριθμητικές (εκτός προφανών meta)
            numeric_df = df.select_dtypes(include=[np.number]).copy()
            meta_cols = {"participant_id", "diagnosis"}
            features = [c for c in numeric_df.columns if c not in meta_cols]
            work = numeric_df[features].copy()
        else:
            work = self.select_features(df, features)

        # low-variance filter (αν ζητηθεί)
        if min_variance > 0.0 and not work.empty:
            variances = work.var(numeric_only=True)
            keep = variances[variances >= min_variance].index.tolist()
            dropped = [c for c in work.columns if c not in keep]
            if dropped:
                logger.info(f"🧹 Dropped low-variance features: {dropped}")
            work = work[keep]

        return work

    def get_feature_matrix(
        self,
        df: pd.DataFrame,
        min_variance: float = 0.0,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
        """
        Επιστρέφει X, y (προαιρετικά), και meta (π.χ. participant_ids) για downstream χρήση.
        - X: numpy array των επιλεγμένων χαρακτηριστικών
        - y: labels αν υπάρχει στήλη diagnosis (ASD/Typical). Αλλιώς None.
        - meta: dict με 'participant_id', 'index'
        """
        cols = self._resolve_columns(df)
        # Features
        feat_df = self.preprocess_features(df, min_variance=min_variance)
        X = feat_df.to_numpy(dtype=float, copy=True)

        # Labels (αν υπάρχει diagnosis)
        y = None
        if cols["diagnosis"] and cols["diagnosis"] in df.columns:
            y = df[cols["diagnosis"]].to_numpy()

        # Meta
        participant_ids = (
            df[cols["participant_id"]].astype(str).to_numpy()
            if cols["participant_id"] and cols["participant_id"] in df.columns
            else np.array([str(i // self.samples_per_participant) for i in range(len(df))])
        )

        meta = {
            "participant_id": participant_ids,
            "index": np.arange(len(df)),
            "feature_names": np.array(feat_df.columns.tolist(), dtype=object),
        }
        return X, y, meta

    # --------------------------- Optional utility for KG builder ---------------------------

    def build_rows_for_graph(self, df: pd.DataFrame) -> List[Dict[str, object]]:
        """
        Ετοιμάζει “rows” για UNWIND στο Neo4j, αν χρειαστεί (participant/sample info).
        - sample_id: χρησιμοποιεί στήλη αν υπάρχει, αλλιώς συνθέτει S_<participant>_<rowindex>
        """
        cols = self._resolve_columns(df)
        rows = []
        for i, (_, r) in enumerate(df.iterrows()):
            pid = str(r[cols["participant_id"]]) if cols["participant_id"] else str(i // self.samples_per_participant)

            if cols["sample_id"] and cols["sample_id"] in r:
                sid = str(r[cols["sample_id"]])
            else:
                sid = f"S_{pid}_{i}"

            rows.append({
                "participant_id": pid,
                "diagnosis": (r[cols["diagnosis"]] if cols["diagnosis"] and cols["diagnosis"] in r else None),
                "split": (r[cols["data_split"]] if cols["data_split"] and cols["data_split"] in r else None),
                "augmentation": (r[cols["augmentation"]] if cols["augmentation"] and cols["augmentation"] in r else "original"),
                "sample_id": sid,
                "velocity": (
                    float(r[cols["velocity"]])
                    if cols["velocity"] and cols["velocity"] in r and pd.notna(r[cols["velocity"]])
                    else None
                ),
            })
        return rows


# --------------------------- Smoke test ---------------------------
if __name__ == "__main__":
    # Μικρό self-test ώστε να δεις ότι φορτώνει σωστά
    builder = EnhancedKGFeatureBuilder(samples_per_participant=8)
    path = os.getenv("NG_DATA", "Final dataset.csv")
    try:
        df0 = builder.load_data(path)
        X, y, meta = builder.get_feature_matrix(df0, min_variance=0.0)
        print("Loaded:", df0.shape, "| X:", X.shape, "| y:", None if y is None else y.shape)
        print("Features:", meta["feature_names"][:10], "...")
    except Exception as e:
        print("❌ Smoke test failed:", e)

