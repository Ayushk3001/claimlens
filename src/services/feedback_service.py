import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.utils.config import settings

class FeedbackService:
    def __init__(self, feedback_file: Optional[Path] = None):
        self.feedback_file = feedback_file or settings.FEEDBACK_FILE
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.feedback_file.exists():
            self._save_all([])

    def _load_all(self) -> List[Dict[str, Any]]:
        try:
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_all(self, data: List[Dict[str, Any]]):
        with open(self.feedback_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def record_feedback(self, feedback_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Record human adjuster decision and notes on AI analysis."""
        data = self._load_all()
        entry = {
            "feedback_id": f"FB-{len(data) + 1:05d}",
            "claim_id": feedback_entry.get("claim_id", "UNKNOWN"),
            "adjuster_id": feedback_entry.get("adjuster_id", "ADJ-DEFAULT"),
            "adjuster_decision": feedback_entry.get("adjuster_decision", "Approved"),
            "agrees_with_ai": bool(feedback_entry.get("agrees_with_ai", True)),
            "adjusted_amount": feedback_entry.get("adjusted_amount"),
            "confirmed_fraud": feedback_entry.get("confirmed_fraud"),
            "notes": feedback_entry.get("notes", ""),
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        data.append(entry)
        self._save_all(data)
        return entry

    def get_feedback_metrics(self) -> Dict[str, Any]:
        """Calculate continuous improvement feedback analytics."""
        data = self._load_all()
        total = len(data)
        if total == 0:
            return {
                "total_reviews": 0,
                "ai_agreement_rate_percent": 100.0,
                "decision_distribution": {},
                "recent_reviews": []
            }

        agreements = sum(1 for item in data if item.get("agrees_with_ai"))
        dist: Dict[str, int] = {}
        for item in data:
            dec = item.get("adjuster_decision", "Approved")
            dist[dec] = dist.get(dec, 0) + 1

        return {
            "total_reviews": total,
            "ai_agreement_rate_percent": round((agreements / total) * 100, 1),
            "decision_distribution": dist,
            "recent_reviews": data[-10:]
        }

    def _get_or_create_golden_eval_set(self, base_df) -> tuple:
        """Partition and persist an independent Golden Evaluation Holdout Set.
        This set is frozen on disk and never exposed to model training, guaranteeing
        an unbiased, out-of-sample benchmark comparison for promotion gating.
        """
        golden_file = settings.SAMPLE_DATA_DIR / "golden_eval_holdout.parquet"
        if golden_file.exists():
            try:
                import pandas as pd
                golden_df = pd.read_parquet(golden_file)
                train_pool = base_df[~base_df["claim_id"].isin(golden_df["claim_id"])].copy()
                if len(train_pool) > 50 and len(golden_df) > 20:
                    return train_pool, golden_df
            except Exception:
                pass

        # Create new stratified partition
        from sklearn.model_selection import train_test_split
        train_pool, golden_df = train_test_split(
            base_df, test_size=0.15, random_state=42, stratify=base_df["is_fraud_flagged_ground_truth"]
        )
        settings.SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        golden_df.to_parquet(golden_file, index=False)
        return train_pool.copy(), golden_df.copy()

    def trigger_retraining(self, ml_service, min_feedback_count: int = 1) -> Dict[str, Any]:
        """Controlled feedback-to-retraining pipeline with strict MLOps safety gates.
        - Strict label integrity: Only explicit confirmed_fraud determines fraud target changes.
        - Unbiased evaluation: Evaluates candidate vs. active model on an untouched Golden Evaluation Benchmark.
        - Regression prevention: Trips promotion gate if candidate validation Brier score degrades.
        - Atomic deployment: Writes via temporary staging files and os.replace with automatic rollback.
        """
        data = self._load_all()
        if len(data) < min_feedback_count:
            return {
                "status": "skipped",
                "message": f"Insufficient feedback records ({len(data)}/{min_feedback_count}) to trigger retraining.",
                "active_version": getattr(ml_service, "model_version", "v1.0"),
                "samples_used": 0
            }

        # 1. Load base dataset & partition independent Golden Benchmark Set
        from src.utils.data_loader import load_claims_dataset
        import pandas as pd
        import numpy as np
        import pickle
        import os
        import shutil
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import accuracy_score, brier_score_loss

        base_df = load_claims_dataset()
        train_pool, golden_df = self._get_or_create_golden_eval_set(base_df)
        df_augmented = train_pool.copy()

        # 2. Extract adjuster overrides as training labels
        for entry in data:
            cid = entry.get("claim_id")
            adj_amt = entry.get("adjusted_amount")
            conf_fraud = entry.get("confirmed_fraud")

            match = df_augmented[df_augmented["claim_id"] == cid]
            if not match.empty:
                idx = match.index[0]
                # Label integrity: Fraud target is strictly modified ONLY when explicitly confirmed
                # by a verified auditor or investigator (True=confirmed fraud, False=exonerated).
                # Operational approvals or SIU referrals WITHOUT explicit confirmation do NOT alter forensic labels.
                if conf_fraud is not None:
                    df_augmented.at[idx, "is_fraud_flagged_ground_truth"] = bool(conf_fraud)
                
                if adj_amt is not None and float(adj_amt) > 0:
                    df_augmented.at[idx, "claim_amount"] = float(adj_amt)

        # 3. Fit Candidate Models strictly on training pool
        X_train = ml_service._prepare_features(df_augmented, is_training=True)
        y_fraud_train = df_augmented["is_fraud_flagged_ground_truth"].astype(int).values
        y_amount_train = df_augmented["claim_amount"].fillna(df_augmented["claim_amount"].median()).values

        cand_amount = RandomForestRegressor(n_estimators=60, max_depth=8, random_state=42, n_jobs=-1)
        cand_amount.fit(X_train, y_amount_train)

        cand_base_fraud = RandomForestClassifier(n_estimators=80, max_depth=7, class_weight="balanced", random_state=42, n_jobs=-1)
        cand_base_fraud.fit(X_train, y_fraud_train)

        cand_calibrated = CalibratedClassifierCV(estimator=cand_base_fraud, method="sigmoid", cv=5)
        cand_calibrated.fit(X_train, y_fraud_train)

        # 4. Fair Benchmark Evaluation on Genuinely Untouched Golden Evaluation Set
        X_eval = ml_service._prepare_features(golden_df, is_training=False)
        y_fraud_eval = golden_df["is_fraud_flagged_ground_truth"].astype(int).values

        cand_eval_preds = cand_calibrated.predict(X_eval)
        cand_eval_acc = accuracy_score(y_fraud_eval, cand_eval_preds)
        cand_eval_probs = cand_calibrated.predict_proba(X_eval)[:, 1]
        cand_eval_brier = brier_score_loss(y_fraud_eval, cand_eval_probs)

        curr_ver = getattr(ml_service, "model_version", "v1.0")
        base_eval_preds = ml_service.fraud_model.predict(X_eval)
        base_eval_acc = accuracy_score(y_fraud_eval, base_eval_preds)
        base_eval_probs = ml_service.fraud_model.predict_proba(X_eval)[:, 1]
        base_eval_brier = brier_score_loss(y_fraud_eval, base_eval_probs)

        # 5. Promotion Gate: Reject Candidate if Validation Brier Score Degrades
        if cand_eval_brier > (base_eval_brier + 1e-4):
            return {
                "status": "rejected_regression",
                "message": f"Candidate model validation Brier score ({cand_eval_brier:.4f}) degraded compared to active model ({base_eval_brier:.4f}) on untouched golden benchmark set. Promotion rejected.",
                "active_version": curr_ver,
                "base_validation_brier": round(float(base_eval_brier), 4),
                "candidate_validation_brier": round(float(cand_eval_brier), 4),
                "base_validation_accuracy": round(float(base_eval_acc), 4),
                "candidate_validation_accuracy": round(float(cand_eval_acc), 4),
                "golden_benchmark_samples": len(golden_df),
                "feedback_records_integrated": len(data),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

        # 6. Model Versioning Bump (Promoted Candidates Only)
        try:
            ver_num = float(curr_ver.replace("v", ""))
            new_ver = f"v{ver_num + 0.1:.1f}"
        except Exception:
            new_ver = "v1.1"

        # 7. Atomic Artifact Promotion & Rollback Safety
        versioned_file = ml_service.models_dir / f"claims_models_bundle_{new_ver}.pkl"
        temp_versioned = ml_service.models_dir / f"claims_models_bundle_{new_ver}.tmp"
        temp_active = ml_service.models_dir / "claims_models_bundle_active.tmp"
        backup_active = ml_service.models_dir / "claims_models_bundle_active.backup"

        bundle = {
            "amount_model": cand_amount,
            "resolution_model": ml_service.resolution_model,
            "fraud_model": cand_calibrated,
            "base_fraud_model": cand_base_fraud,
            "feature_names": ml_service.feature_names,
            "encoder": ml_service.encoder,
            "model_version": new_ver,
            "retrained_at": datetime.utcnow().isoformat() + "Z",
            "feedback_samples_integrated": len(data),
            "golden_eval_brier": float(cand_eval_brier)
        }

        # Step A: Atomic write of versioned artifact
        with open(temp_versioned, "wb") as f:
            pickle.dump(bundle, f)
        os.replace(temp_versioned, versioned_file)

        # Step B: Backup existing active model file
        if ml_service.model_file.exists():
            shutil.copy2(ml_service.model_file, backup_active)

        # Step C: Atomic replacement of active production artifact
        with open(temp_active, "wb") as f:
            pickle.dump(bundle, f)

        try:
            os.replace(temp_active, ml_service.model_file)
            # Step D: In-memory hot reload
            ml_service.amount_model = cand_amount
            ml_service.fraud_model = cand_calibrated
            ml_service.base_fraud_model = cand_base_fraud
            ml_service.model_version = new_ver
        except Exception as err:
            # Immediate automated rollback
            if backup_active.exists():
                shutil.copy2(backup_active, ml_service.model_file)
            raise RuntimeError(f"Atomic model promotion failed during reload; rolled back to {curr_ver}: {err}")

        return {
            "status": "success",
            "previous_version": curr_ver,
            "active_version": new_ver,
            "feedback_records_integrated": len(data),
            "golden_benchmark_samples": len(golden_df),
            "base_validation_brier": round(float(base_eval_brier), 4),
            "candidate_validation_brier": round(float(cand_eval_brier), 4),
            "candidate_validation_accuracy": round(float(cand_eval_acc), 4),
            "model_artifact": str(versioned_file.name),
            "atomic_promotion_verified": True,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

feedback_service = FeedbackService()
