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

    def trigger_retraining(self, ml_service, min_feedback_count: int = 1) -> Dict[str, Any]:
        """Controlled feedback-to-retraining pipeline with strict MLOps safety gates.
        Incorporates adjuster audit decisions, validates candidate models against active models
        on a holdout validation set, enforces a promotion gate (Brier score regression check),
        and saves versioned model artifacts with audit logging.
        """
        data = self._load_all()
        if len(data) < min_feedback_count:
            return {
                "status": "skipped",
                "message": f"Insufficient feedback records ({len(data)}/{min_feedback_count}) to trigger retraining.",
                "active_version": getattr(ml_service, "model_version", "v1.0"),
                "samples_used": 0
            }

        # 1. Load base dataset
        from src.utils.data_loader import load_claims_dataset
        import pandas as pd
        import numpy as np
        import pickle
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import accuracy_score, brier_score_loss

        base_df = load_claims_dataset()
        df_augmented = base_df.copy()

        # 2. Extract adjuster overrides as training labels
        for entry in data:
            cid = entry.get("claim_id")
            adj_dec = entry.get("adjuster_decision", "")
            adj_amt = entry.get("adjusted_amount")
            conf_fraud = entry.get("confirmed_fraud")

            match = df_augmented[df_augmented["claim_id"] == cid]
            if not match.empty:
                idx = match.index[0]
                # Label integrity: SIU referral is merely suspicion; require explicit confirmation
                if conf_fraud is not None:
                    df_augmented.at[idx, "is_fraud_flagged_ground_truth"] = bool(conf_fraud)
                elif adj_dec == "Approved":
                    df_augmented.at[idx, "is_fraud_flagged_ground_truth"] = False
                
                if adj_amt is not None and float(adj_amt) > 0:
                    df_augmented.at[idx, "claim_amount"] = float(adj_amt)

        # 3. Prepare features & partition 80/20 train/validation holdout split
        X = ml_service._prepare_features(df_augmented, is_training=True)
        y_fraud = df_augmented["is_fraud_flagged_ground_truth"].astype(int).values
        y_amount = df_augmented["claim_amount"].fillna(df_augmented["claim_amount"].median()).values

        X_train, X_val, y_fraud_train, y_fraud_val, y_amt_train, y_amt_val = train_test_split(
            X, y_fraud, y_amount, test_size=0.2, random_state=42, stratify=y_fraud
        )

        # 4. Fit candidate models on train split
        cand_amount = RandomForestRegressor(n_estimators=60, max_depth=8, random_state=42, n_jobs=-1)
        cand_amount.fit(X_train, y_amt_train)

        cand_base_fraud = RandomForestClassifier(n_estimators=80, max_depth=7, class_weight="balanced", random_state=42, n_jobs=-1)
        cand_base_fraud.fit(X_train, y_fraud_train)

        cand_calibrated = CalibratedClassifierCV(estimator=cand_base_fraud, method="sigmoid", cv=5)
        cand_calibrated.fit(X_train, y_fraud_train)

        # 5. Evaluate Candidate Model on Holdout Validation Set
        cand_val_preds = cand_calibrated.predict(X_val)
        cand_val_acc = accuracy_score(y_fraud_val, cand_val_preds)
        cand_val_probs = cand_calibrated.predict_proba(X_val)[:, 1]
        cand_val_brier = brier_score_loss(y_fraud_val, cand_val_probs)

        # 6. Evaluate Active Base Model on the Same Holdout Validation Set
        curr_ver = getattr(ml_service, "model_version", "v1.0")
        base_val_preds = ml_service.fraud_model.predict(X_val)
        base_val_acc = accuracy_score(y_fraud_val, base_val_preds)
        base_val_probs = ml_service.fraud_model.predict_proba(X_val)[:, 1]
        base_val_brier = brier_score_loss(y_fraud_val, base_val_probs)

        # 7. Promotion Gate: Prevent Model Regression
        # Candidate must not degrade validation Brier score (allow 1e-4 tolerance for numerical variance)
        if cand_val_brier > (base_val_brier + 1e-4):
            return {
                "status": "rejected_regression",
                "message": f"Candidate model validation Brier score ({cand_val_brier:.4f}) degraded compared to active model ({base_val_brier:.4f}). Promotion rejected.",
                "active_version": curr_ver,
                "base_validation_brier": round(float(base_val_brier), 4),
                "candidate_validation_brier": round(float(cand_val_brier), 4),
                "base_validation_accuracy": round(float(base_val_acc), 4),
                "candidate_validation_accuracy": round(float(cand_val_acc), 4),
                "feedback_records_integrated": len(data),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

        # 8. Model Versioning Bump (Only when promoted)
        try:
            ver_num = float(curr_ver.replace("v", ""))
            new_ver = f"v{ver_num + 0.1:.1f}"
        except Exception:
            new_ver = "v1.1"

        # 9. Save Versioned Artifact
        versioned_file = ml_service.models_dir / f"claims_models_bundle_{new_ver}.pkl"
        bundle = {
            "amount_model": cand_amount,
            "resolution_model": ml_service.resolution_model,
            "fraud_model": cand_calibrated,
            "base_fraud_model": cand_base_fraud,
            "feature_names": ml_service.feature_names,
            "encoder": ml_service.encoder,
            "model_version": new_ver,
            "retrained_at": datetime.utcnow().isoformat() + "Z",
            "feedback_samples_integrated": len(data)
        }
        with open(versioned_file, "wb") as f:
            pickle.dump(bundle, f)

        # Update active bundle
        with open(ml_service.model_file, "wb") as f:
            pickle.dump(bundle, f)

        # Hot-reload in memory
        ml_service.amount_model = cand_amount
        ml_service.fraud_model = cand_calibrated
        ml_service.base_fraud_model = cand_base_fraud
        ml_service.model_version = new_ver

        return {
            "status": "success",
            "previous_version": curr_ver,
            "active_version": new_ver,
            "feedback_records_integrated": len(data),
            "base_validation_brier": round(float(base_val_brier), 4),
            "candidate_validation_brier": round(float(cand_val_brier), 4),
            "candidate_validation_accuracy": round(float(cand_val_acc), 4),
            "model_artifact": str(versioned_file.name),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

feedback_service = FeedbackService()
