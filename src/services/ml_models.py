import os
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import OneHotEncoder

from src.utils.config import settings
from src.utils.data_loader import load_claims_dataset

class ClaimsMLService:
    def __init__(self):
        self.models_dir = settings.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_file = self.models_dir / "claims_models_bundle.pkl"
        
        self.amount_model: Optional[RandomForestRegressor] = None
        self.resolution_model: Optional[RandomForestRegressor] = None
        self.base_fraud_model: Optional[RandomForestClassifier] = None
        self.fraud_model: Any = None
        self.feature_names: List[str] = []
        self.encoder: Optional[OneHotEncoder] = None
        self.model_version: str = "v1.0"
        self.is_trained = False

    def _prepare_features(self, df: pd.DataFrame, is_training: bool = False) -> np.ndarray:
        df_clean = df.copy()
        
        # Calculate filing delay
        def calc_delay(row):
            try:
                inc = datetime.fromisoformat(str(row["incident_date"]).replace("Z", ""))
                filed = datetime.fromisoformat(str(row["claim_filed_date"]).replace("Z", ""))
                return max(0, (filed - inc).days)
            except Exception:
                return 0
                
        df_clean["filing_delay_days"] = df_clean.apply(calc_delay, axis=1)
        
        # Categorical features
        cat_cols = ["claim_type"]
        num_cols = [
            "policyholder_tenure_years",
            "previous_claims_count",
            "deductible",
            "filing_delay_days"
        ]
        
        cat_data = df_clean[cat_cols].fillna("Auto").values
        if is_training:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded_cats = self.encoder.fit_transform(cat_data)
            encoded_cat_names = list(self.encoder.get_feature_names_out(cat_cols))
            self.feature_names = encoded_cat_names + num_cols
        else:
            if self.encoder is None:
                raise ValueError("Encoder is not fitted.")
            encoded_cats = self.encoder.transform(cat_data)
            
        num_data = df_clean[num_cols].fillna(0).values
        return np.hstack([encoded_cats, num_data])

    def train_or_load(self, force_retrain: bool = False):
        """Train regression and classification models or load cached bundle."""
        if not force_retrain and self.model_file.exists():
            try:
                with open(self.model_file, "rb") as f:
                    bundle = pickle.load(f)
                    self.amount_model = bundle["amount_model"]
                    self.resolution_model = bundle["resolution_model"]
                    self.fraud_model = bundle["fraud_model"]
                    self.base_fraud_model = bundle.get("base_fraud_model", self.fraud_model)
                    self.feature_names = bundle["feature_names"]
                    self.encoder = bundle["encoder"]
                    self.model_version = bundle.get("model_version", "v1.0")
                    self.is_trained = True
                    return
            except Exception:
                pass

        # Load training dataset
        df = load_claims_dataset()
        if len(df) == 0:
            raise ValueError("No data available to train ML models.")

        X = self._prepare_features(df, is_training=True)
        
        # 1. Train Claim Amount Regressor
        y_amount = df["claim_amount"].fillna(df["claim_amount"].median()).values
        self.amount_model = RandomForestRegressor(n_estimators=60, max_depth=8, random_state=42, n_jobs=-1)
        self.amount_model.fit(X, y_amount)

        # 2. Train Resolution Time Regressor (on claims with known resolution days)
        res_mask = df["days_to_resolution"].notna() & (df["days_to_resolution"] > 0)
        if res_mask.sum() > 50:
            X_res = X[res_mask]
            y_res = df.loc[res_mask, "days_to_resolution"].values
            self.resolution_model = RandomForestRegressor(n_estimators=60, max_depth=8, random_state=42, n_jobs=-1)
            self.resolution_model.fit(X_res, y_res)
        else:
            self.resolution_model = self.amount_model

        # 3. Train Fraud Classifier with Platt Scaling Probability Calibration
        y_fraud = df["is_fraud_flagged_ground_truth"].astype(int).values
        self.base_fraud_model = RandomForestClassifier(
            n_estimators=80,
            max_depth=7,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        self.base_fraud_model.fit(X, y_fraud)

        # 5-fold probability calibration for true posterior probability on imbalanced data
        self.fraud_model = CalibratedClassifierCV(
            estimator=self.base_fraud_model,
            method="sigmoid",
            cv=5
        )
        self.fraud_model.fit(X, y_fraud)
        self.model_version = "v1.0"
        self.is_trained = True
        
        # Save bundle
        bundle = {
            "amount_model": self.amount_model,
            "resolution_model": self.resolution_model,
            "fraud_model": self.fraud_model,
            "base_fraud_model": self.base_fraud_model,
            "feature_names": self.feature_names,
            "encoder": self.encoder,
            "model_version": self.model_version
        }
        with open(self.model_file, "wb") as f:
            pickle.dump(bundle, f)

    def predict(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions for amount, resolution time, and calibrated fraud probability with explainability."""
        if not self.is_trained:
            self.train_or_load()

        df_single = pd.DataFrame([claim_data])
        X_single = self._prepare_features(df_single, is_training=False)

        # Predict amount
        predicted_amount = float(self.amount_model.predict(X_single)[0])
        amount_lower = max(100.0, round(predicted_amount * 0.85, 2))
        amount_upper = round(predicted_amount * 1.20, 2)

        # Predict resolution days
        predicted_days = float(self.resolution_model.predict(X_single)[0])
        predicted_days = max(1.0, round(predicted_days, 1))

        # Predict calibrated and base fraud probability
        calibrated_probs = self.fraud_model.predict_proba(X_single)[0]
        calibrated_prob = float(calibrated_probs[1]) if len(calibrated_probs) > 1 else 0.05
        
        base_probs = self.base_fraud_model.predict_proba(X_single)[0] if self.base_fraud_model is not None else calibrated_probs
        raw_prob = float(base_probs[1]) if len(base_probs) > 1 else calibrated_prob
        
        fraud_risk_score = round(raw_prob * 100, 1)
        calibrated_fraud_score = round(calibrated_prob * 100, 1)

        # Risk classification (calibrated for 8.5% industry prevalence)
        if calibrated_fraud_score >= 18.0 or fraud_risk_score >= 60.0:
            risk_tier = "High Risk"
            risk_action = "Refer to SIU (Special Investigation Unit)"
        elif calibrated_fraud_score >= 8.5 or fraud_risk_score >= 32.0:
            risk_tier = "Moderate Risk"
            risk_action = "Senior Adjuster Manual Review"
        else:
            risk_tier = "Low Risk"
            risk_action = "Standard / Fast-Track Processing"

        # Feature importances (Tree Gini split reduction)
        if self.base_fraud_model is not None and hasattr(self.base_fraud_model, "feature_importances_"):
            importances = self.base_fraud_model.feature_importances_
        elif hasattr(self.fraud_model, "feature_importances_"):
            importances = self.fraud_model.feature_importances_
        else:
            importances = np.array([1.0 / max(1, len(self.feature_names))] * len(self.feature_names))

        feat_imp_map = {
            name: round(float(imp), 4)
            for name, imp in zip(self.feature_names, importances)
        }

        # Local feature contribution for this specific claim
        local_attributions = {}
        row_vals = X_single[0]
        for idx, (name, val) in enumerate(zip(self.feature_names, row_vals)):
            weight = float(importances[idx])
            if val > 0:
                local_attributions[name] = round(weight * min(4.0, float(val)), 4)
            else:
                local_attributions[name] = 0.0

        # Underwriting policy flags (rule-based)
        policy_flags = []
        c_amt = float(claim_data.get("claim_amount", 0))
        c_ded = float(claim_data.get("deductible", 500))
        tenure = float(claim_data.get("policyholder_tenure_years", 3.0))
        prev_count = int(claim_data.get("previous_claims_count", 0))

        if tenure < 1.0:
            policy_flags.append(f"Short policy tenure ({tenure:.1f} yrs) provides low historical policyholder baseline.")
        if prev_count >= 2:
            policy_flags.append(f"Elevated prior claim frequency ({prev_count} previous claims filed).")
        if c_amt > 15000:
            policy_flags.append(f"Claim amount (${c_amt:,.2f}) is in the top quartile of loss distributions.")
        if c_amt > 0 and c_ded > 0 and (c_amt / c_ded) > 20:
            policy_flags.append(f"Disproportionate claim-to-deductible ratio ({c_amt/c_ded:.1f}x).")

        # Top Risk Drivers with clear provenance labels
        risk_drivers = []
        for pf in policy_flags:
            risk_drivers.append(f"[Policy Rule] {pf}")

        top_feats = sorted(feat_imp_map.items(), key=lambda x: x[1], reverse=True)[:2]
        for fname, fval in top_feats:
            risk_drivers.append(f"[ML Model Driver] {fname} ({fval*100:.1f}% global feature importance)")

        if not risk_drivers:
            risk_drivers.append("[ML Model Driver] Claim attributes align with standard non-fraud baseline distributions.")

        return {
            "predicted_claim_amount": round(predicted_amount, 2),
            "estimated_amount_range": f"${amount_lower:,.2f} - ${amount_upper:,.2f}",
            "predicted_days_to_resolution": predicted_days,
            "fraud_probability_percent": fraud_risk_score,
            "calibrated_fraud_probability_percent": calibrated_fraud_score,
            "risk_tier": risk_tier,
            "recommended_workflow": risk_action,
            "top_risk_drivers": risk_drivers,
            "underwriting_policy_flags": policy_flags,
            "feature_importance_summary": feat_imp_map,
            "local_feature_attributions": local_attributions,
            "model_version": getattr(self, "model_version", "v1.0")
        }

claims_ml_service = ClaimsMLService()
