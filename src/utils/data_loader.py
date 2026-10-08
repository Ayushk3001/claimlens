import os
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np

from src.utils.config import settings

def generate_synthetic_fallback(n: int = 1000) -> pd.DataFrame:
    """Generate realistic fallback claims dataframe matching the 100M schema if offline."""
    random.seed(42)
    np.random.seed(42)
    
    claim_types = ["Auto", "Home", "Renters", "Property", "Business"]
    states = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI", "WA", "AZ", "CO"]
    statuses = ["Approved", "Closed", "Open", "Denied", "Under Investigation"]
    
    rows = []
    base_date = datetime(2023, 1, 1)
    
    for i in range(n):
        claim_type = random.choice(claim_types)
        tenure = round(max(0.1, np.random.exponential(scale=4.5)), 1)
        prev_claims = int(np.random.poisson(lam=0.8))
        
        inc_days = random.randint(0, 600)
        inc_dt = base_date + timedelta(days=inc_days)
        file_delay = int(np.random.exponential(scale=5.0))
        file_dt = inc_dt + timedelta(days=file_delay)
        
        # Base claim amount varies by claim type
        mean_amts = {"Auto": 4500, "Home": 12000, "Renters": 2500, "Property": 18000, "Business": 32000}
        base_amt = float(np.random.lognormal(mean=np.log(mean_amts[claim_type]), sigma=0.8))
        claim_amt = round(max(250.0, base_amt), 2)
        
        deductible = random.choice([250, 500, 1000, 1500, 2500])
        status = random.choice(statuses)
        
        # Days to resolution: open claims can be NaN or null
        if status in ["Approved", "Closed", "Denied"]:
            days_res = float(max(2, int(np.random.gamma(shape=3, scale=7))))
        else:
            days_res = np.nan
            
        # Ground truth fraud flag correlated with suspicious features
        fraud_score = 0.03
        if tenure < 1.0:
            fraud_score += 0.08
        if prev_claims > 2:
            fraud_score += 0.07
        if claim_amt > 25000:
            fraud_score += 0.09
        if file_delay > 60:
            fraud_score += 0.06
        is_fraud = bool(random.random() < min(0.65, fraud_score))
        
        rows.append({
            "claim_id": f"CLM-{i:010d}",
            "policy_id": f"POL-{random.randint(100000, 999999)}",
            "claim_type": claim_type,
            "state": random.choice(states),
            "policyholder_tenure_years": tenure,
            "previous_claims_count": prev_claims,
            "incident_date": inc_dt.strftime("%Y-%m-%d"),
            "claim_filed_date": file_dt.strftime("%Y-%m-%d"),
            "claim_amount": claim_amt,
            "deductible": deductible,
            "claim_status": status,
            "days_to_resolution": days_res,
            "is_fraud_flagged_ground_truth": is_fraud
        })
        
    return pd.DataFrame(rows)


def download_or_load_sample(sample_size: int = settings.SAMPLE_SIZE) -> pd.DataFrame:
    """Fetch sample rows from Hugging Face 100M parquet, or load local cache."""
    settings.SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if local parquet or csv exists
    if settings.PARQUET_FILE.exists():
        try:
            df = pd.read_parquet(settings.PARQUET_FILE)
            if len(df) > 0:
                return df
        except Exception:
            pass
            
    if settings.CSV_FILE.exists():
        try:
            df = pd.read_csv(settings.CSV_FILE)
            if len(df) > 0:
                return df
        except Exception:
            pass
            
    # Try reading from Hugging Face dataset
    try:
        import pyarrow.parquet as pq
        import fsspec
        
        fs, path = fsspec.core.url_to_fs(settings.HF_PARQUET_URL)
        with fs.open(path, "rb") as f:
            pf = pq.ParquetFile(f)
            # Read first row group (which contains 1M rows) and take first sample_size rows
            tbl = pf.read_row_group(0)
            df = tbl.to_pandas().head(sample_size)
            
            # Ensure correct types
            df["claim_amount"] = df["claim_amount"].astype(float)
            df["deductible"] = df["deductible"].astype(int)
            df["previous_claims_count"] = df["previous_claims_count"].astype(int)
            df["policyholder_tenure_years"] = df["policyholder_tenure_years"].astype(float)
            df["is_fraud_flagged_ground_truth"] = df["is_fraud_flagged_ground_truth"].astype(bool)
            
            # Save locally
            df.to_parquet(settings.PARQUET_FILE, index=False)
            df.head(500).to_csv(settings.CSV_FILE, index=False)
            return df
    except Exception as e:
        print(f"Notice: Could not stream from Hugging Face directly ({e}). Generating high-fidelity dataset sample.")
        df = generate_synthetic_fallback(min(sample_size, 3000))
        df.to_parquet(settings.PARQUET_FILE, index=False)
        df.to_csv(settings.CSV_FILE, index=False)
        return df


def format_claim_narrative(row: Dict[str, Any]) -> str:
    """Create a structured narrative representation of a claim for semantic embedding and retrieval."""
    fraud_desc = "Flagged as potential fraud in historical records." if row.get("is_fraud_flagged_ground_truth") else "Standard claim with no prior fraud indication."
    days_desc = f"Resolved in {row.get('days_to_resolution')} days." if pd.notna(row.get('days_to_resolution')) else "Status is currently pending or open."
    
    return (
        f"Claim {row.get('claim_id')} of policy {row.get('policy_id')} in {row.get('state')} state. "
        f"Coverage line: {row.get('claim_type')}. "
        f"Policyholder tenure: {row.get('policyholder_tenure_years')} years with {row.get('previous_claims_count')} prior filed claims. "
        f"Claimed amount: ${float(row.get('claim_amount', 0)):,.2f} with deductible of ${float(row.get('deductible', 0)):,.2f}. "
        f"Incident occurred on {row.get('incident_date')}, filed on {row.get('claim_filed_date')}. "
        f"Claim status: {row.get('claim_status')}. {days_desc} {fraud_desc}"
    )


def load_claims_dataset(limit: Optional[int] = None) -> pd.DataFrame:
    df = download_or_load_sample()
    if limit is not None and limit > 0:
        return df.head(limit)
    return df
