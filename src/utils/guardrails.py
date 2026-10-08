import re
from datetime import datetime
from typing import Dict, Any, List, Tuple

VALID_CLAIM_TYPES = {"auto", "home", "renters", "property", "business"}

VALID_US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"
}

VALID_STATUSES = {"open", "closed", "approved", "denied", "under investigation", "pending"}

PROMPT_INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"disregard (all )?(prior|previous)",
    r"system override",
    r"you are now (an? )?dan",
    r"<script\b",
    r"drop\s+table",
    r"delete\s+from",
]

class GuardrailValidationResult:
    def __init__(self, is_valid: bool, errors: List[str], warnings: List[str], sanitized_data: Dict[str, Any]):
        self.is_valid = is_valid
        self.errors = errors
        self.warnings = warnings
        self.sanitized_data = sanitized_data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "sanitized_data": self.sanitized_data
        }


def sanitize_text(text: str) -> Tuple[str, List[str]]:
    warnings = []
    if not text:
        return "", warnings
    
    # Check for prompt injection keywords
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            warnings.append(f"Potential prompt injection detected and neutralized: pattern '{pattern}'")
            text = re.sub(pattern, "[FILTERED]", text, flags=re.IGNORECASE)
            
    # Strip HTML tags
    cleaned = re.sub(r"<[^>]*>", "", text).strip()
    return cleaned, warnings


def validate_claim_input(claim_data: Dict[str, Any]) -> GuardrailValidationResult:
    errors = []
    warnings = []
    sanitized = dict(claim_data)

    # 1. Claim Type Validation
    c_type = str(claim_data.get("claim_type", "")).strip().lower()
    if not c_type or c_type not in VALID_CLAIM_TYPES:
        errors.append(f"Invalid claim_type: '{claim_data.get('claim_type')}'. Must be one of {sorted(VALID_CLAIM_TYPES)}.")
    else:
        sanitized["claim_type"] = c_type.capitalize()

    # 2. State Validation
    state = str(claim_data.get("state", "")).strip().upper()
    if state not in VALID_US_STATES:
        errors.append(f"Invalid state code: '{claim_data.get('state')}'. Must be a valid 2-letter US state code.")
    else:
        sanitized["state"] = state

    # 3. Numeric Guardrails: Claim Amount & Deductible
    try:
        amt = float(claim_data.get("claim_amount", 0))
        if amt < 0:
            errors.append("claim_amount cannot be negative.")
        elif amt > 5_000_000:
            warnings.append(f"High-value claim detected (${amt:,.2f}), triggers mandatory supervisor audit.")
        sanitized["claim_amount"] = round(amt, 2)
    except (ValueError, TypeError):
        errors.append("claim_amount must be a valid numeric value.")

    try:
        deductible = float(claim_data.get("deductible", 0))
        if deductible < 0:
            errors.append("deductible cannot be negative.")
        sanitized["deductible"] = round(deductible, 2)
        
        if "claim_amount" in sanitized and sanitized["claim_amount"] < sanitized["deductible"]:
            warnings.append("claim_amount is lower than deductible. Claim may result in zero payout.")
    except (ValueError, TypeError):
        errors.append("deductible must be a valid numeric value.")

    # 4. Policyholder Tenure & Claims Count
    try:
        tenure = float(claim_data.get("policyholder_tenure_years", 0))
        if tenure < 0:
            errors.append("policyholder_tenure_years cannot be negative.")
        elif tenure > 80:
            errors.append("policyholder_tenure_years exceeds valid range (> 80 years).")
        sanitized["policyholder_tenure_years"] = round(tenure, 2)
    except (ValueError, TypeError):
        errors.append("policyholder_tenure_years must be a valid number.")

    try:
        prev_claims = int(claim_data.get("previous_claims_count", 0))
        if prev_claims < 0:
            errors.append("previous_claims_count cannot be negative.")
        sanitized["previous_claims_count"] = prev_claims
        if prev_claims >= 5:
            warnings.append(f"Policyholder has high frequency of previous claims ({prev_claims}). Elevated frequency risk.")
    except (ValueError, TypeError):
        errors.append("previous_claims_count must be an integer.")

    # 5. Date Consistency Check
    inc_date_str = str(claim_data.get("incident_date", "")).strip()
    filed_date_str = str(claim_data.get("claim_filed_date", "")).strip()
    
    inc_dt, filed_dt = None, None
    if inc_date_str:
        try:
            inc_dt = datetime.fromisoformat(inc_date_str.replace("Z", ""))
            sanitized["incident_date"] = inc_dt.strftime("%Y-%m-%d")
        except ValueError:
            errors.append("incident_date must be in ISO format (YYYY-MM-DD).")
            
    if filed_date_str:
        try:
            filed_dt = datetime.fromisoformat(filed_date_str.replace("Z", ""))
            sanitized["claim_filed_date"] = filed_dt.strftime("%Y-%m-%d")
        except ValueError:
            errors.append("claim_filed_date must be in ISO format (YYYY-MM-DD).")
            
    if inc_dt and filed_dt:
        if filed_dt < inc_dt:
            errors.append("claim_filed_date cannot be earlier than incident_date.")
        delay_days = (filed_dt - inc_dt).days
        if delay_days > 180:
            warnings.append(f"Claim filed {delay_days} days after incident. Exceeds standard 180-day notification threshold.")

    # 6. Description / Notes Sanitization if present
    if "description" in claim_data:
        cleaned_desc, desc_warnings = sanitize_text(str(claim_data["description"]))
        sanitized["description"] = cleaned_desc
        warnings.extend(desc_warnings)

    is_valid = len(errors) == 0
    return GuardrailValidationResult(is_valid=is_valid, errors=errors, warnings=warnings, sanitized_data=sanitized)
