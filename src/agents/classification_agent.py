from typing import Dict, Any

class ClassificationAndPrioritizationAgent:
    """Agent responsible for classifying claim complexity and computing priority triage scores."""

    def classify_and_prioritize(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        amt = float(claim_data.get("claim_amount", 0))
        tenure = float(claim_data.get("policyholder_tenure_years", 1.0))
        prev_claims = int(claim_data.get("previous_claims_count", 0))
        c_type = str(claim_data.get("claim_type", "Auto")).capitalize()
        status = str(claim_data.get("claim_status", "Open")).capitalize()

        # 1. Complexity Classification
        complexity_score = 0
        if amt > 25000 or c_type in ["Business", "Property"]:
            complexity_score += 40
        elif amt > 8000:
            complexity_score += 25
        else:
            complexity_score += 10

        if prev_claims >= 2:
            complexity_score += 20
        if tenure < 1.0:
            complexity_score += 15

        if complexity_score >= 60:
            complexity = "High Complexity"
        elif complexity_score >= 35:
            complexity = "Moderate Complexity"
        else:
            complexity = "Low / Standard Complexity"

        # 2. Priority Scoring (0 - 100)
        # Urgency factors: high financial exposure, repeat claim history, customer tenure
        priority_score = 20.0
        
        # Financial exposure
        if amt > 50000:
            priority_score += 35
        elif amt > 15000:
            priority_score += 25
        elif amt > 5000:
            priority_score += 15

        # Prior frequency
        if prev_claims >= 3:
            priority_score += 20
        elif prev_claims >= 1:
            priority_score += 10

        # High-value loyal customer retention factor
        if tenure >= 8.0:
            priority_score += 15
        elif tenure < 0.5:
            priority_score += 10  # new policyholder check

        priority_score = min(100.0, round(priority_score, 1))

        if priority_score >= 75.0:
            priority_tier = "P1 - Critical Priority"
            sla_hours = 24
        elif priority_score >= 50.0:
            priority_tier = "P2 - High Priority"
            sla_hours = 48
        elif priority_score >= 30.0:
            priority_tier = "P3 - Medium Priority"
            sla_hours = 72
        else:
            priority_tier = "P4 - Low / Routine"
            sla_hours = 120

        return {
            "complexity_level": complexity,
            "priority_tier": priority_tier,
            "priority_score": priority_score,
            "target_sla_resolution_hours": sla_hours,
            "triage_summary": f"Classified as {complexity} with triage rank {priority_tier} (Score: {priority_score}/100, Target SLA: {sla_hours} hrs)."
        }

classification_agent = ClassificationAndPrioritizationAgent()
