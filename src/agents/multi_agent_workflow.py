import json
from typing import Dict, Any, List, Optional
from openai import OpenAI

from src.utils.config import settings
from src.rag.hybrid_retriever import hybrid_retriever
from src.services.ml_models import claims_ml_service
from src.agents.classification_agent import classification_agent

class MultiAgentClaimsWorkflow:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.base_url = settings.OPENAI_BASE_URL
        self.model = settings.OPENAI_MODEL
        self.client = None
        if self.api_key and not self.api_key.startswith("your_"):
            try:
                client_kwargs = {"api_key": self.api_key}
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url
                self.client = OpenAI(**client_kwargs)
            except Exception:
                self.client = None

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if not self.client:
            return None
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=600
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return None

    def run_investigation_agent(self, claim_data: Dict[str, Any], similar_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Agent 1: Investigates policy history, incident facts, and precedents."""
        cid = claim_data.get("claim_id", "NEW_CLAIM")
        c_type = claim_data.get("claim_type", "Auto")
        amt = float(claim_data.get("claim_amount", 0))
        tenure = float(claim_data.get("policyholder_tenure_years", 1.0))
        prev_count = int(claim_data.get("previous_claims_count", 0))

        # Check precedents in similar claims
        precedents = []
        approved_count = 0
        denied_count = 0
        fraud_flagged_count = 0
        for c in similar_claims:
            precedents.append(f"Claim {c.get('claim_id')}: ${c.get('claim_amount', 0):,.2f} ({c.get('claim_status')})")
            if str(c.get("claim_status")).lower() == "approved":
                approved_count += 1
            elif str(c.get("claim_status")).lower() == "denied":
                denied_count += 1
            if c.get("is_fraud_flagged_ground_truth"):
                fraud_flagged_count += 1

        fraud_note = f" (including {fraud_flagged_count} with historical fraud flag)" if fraud_flagged_count > 0 else ""
        summary = (
            f"Policyholder has {tenure} years of policy history with {prev_count} previous claims filed. "
            f"Claim pertains to {c_type} coverage with claimed amount ${amt:,.2f}. "
            f"Retrieved {len(similar_claims)} historical peer claims: {approved_count} approved, "
            f"{denied_count} denied{fraud_note}."
        )

        return {
            "agent_name": "Investigation Agent",
            "findings": summary,
            "similar_claims_count": len(similar_claims),
            "historical_approved_count": approved_count,
            "historical_fraud_count": fraud_flagged_count,
            "key_precedents": precedents[:3]
        }

    def run_risk_agent(self, claim_data: Dict[str, Any], ml_results: Dict[str, Any], investigation: Dict[str, Any]) -> Dict[str, Any]:
        """Agent 2: Evaluates fraud risk, financial anomaly, and risk flags."""
        fraud_prob = ml_results.get("fraud_probability_percent", 5.0)
        risk_tier = ml_results.get("risk_tier", "Low Risk")
        drivers = ml_results.get("top_risk_drivers", [])

        risk_narrative = (
            f"Risk analysis calculated a fraud probability of {fraud_prob}%, placing this claim in the '{risk_tier}' tier. "
            f"Key risk triggers identified: {'; '.join(drivers)}."
        )

        return {
            "agent_name": "Risk Assessment Agent",
            "fraud_probability_percent": fraud_prob,
            "risk_tier": risk_tier,
            "risk_narrative": risk_narrative,
            "risk_drivers": drivers
        }

    def run_recommendation_agent(
        self,
        claim_data: Dict[str, Any],
        investigation: Dict[str, Any],
        risk_analysis: Dict[str, Any],
        ml_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Agent 3: Synthesizes investigation and risk findings into handling decisions with strict financial settlement rules."""
        fraud_prob = risk_analysis.get("fraud_probability_percent", 0.0)
        amt = float(claim_data.get("claim_amount", 0))
        ded_val = claim_data.get("deductible") if claim_data.get("deductible") is not None else claim_data.get("policy_deductible", 500)
        deductible = float(ded_val)
        predicted_amt = ml_results.get("predicted_claim_amount", amt)
        c_type = str(claim_data.get("claim_type", "Auto")).capitalize()

        # Strict Insurance Settlement Invariants:
        prev_claims = int(claim_data.get("previous_claims_count", 0))
        tenure = float(claim_data.get("policyholder_tenure_years", 1.0))
        max_net_payout = max(0.0, round(amt - deductible, 2))

        # Decision routing logic & administrative authorization governance
        if fraud_prob >= 45.0 or (amt >= 50000 and fraud_prob >= 40.0 and risk_analysis.get("risk_tier") != "Low Risk") or risk_analysis.get("risk_tier") == "High Risk":
            decision = "SIU_REFERRAL"
            action = "Refer to Special Investigation Unit (SIU) for comprehensive anti-fraud review."
            fast_track = False
            recommended_net = 0.0
            auth_status = "DISBURSEMENT_WITHHELD_SIU_INQUIRY"
            payout_range = f"$0.00 (Disbursement withheld pending SIU investigation; Net ceiling: ${max_net_payout:,.2f})"
            steps = [
                "Place automated hold on claim settlement disbursement.",
                "Assign SIU investigator to verify incident location, physical evidence, and scene inspection.",
                "Request formal proof-of-loss and sworn affidavit statement from policyholder.",
                "Cross-check National Insurance Crime Bureau (NICB) and internal multi-carrier loss registry."
            ]
        elif amt <= deductible:
            decision = "CLAIM_WITHIN_DEDUCTIBLE"
            action = f"Claimed loss (${amt:,.2f}) does not exceed the applicable policy deductible (${deductible:,.2f}). Zero net indemnity due."
            fast_track = True
            recommended_net = 0.0
            auth_status = "ZERO_INDEMNITY_CLOSED"
            payout_range = "$0.00 (Loss within deductible)"
            steps = [
                f"Verify incident damage assessment (${amt:,.2f}) against applicable policy deductible (${deductible:,.2f}).",
                "Notify policyholder that covered repair costs are absorbed within the elected deductible limit.",
                "Close file with zero indemnity disbursement issued."
            ]
        elif (
            amt > 5000.0
            or prev_claims > 0
            or tenure < 1.0
            or fraud_prob >= 38.0
            or risk_analysis.get("risk_tier") in ["Moderate Risk", "High Risk"]
        ):
            decision = "MANUAL_ADJUSTER_REVIEW"
            action = "Assign to Senior Claims Adjuster for detailed estimate audit."
            fast_track = False
            auth_status = "PENDING_SENIOR_ADJUSTER_SIGN_OFF"

            # Bounded net settlement range: evaluated based on documented loss, bounded by max_net_payout
            est_base_loss = amt
            lower_net = max(0.0, round(est_base_loss * 0.85 - deductible, 2))
            upper_net = max(lower_net, round(min(max_net_payout, est_base_loss - deductible), 2))
            recommended_net = upper_net

            if lower_net >= upper_net or upper_net == 0.0:
                payout_range = f"${max_net_payout:,.2f}"
            else:
                payout_range = f"${lower_net:,.2f} - ${upper_net:,.2f} (Net max: ${max_net_payout:,.2f})"

            # Tailor checklist specifically by claim type
            if c_type == "Auto":
                steps = [
                    "Obtain itemized body shop repair estimates and vehicle damage photographs.",
                    f"Verify deductible application (${deductible:,.2f}) against policy collision terms.",
                    f"Audit historical loss record ({claim_data.get('previous_claims_count', 0)} prior claims) for duplicate or overlapping damage.",
                    "Verify incident circumstances against police report or driver exchange documentation."
                ]
            elif c_type in ["Home", "Property"]:
                steps = [
                    "Obtain licensed contractor itemized rebuild/repair estimates and physical loss photos.",
                    f"Verify deductible application (${deductible:,.2f}) against dwelling policy terms.",
                    "Review emergency water mitigation or structural remediation invoices.",
                    "Confirm date of loss aligns with regional weather and municipal service logs."
                ]
            else:
                steps = [
                    "Request CPA-verified commercial loss records, equipment receipts, and business interruption logs.",
                    f"Review deductible application (${deductible:,.2f}) against commercial casualty limits.",
                    "Obtain vendor invoices and incident reports to verify business property ownership."
                ]
        else:
            decision = "AUTO_APPROVE"
            action = "Eligible for Fast-Track Automated Settlement."
            fast_track = True
            recommended_net = max_net_payout
            auth_status = "ELIGIBLE_FAST_TRACK_DISBURSEMENT"
            payout_range = f"${max_net_payout:,.2f}"
            steps = [
                f"Apply policy deductible of ${deductible:,.2f} to claimed loss of ${amt:,.2f}.",
                f"Issue payment authorization of ${max_net_payout:,.2f} via Direct Deposit / ACH.",
                "Send closing documentation and customer satisfaction survey to policyholder."
            ]

        # LLM enrichment if available
        llm_reasoning = None
        if self.client:
            sys_prompt = "You are an expert Insurance Claims Advisory System. Synthesize the final claim recommendation concisely."
            user_prompt = (
                f"Claim Data: {json.dumps(claim_data)}\n"
                f"Investigation Findings: {investigation.get('findings')}\n"
                f"Risk Findings: {risk_analysis.get('risk_narrative')}\n"
                f"Decision: {decision}\n"
                f"Net Payout Range: {payout_range}\n"
                "Provide a 2-sentence executive rationale for the adjuster."
            )
            llm_reasoning = self._call_llm(sys_prompt, user_prompt)

        return {
            "agent_name": "Recommendation Agent",
            "decision": decision,
            "action_statement": action,
            "fast_track_eligible": fast_track,
            "net_settlement_ceiling": max_net_payout,
            "recommended_net_payout": recommended_net,
            "authorized_net_payout": recommended_net,  # Maintained as backward-compatible alias
            "payment_authorization_status": auth_status,
            "recommended_payout": payout_range,
            "actionable_steps": steps,
            "executive_rationale": llm_reasoning or f"Based on {risk_analysis.get('risk_tier')} profile and {investigation.get('findings')}, {action}"
        }

    def run_llm_as_judge(
        self,
        claim_data: Dict[str, Any],
        investigation: Dict[str, Any],
        risk_analysis: Dict[str, Any],
        recommendation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """LLM-as-Judge validation module evaluating factual consistency, completeness, and financial compliance."""
        amt = float(claim_data.get("claim_amount", 0))
        ded_val = claim_data.get("deductible") if claim_data.get("deductible") is not None else claim_data.get("policy_deductible", 500)
        deductible = float(ded_val)
        fraud_prob = float(risk_analysis.get("fraud_probability_percent", 0.0))
        decision = str(recommendation.get("decision", ""))

        # Scoring heuristics & LLM validation
        factual_score = 9.5
        completeness_score = 9.0
        compliance_score = 9.5
        issues = []

        # 1. Financial Consistency: Invariant Verification
        max_allowable_net = max(0.0, round(amt - deductible, 2))

        # Check structured numeric authorized payout if provided
        if "authorized_net_payout" in recommendation and recommendation["authorized_net_payout"] is not None:
            structured_payout = float(recommendation["authorized_net_payout"])
        else:
            # Fallback to regex parsing of recommended_payout string
            import re
            rec_payout_str = str(recommendation.get("recommended_payout", ""))
            found_numbers = [float(x.replace(",", "")) for x in re.findall(r"\$([0-9,]+\.?[0-9]*)", rec_payout_str)]
            structured_payout = max(found_numbers) if found_numbers else 0.0

        # Invariant Rule A: Payout cannot exceed Net Settlement Ceiling
        if structured_payout > max_allowable_net + 0.01:
            factual_score -= 5.0
            compliance_score -= 6.0
            issues.append(f"Financial Inconsistency: Authorized payout (${structured_payout:,.2f}) exceeds maximum allowable net loss after deductible (${max_allowable_net:,.2f}).")

        # Invariant Rule B: Loss within deductible must have $0.00 authorized payout
        if amt <= deductible and structured_payout > 0.0:
            factual_score -= 5.0
            compliance_score -= 6.0
            issues.append(f"Financial Inconsistency: Claimed loss (${amt:,.2f}) is within deductible (${deductible:,.2f}); authorized payout must be $0.00 but got ${structured_payout:,.2f}.")

        # Invariant Rule C: SIU referrals must not authorize payout disbursement
        if decision == "SIU_REFERRAL" and structured_payout > 0.0:
            factual_score -= 5.0
            compliance_score -= 6.0
            issues.append(f"Financial Inconsistency: SIU referral must have $0.00 authorized payout disbursement pending investigation, but got ${structured_payout:,.2f}.")

        # Invariant Rule D: Regex cross-validation for narrative leakage of exceeding payouts
        import re
        rec_payout_str = str(recommendation.get("recommended_payout", ""))
        for num in [float(x.replace(",", "")) for x in re.findall(r"\$([0-9,]+\.?[0-9]*)", rec_payout_str)]:
            # Allow mentioning the ceiling itself e.g. "(Net max: $3,300.00)" or "(Net ceiling: $3,300.00)"
            if num > max_allowable_net + 1.0 and decision != "SIU_REFERRAL":
                msg = f"Financial Inconsistency: Payout narrative contains amount (${num:,.2f}) exceeding net deductible ceiling (${max_allowable_net:,.2f})."
                if msg not in issues:
                    factual_score -= 4.0
                    compliance_score -= 5.0
                    issues.append(msg)

        # 2. Decision Logic Alignment
        if fraud_prob >= 50.0 and decision == "AUTO_APPROVE":
            factual_score -= 5.0
            compliance_score -= 6.0
            issues.append("Contradiction: High fraud probability cannot be auto-approved.")
        if amt > 50000 and recommendation.get("fast_track_eligible"):
            compliance_score -= 4.0
            issues.append("High claim amount exceeds standard fast-track compliance limits.")

        overall_score = round(max(0.0, min(10.0, (factual_score + completeness_score + compliance_score) / 3.0)), 1)
        verdict = "PASS" if overall_score >= 8.0 and not issues else "FLAGGED_FOR_AUDIT"

        # Optional LLM-as-Judge critique
        critique = "Recommendation adheres to underwriting policy standards, financial settlement rules, and factual claim attributes."
        if issues:
            critique = "Audit warning: " + "; ".join(issues)
        elif self.client:
            judge_sys = "You are an independent Insurance Audit Judge. Review the recommendation for financial consistency, policy compliance, and accuracy."
            judge_prompt = f"Claim: {claim_data}\nDecision: {decision}\nPayout: {recommendation.get('recommended_payout')}\nAction: {recommendation.get('action_statement')}. Provide a 1-sentence audit verdict."
            llm_judge = self._call_llm(judge_sys, judge_prompt)
            if llm_judge:
                critique = llm_judge

        return {
            "judge_name": "LLM-as-Judge Evaluator",
            "verdict": verdict,
            "overall_quality_score": overall_score,
            "metric_scores": {
                "factual_consistency": factual_score,
                "completeness": completeness_score,
                "policy_compliance": compliance_score
            },
            "audit_critique": critique,
            "detected_issues": issues
        }

    def process_claim(self, claim_data: Dict[str, Any], top_k_similar: int = 5) -> Dict[str, Any]:
        """Execute the entire multi-agent workflow for an insurance claim."""
        # 1. Classification & Prioritization
        classification = classification_agent.classify_and_prioritize(claim_data)

        # 2. Similar claims retrieval via Hybrid RAG (Damage-Aware & Severity-Tiered)
        c_type = str(claim_data.get("claim_type", "Auto")).capitalize()
        c_state = claim_data.get("state")
        c_amt = float(claim_data.get("claim_amount", 0.0))
        c_desc = str(claim_data.get("description", "")).strip()

        # Build damage-aware semantic query capturing specific physical impact and loss magnitude
        if c_desc:
            query = f"{c_type} damage in {c_state}: {c_desc} (loss amount ${c_amt:,.2f})"
        else:
            query = f"{c_type} insurance claim in state {c_state} amount ${c_amt:,.2f}"

        # Dynamic financial bracketing to retrieve peer claims in comparable loss severity tiers:
        # High-severity claims (>= $25k) -> peer claims >= $12,000
        # Routine minor claims (<= $5k) -> peer claims <= $8,000
        min_amt_filter = None
        max_amt_filter = None
        if c_amt >= 25000.0:
            min_amt_filter = 12000.0
        elif c_amt <= 5000.0 and c_amt > 0:
            max_amt_filter = 8000.0

        similar_claims = hybrid_retriever.search(
            query=query,
            top_k=top_k_similar,
            claim_type=c_type,
            state=c_state,
            min_amount=min_amt_filter,
            max_amount=max_amt_filter
        )
        # Fallback if filtered bracket returned fewer than 2 claims
        if len(similar_claims) < 2 and (min_amt_filter or max_amt_filter):
            similar_claims = hybrid_retriever.search(
                query=query,
                top_k=top_k_similar,
                claim_type=c_type,
                state=c_state
            )

        # 3. Machine Learning predictions
        ml_results = claims_ml_service.predict(claim_data)

        # 4. Agent 1: Investigation Agent
        investigation = self.run_investigation_agent(claim_data, similar_claims)

        # 5. Agent 2: Risk Assessment Agent
        risk_analysis = self.run_risk_agent(claim_data, ml_results, investigation)

        # 6. Agent 3: Recommendation Agent
        recommendation = self.run_recommendation_agent(claim_data, investigation, risk_analysis, ml_results)

        # 7. LLM-as-Judge Validation
        judge_review = self.run_llm_as_judge(claim_data, investigation, risk_analysis, recommendation)

        return {
            "claim_id": claim_data.get("claim_id", "NEW_CLAIM"),
            "classification": classification,
            "ml_predictions": ml_results,
            "similar_historical_claims": similar_claims,
            "investigation": investigation,
            "risk_assessment": risk_analysis,
            "recommendation": recommendation,
            "llm_as_judge_review": judge_review
        }

multi_agent_workflow = MultiAgentClaimsWorkflow()
