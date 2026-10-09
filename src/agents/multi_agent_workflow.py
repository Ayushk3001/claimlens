"""Multi-agent insurance-claims workflow with deterministic guardrails.

This module is an advisory triage component, not a policy adjudication engine.
It does not determine coverage or authorize payment; an authorized adjuster must
verify policy terms, limits, exclusions, covered loss, and supporting evidence.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

from src.utils.config import settings
from src.rag.hybrid_retriever import hybrid_retriever
from src.services.ml_models import claims_ml_service
from src.agents.classification_agent import classification_agent


class MultiAgentClaimsWorkflow:
    """Orchestrates claim classification, retrieval, ML risk, and recommendations."""

    SIU_THRESHOLD_PERCENT = 45.0
    MANUAL_REVIEW_AMOUNT_THRESHOLD = 5_000.0
    HIGH_AMOUNT_SIU_THRESHOLD = 50_000.0

    def __init__(self) -> None:
        self.api_key = getattr(settings, "OPENAI_API_KEY", None)
        self.base_url = getattr(settings, "OPENAI_BASE_URL", None)
        self.model = getattr(settings, "OPENAI_MODEL", None)
        self.client: Optional[OpenAI] = None

        if self.api_key and not str(self.api_key).lower().startswith("your_"):
            try:
                client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url
                self.client = OpenAI(**client_kwargs)
            except Exception:
                # The workflow remains usable with deterministic fallbacks.
                self.client = None

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        """Convert to a finite float; never propagate NaN or infinity."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if math.isfinite(number) else default

    @staticmethod
    def _claim_type(claim_data: Dict[str, Any]) -> str:
        raw = str(claim_data.get("claim_type", "Auto") or "Auto").strip().lower()
        aliases = {
            "tenant": "Renters",
            "renter": "Renters",
            "renters": "Renters",
            "homeowners": "Home",
            "homeowner": "Home",
            "property": "Home",
            "home": "Home",
            "commercial": "Business",
            "business": "Business",
            "auto": "Auto",
            "vehicle": "Auto",
        }
        return aliases.get(raw, raw.title() or "Unknown")

    @staticmethod
    def _description(claim_data: Dict[str, Any]) -> str:
        # Support the common field names used by the intake UI and API.
        for key in ("description", "incident_description", "incident_notes", "notes"):
            value = claim_data.get(key)
            if value:
                return str(value).strip()
        return ""

    @staticmethod
    def _fraud_percent(ml_results: Dict[str, Any]) -> Optional[float]:
        """Return a validated percentage, or None if the model did not provide one.

        Supports a 0..1 probability only when the key is named fraud_probability.
        The explicit *_percent field is always interpreted as a percentage.
        """
        if ml_results.get("fraud_probability_percent") is not None:
            value = MultiAgentClaimsWorkflow._number(
                ml_results.get("fraud_probability_percent"), float("nan")
            )
        elif ml_results.get("fraud_probability") is not None:
            value = MultiAgentClaimsWorkflow._number(
                ml_results.get("fraud_probability"), float("nan")
            ) * 100.0
        else:
            return None
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            return None
        return round(value, 2)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if not self.client or not self.model:
            return None
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=600,
            )
            content = response.choices[0].message.content
            return content.strip() if content else None
        except Exception:
            # A provider/API failure should not break deterministic validation.
            return None

    def run_investigation_agent(
        self,
        claim_data: Dict[str, Any],
        similar_claims: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Summarize intake facts and retrieved precedents without inferring coverage."""
        claim_id = claim_data.get("claim_id", "NEW_CLAIM")
        claim_type = self._claim_type(claim_data)
        amount = self._number(claim_data.get("claim_amount"))
        tenure = self._number(claim_data.get("policyholder_tenure_years"), 0.0)
        previous_claims = max(0, int(self._number(claim_data.get("previous_claims_count"))))

        precedents: List[str] = []
        approved_count = 0
        denied_count = 0
        fraud_flagged_count = 0
        status_counts: Dict[str, int] = {}

        for claim in similar_claims or []:
            status = str(claim.get("claim_status", "unknown") or "unknown").strip().lower()
            status_counts[status] = status_counts.get(status, 0) + 1
            claim_amount = self._number(claim.get("claim_amount"))
            precedents.append(
                f"Claim {claim.get('claim_id', 'unknown')}: "
                f"${claim_amount:,.2f} ({status})"
            )
            if status == "approved":
                approved_count += 1
            elif status == "denied":
                denied_count += 1
            flag = claim.get("is_fraud_flagged_ground_truth", False)
            if str(flag).strip().lower() in {"1", "true", "yes"} or flag is True:
                fraud_flagged_count += 1

        fraud_note = (
            f"; {fraud_flagged_count} with historical fraud flags"
            if fraud_flagged_count
            else ""
        )
        summary = (
            f"Claim {claim_id}: policy tenure {tenure:.1f} years, "
            f"{previous_claims} prior claims; {claim_type} claim for ${amount:,.2f}. "
            f"Retrieved {len(similar_claims or [])} historical records: "
            f"{approved_count} approved, {denied_count} denied{fraud_note}."
        )

        return {
            "agent_name": "Investigation Agent",
            "findings": summary,
            "similar_claims_count": len(similar_claims or []),
            "historical_approved_count": approved_count,
            "historical_denied_count": denied_count,
            "historical_fraud_count": fraud_flagged_count,
            "historical_status_counts": status_counts,
            "key_precedents": precedents[:3],
        }

    def run_risk_agent(
        self,
        claim_data: Dict[str, Any],
        ml_results: Dict[str, Any],
        investigation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Expose model outputs carefully; distinguish global importance from local reasons."""
        fraud_probability = self._fraud_percent(ml_results)
        risk_tier = ml_results.get("risk_tier")
        if not risk_tier:
            if fraud_probability is None:
                risk_tier = "Unknown"
            elif fraud_probability >= 65:
                risk_tier = "High Risk"
            elif fraud_probability >= 35:
                risk_tier = "Moderate Risk"
            else:
                risk_tier = "Low Risk"

        raw_drivers = ml_results.get("top_risk_drivers") or []
        drivers = [str(item) for item in raw_drivers if str(item).strip()]
        global_importance_notes = [
            driver for driver in drivers if "global feature importance" in driver.lower()
        ]
        case_specific_indicators = [
            driver for driver in drivers if "global feature importance" not in driver.lower()
        ]

        if fraud_probability is None:
            risk_narrative = (
                "The ML service did not provide a valid fraud probability. "
                "Do not infer a probability or automatically classify the claim as low risk."
            )
        else:
            risk_narrative = (
                f"The ML service reported a fraud probability of {fraud_probability:.1f}% "
                f"and tier '{risk_tier}'. This is a model estimate, not proof of fraud."
            )

        if case_specific_indicators:
            risk_narrative += " Case-level indicators: " + "; ".join(case_specific_indicators) + "."
        if global_importance_notes:
            risk_narrative += (
                " Model-wide feature-importance context (not case-specific explanations): "
                + "; ".join(global_importance_notes)
                + "."
            )

        return {
            "agent_name": "Risk Assessment Agent",
            "fraud_probability_percent": fraud_probability,
            "risk_tier": risk_tier,
            "risk_narrative": risk_narrative,
            "risk_drivers": drivers,
            "case_specific_indicators": case_specific_indicators,
            "global_feature_importance_notes": global_importance_notes,
        }

    def _build_claim_checklist(
        self, claim_data: Dict[str, Any], deductible: float
    ) -> List[str]:
        """Build incident-aware steps; do not reuse water-damage steps for theft, etc."""
        claim_type = self._claim_type(claim_data)
        description = self._description(claim_data).lower()
        deductible_text = f"Confirm the ${deductible:,.2f} deductible and applicable policy terms."

        if claim_type == "Auto":
            if any(word in description for word in ("windshield", "glass", "stone struck", "cracked windshield")):
                return [
                    "Review windshield photographs and the itemized auto-glass repair/replacement estimate.",
                    "Confirm comprehensive coverage and whether a separate glass deductible or waiver applies.",
                    "Check that the estimate addresses the reported glass damage and excludes unrelated damage.",
                    deductible_text,
                ]
            if any(word in description for word in ("theft", "stolen", "vehicle was taken")):
                return [
                    "Verify the police report, theft timeline, vehicle identification number, and ownership records.",
                    "Confirm recovery status and document any recovered-vehicle damage or missing property.",
                    "Review applicable comprehensive coverage, exclusions, and valuation requirements.",
                    deductible_text,
                ]
            return [
                "Compare the itemized repair estimate with vehicle damage photographs and inspection findings.",
                "Verify the police report or driver-exchange details and the reported collision circumstances.",
                "Check for pre-existing or unrelated damage only where the evidence indicates a concern.",
                deductible_text,
            ]

        if claim_type in {"Renters", "Home"}:
            is_theft = any(word in description for word in ("theft", "stolen", "break-in", "burglary", "robbery"))
            is_water = any(word in description for word in ("water", "burst pipe", "plumb", "flood", "leak", "leaking"))
            is_fire = any(word in description for word in ("fire", "smoke", "burn"))
            is_weather = any(word in description for word in ("storm", "hail", "wind damage", "tornado", "hurricane"))

            if is_theft:
                steps = [
                    "Verify the police report, incident timeline, and reported point of entry where applicable.",
                    "Obtain an itemized inventory and match receipts, serial numbers, photographs, or alternative ownership evidence to claimed items.",
                    "Check applicable personal-property coverage, item sublimits, scheduled-property endorsements, exclusions, and valuation rules.",
                ]
                if claim_type == "Renters":
                    steps.append("Confirm whether the claim concerns tenant-owned contents and whether any building damage belongs to the landlord's policy.")
                else:
                    steps.append("Separate stolen personal property from any claimed dwelling or structural damage.")
                steps.append(deductible_text)
                return steps

            if is_water:
                steps = [
                    "Review damage photographs, the plumber or mitigation report, and the documented source of the water.",
                    "Obtain an itemized inventory and repair estimates; match receipts or alternative ownership evidence to damaged belongings.",
                    "Check policy provisions for the reported water source, applicable exclusions, mitigation duties, and personal-property or dwelling limits.",
                ]
                if claim_type == "Renters":
                    steps.append("Confirm responsibility for tenant-owned contents versus landlord-owned building components, and request landlord maintenance records if relevant.")
                else:
                    steps.append("Review reasonable emergency mitigation and structural repair invoices for damage supported by the evidence.")
                steps.append(deductible_text)
                return steps

            if is_fire:
                return [
                    "Review fire department or incident reports and photographs documenting the affected areas.",
                    "Obtain an itemized inventory and repair estimates, with ownership or valuation evidence for claimed contents.",
                    "Check fire/smoke coverage, applicable exclusions, limits, and actual-cash-value or replacement-cost provisions.",
                    deductible_text,
                ]

            if is_weather:
                return [
                    "Compare photographs and itemized repair estimates with the reported storm-related damage.",
                    "Verify the date and cause of loss using available incident, contractor, or weather documentation where relevant.",
                    "Check applicable wind, hail, storm, water, and personal-property provisions, including exclusions and sublimits.",
                    deductible_text,
                ]

            if claim_type == "Renters":
                return [
                    "Obtain an itemized list of claimed personal property and supporting receipts, photographs, serial numbers, or alternative ownership evidence.",
                    "Confirm renters personal-property coverage, applicable limits, exclusions, and valuation/depreciation rules.",
                    "Request incident-specific documentation and repair or replacement estimates for the reported loss.",
                    deductible_text,
                ]
            return [
                "Obtain itemized contractor estimates and photographs of the reported dwelling or personal-property loss.",
                "Verify ownership, cause of loss, and any supporting incident or mitigation documentation.",
                "Check applicable coverage limits, exclusions, endorsements, and valuation provisions.",
                deductible_text,
            ]

        if claim_type == "Business":
            return [
                "Obtain itemized repair/replacement estimates, relevant vendor invoices, and evidence of business-property ownership.",
                "Request business interruption or financial loss records only if that type of loss is being claimed.",
                "Check applicable commercial coverage, limits, exclusions, and valuation provisions.",
                deductible_text,
            ]

        return [
            "Obtain itemized estimates, photographs, and supporting proof of loss relevant to the reported incident.",
            "Verify ownership or insurable interest and the applicable coverage, limits, exclusions, and valuation provisions.",
            deductible_text,
        ]

    def run_recommendation_agent(
        self,
        claim_data: Dict[str, Any],
        investigation: Dict[str, Any],
        risk_analysis: Dict[str, Any],
        ml_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Recommend a handling pathway with bounded amounts and no implied payment authorization."""
        amount = max(0.0, self._number(claim_data.get("claim_amount")))
        deductible_value = claim_data.get("deductible")
        if deductible_value is None:
            deductible_value = claim_data.get("policy_deductible", 500.0)
        deductible = max(0.0, self._number(deductible_value, 500.0))
        previous_claims = max(0, int(self._number(claim_data.get("previous_claims_count"))))
        tenure = max(0.0, self._number(claim_data.get("policyholder_tenure_years"), 0.0))
        fraud_probability = risk_analysis.get("fraud_probability_percent")
        risk_tier = str(risk_analysis.get("risk_tier", "Unknown"))
        max_net_payout = round(max(0.0, amount - deductible), 2)

        # Do not treat a missing model score as 0% fraud risk.
        score_missing = fraud_probability is None
        fraud_score = self._number(fraud_probability, -1.0)
        high_risk = risk_tier.lower() == "high risk"
        moderate_risk = risk_tier.lower() == "moderate risk"
        description = self._description(claim_data).lower()

        # SIU requires an explicit high-risk signal; a large amount by itself is not fraud.
        siu_trigger = (
            (not score_missing and fraud_score >= self.SIU_THRESHOLD_PERCENT)
            or high_risk
            or (
                amount >= self.HIGH_AMOUNT_SIU_THRESHOLD
                and not score_missing
                and fraud_score >= 40.0
                and risk_tier.lower() != "low risk"
            )
        )
        if siu_trigger:
            decision = "SIU_REFERRAL"
            action = "Refer for specialist investigation based on the configured risk trigger; the referral is not a finding of fraud."
            fast_track = False
            recommended_net = 0.0
            authorized_net = 0.0
            auth_status = "DISBURSEMENT_WITHHELD_PENDING_SIU_REVIEW"
            payout_range = (
                f"$0.00 authorized while SIU review is pending "
                f"(simplified net ceiling: ${max_net_payout:,.2f})"
            )
            steps = [
                "Document the specific rule or model signal that triggered referral and preserve supporting evidence.",
                "Request only claim-relevant proof of loss and incident documentation; explain outstanding requirements to the policyholder.",
                "Keep disbursement on hold pending authorized review; do not treat referral alone as proof of fraud.",
            ]
        elif amount <= deductible:
            decision = "CLAIM_WITHIN_DEDUCTIBLE"
            action = (
                f"The claimed amount (${amount:,.2f}) does not exceed the entered deductible "
                f"(${deductible:,.2f}); the simplified net indemnity is $0.00, subject to policy verification."
            )
            fast_track = False
            recommended_net = 0.0
            authorized_net = 0.0
            auth_status = "ZERO_NET_INDEMNITY_SUBJECT_TO_POLICY_VERIFICATION"
            payout_range = "$0.00 (claimed amount is within the entered deductible)"
            steps = [
                "Verify the estimate and confirm that the entered deductible applies to this coverage and loss type.",
                "Explain the calculated zero net amount to the policyholder, subject to policy terms and any applicable special deductible.",
            ]
        else:
            # Review routing is based on explicit operational criteria. A previous claim
            # or tenure alone should not be represented as evidence of fraud.
            manual_review = (
                amount > self.MANUAL_REVIEW_AMOUNT_THRESHOLD
                or previous_claims >= 2
                or tenure < 1.0
                or moderate_risk
                or high_risk
                or (not score_missing and fraud_score >= 38.0)
            )
            if manual_review:
                decision = "MANUAL_ADJUSTER_REVIEW"
                action = "Route to a claims adjuster to verify evidence, applicable coverage, and the supported loss amount."
                fast_track = False
                auth_status = "PENDING_ADJUSTER_REVIEW"
                # Do not fabricate a lower payout estimate (e.g. 85% of the claim). Only
                # the simplified maximum can be calculated from amount and deductible.
                recommended_net = max_net_payout
                authorized_net = 0.0
                payout_range = (
                    f"Pending itemized loss and policy verification; "
                    f"simplified net ceiling: ${max_net_payout:,.2f}"
                )
                steps = self._build_claim_checklist(claim_data, deductible)
            else:
                # This means the claim appears eligible for routine processing, not that
                # policy coverage has been proven. Keep actual authorization separate.
                decision = "ROUTINE_REVIEW"
                action = "Proceed through routine claim validation before any settlement authorization."
                fast_track = True
                auth_status = "PENDING_ROUTINE_COVERAGE_VALIDATION"
                recommended_net = max_net_payout
                authorized_net = 0.0
                payout_range = (
                    f"Pending coverage and estimate validation; "
                    f"simplified net ceiling: ${max_net_payout:,.2f}"
                )
                steps = self._build_claim_checklist(claim_data, deductible)

        llm_reasoning = None
        if self.client:
            system_prompt = (
                "You are an insurance claims triage assistant. Do not decide coverage, "
                "invent payout estimates, or state that payment is authorized. Explain "
                "the recommended handling pathway using only supplied facts."
            )
            user_prompt = (
                f"Claim data: {json.dumps(claim_data, default=str)}\n"
                f"Investigation: {investigation.get('findings', '')}\n"
                f"Risk analysis: {risk_analysis.get('risk_narrative', '')}\n"
                f"Decision: {decision}\n"
                f"Payout note: {payout_range}\n"
                f"Checklist: {json.dumps(steps)}\n"
                "Write a concise two-sentence rationale. Do not introduce facts or policy terms not provided."
            )
            llm_reasoning = self._call_llm(system_prompt, user_prompt)

        fallback_rationale = (
            f"{decision}: {action} The calculated amount is a simplified ceiling only; "
            "coverage, limits, exclusions, valuation, and policy terms require verification."
        )
        return {
            "agent_name": "Recommendation Agent",
            "decision": decision,
            "action_statement": action,
            "fast_track_eligible": fast_track,
            "net_settlement_ceiling": max_net_payout,
            "recommended_net_payout": recommended_net,
            # Actual authorization remains zero until an authorized workflow approves payment.
            "authorized_net_payout": authorized_net,
            "payment_authorization_status": auth_status,
            "recommended_payout": payout_range,
            "actionable_steps": steps,
            "executive_rationale": llm_reasoning or fallback_rationale,
        }

    def run_llm_as_judge(
        self,
        claim_data: Dict[str, Any],
        investigation: Dict[str, Any],
        risk_analysis: Dict[str, Any],
        recommendation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Deterministic audit checks plus optional LLM critique.

        Scores are heuristic rule-based scores unless the configured LLM critique is
        returned. They should not be described as an independent LLM score by default.
        """
        amount = max(0.0, self._number(claim_data.get("claim_amount")))
        deductible_value = claim_data.get("deductible")
        if deductible_value is None:
            deductible_value = claim_data.get("policy_deductible", 500.0)
        deductible = max(0.0, self._number(deductible_value, 500.0))
        ceiling = round(max(0.0, amount - deductible), 2)
        decision = str(recommendation.get("decision", ""))
        issues: List[str] = []
        factual_score = 10.0
        completeness_score = 10.0
        compliance_score = 10.0

        # Financial constraints.
        authorized = self._number(recommendation.get("authorized_net_payout"), 0.0)
        recommended = self._number(recommendation.get("recommended_net_payout"), 0.0)
        if authorized < 0 or authorized > ceiling + 0.01:
            factual_score -= 5.0
            compliance_score -= 6.0
            issues.append("Authorized payout is negative or exceeds the simplified net ceiling.")
        if recommended < 0 or recommended > ceiling + 0.01:
            factual_score -= 4.0
            compliance_score -= 5.0
            issues.append("Recommended payout is negative or exceeds the simplified net ceiling.")
        if amount <= deductible and (authorized > 0 or recommended > 0):
            factual_score -= 4.0
            compliance_score -= 5.0
            issues.append("Claim is within the entered deductible but a positive payout was recommended or authorized.")
        if decision == "SIU_REFERRAL" and authorized > 0:
            factual_score -= 5.0
            compliance_score -= 6.0
            issues.append("SIU referral must not authorize disbursement while review is pending.")

        # Checklist relevance guardrails.
        claim_type = self._claim_type(claim_data)
        description = self._description(claim_data).lower()
        checklist = [str(step) for step in recommendation.get("actionable_steps", [])]
        checklist_text = " ".join(checklist).lower()
        is_theft = any(word in description for word in ("theft", "stolen", "break-in", "burglary", "robbery"))
        is_water = any(word in description for word in ("water", "burst pipe", "plumb", "flood", "leak", "leaking"))
        is_glass = any(word in description for word in ("windshield", "glass", "stone struck", "cracked windshield"))
        is_auto = claim_type == "Auto"
        is_renters_or_home = claim_type in {"Renters", "Home"}

        irrelevant_patterns: List[Tuple[bool, str]] = [
            (is_renters_or_home and is_theft and any(w in checklist_text for w in ("business interruption", "commercial casualty", "water mitigation", "plumber repair invoice")),
             "Checklist contains commercial or water-damage steps for a theft claim."),
            (is_renters_or_home and is_water and any(w in checklist_text for w in ("business interruption", "commercial casualty", "driver exchange", "body shop repair")),
             "Checklist contains unrelated commercial or auto-collision steps for a water-damage claim."),
            (is_auto and is_glass and any(w in checklist_text for w in ("dwelling policy", "landlord", "business interruption", "body shop collision")),
             "Checklist contains steps unrelated to an auto-glass claim."),
            (is_auto and any(w in checklist_text for w in ("dwelling policy", "landlord maintenance", "commercial casualty", "business interruption")),
             "Checklist contains steps unrelated to an auto claim."),
            (is_renters_or_home and any(w in checklist_text for w in ("commercial casualty", "business interruption logs", "cpa-verified commercial")),
             "Checklist contains commercial-claims requirements for a personal-lines claim."),
        ]
        for condition, message in irrelevant_patterns:
            if condition and message not in issues:
                issues.append(message)
                completeness_score -= 3.0

        if not checklist:
            completeness_score -= 4.0
            issues.append("Recommendation has no actionable checklist.")
        if not recommendation.get("action_statement"):
            completeness_score -= 2.0
            issues.append("Recommendation is missing an action statement.")

        # The system should not label global feature importance as an individual cause.
        drivers = risk_analysis.get("risk_drivers", []) or []
        if any("global feature importance" in str(driver).lower() for driver in drivers):
            narrative = str(risk_analysis.get("risk_narrative", "")).lower()
            if "not case-specific" not in narrative and "model-wide" not in narrative:
                issues.append("Global feature importance is presented without clarifying that it is not a case-specific explanation.")
                completeness_score -= 1.5

        factual_score = max(0.0, min(10.0, factual_score))
        completeness_score = max(0.0, min(10.0, completeness_score))
        compliance_score = max(0.0, min(10.0, compliance_score))
        overall_score = round((factual_score + completeness_score + compliance_score) / 3.0, 1)
        verdict = "PASS" if overall_score >= 8.0 and not issues else "FLAGGED_FOR_AUDIT"

        critique = (
            "Deterministic validation passed the configured checks. Coverage and payment still require authorized policy review."
            if not issues
            else "Audit warning: " + "; ".join(issues)
        )
        llm_critique = None
        if self.client:
            judge_system = (
                "You are an independent quality auditor for an insurance claim triage assistant. "
                "Check incident-specific checklist relevance, financial arithmetic, unsupported assumptions, "
                "and whether model-wide feature importance is misrepresented as a local explanation. "
                "Do not infer coverage from incomplete policy information."
            )
            judge_prompt = (
                f"Claim data: {json.dumps(claim_data, default=str)}\n"
                f"Investigation: {json.dumps(investigation, default=str)}\n"
                f"Risk analysis: {json.dumps(risk_analysis, default=str)}\n"
                f"Recommendation: {json.dumps(recommendation, default=str)}\n"
                f"Deterministic issues: {json.dumps(issues)}\n"
                "Return a concise critique, specifically identifying any irrelevant checklist items."
            )
            llm_critique = self._call_llm(judge_system, judge_prompt)
            if llm_critique:
                critique = llm_critique

        return {
            "judge_name": "Claims Quality Audit",
            "judge_mode": "rules_plus_llm" if llm_critique else "deterministic_rules_only",
            "verdict": verdict,
            "overall_quality_score": overall_score,
            "metric_scores": {
                "factual_consistency": round(factual_score, 1),
                "completeness": round(completeness_score, 1),
                "policy_compliance": round(compliance_score, 1),
            },
            "audit_critique": critique,
            "detected_issues": issues,
        }

    def process_claim(
        self, claim_data: Dict[str, Any], top_k_similar: int = 5
    ) -> Dict[str, Any]:
        """Execute classification, hybrid retrieval, ML prediction, and agent workflow."""
        classification = classification_agent.classify_and_prioritize(claim_data)

        claim_type = self._claim_type(claim_data)
        claim_state = claim_data.get("state")
        claim_amount = max(0.0, self._number(claim_data.get("claim_amount")))
        description = self._description(claim_data)
        if description:
            query = (
                f"{claim_type} insurance claim in {claim_state}: {description} "
                f"(loss amount ${claim_amount:,.2f})"
            )
        else:
            query = f"{claim_type} insurance claim in state {claim_state}, amount ${claim_amount:,.2f}"

        min_amount_filter = None
        max_amount_filter = None
        if claim_amount >= 25_000.0:
            min_amount_filter = 12_000.0
        elif 0 < claim_amount <= 5_000.0:
            max_amount_filter = 8_000.0

        similar_claims = hybrid_retriever.search(
            query=query,
            top_k=top_k_similar,
            claim_type=claim_type,
            state=claim_state,
            min_amount=min_amount_filter,
            max_amount=max_amount_filter,
        ) or []

        # If financial filtering yields too few peers, retry without the amount filter.
        if len(similar_claims) < 2 and (min_amount_filter is not None or max_amount_filter is not None):
            similar_claims = hybrid_retriever.search(
                query=query,
                top_k=top_k_similar,
                claim_type=claim_type,
                state=claim_state,
            ) or []

        ml_results = claims_ml_service.predict(claim_data) or {}
        investigation = self.run_investigation_agent(claim_data, similar_claims)
        risk_analysis = self.run_risk_agent(claim_data, ml_results, investigation)
        recommendation = self.run_recommendation_agent(
            claim_data, investigation, risk_analysis, ml_results
        )
        judge_review = self.run_llm_as_judge(
            claim_data, investigation, risk_analysis, recommendation
        )

        return {
            "claim_id": claim_data.get("claim_id", "NEW_CLAIM"),
            "classification": classification,
            "ml_predictions": ml_results,
            "similar_historical_claims": similar_claims,
            "investigation": investigation,
            "risk_assessment": risk_analysis,
            "recommendation": recommendation,
            "llm_as_judge_review": judge_review,
        }


multi_agent_workflow = MultiAgentClaimsWorkflow()
