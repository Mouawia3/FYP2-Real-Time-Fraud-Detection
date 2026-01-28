import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import threading

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


class TransactionValidationError(Exception):
    """Raised when transaction input fails validation."""
    pass


class ComplianceStatus(Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"


@dataclass
class RiskThresholds:
    """Immutable threshold configuration for a risk profile."""
    global_risk: float
    very_high_risk: float

    @classmethod
    def from_profile(cls, profile: str) -> "RiskThresholds":
        profiles = {
            "Conservative": cls(global_risk=0.60, very_high_risk=0.90),
            "Normal": cls(global_risk=0.70, very_high_risk=0.95),
            "Aggressive": cls(global_risk=0.80, very_high_risk=0.97),
        }
        return profiles.get(profile, profiles["Normal"])


@dataclass
class ParsedTransaction:
    """Structured representation of a transaction."""
    raw_text: str
    tx_type: str
    amount: float
    name_orig: str
    name_dest: str
    step: int
    old_balance: Optional[float] = None
    new_balance: Optional[float] = None


@dataclass
class RiskAssessment:
    """Output from the risk profiling agent."""
    fraud_probability: float
    risk_score: float
    model_error: Optional[str] = None

    @property
    def had_model_error(self) -> bool:
        return self.model_error is not None


@dataclass
class ComplianceDecision:
    """Output from the compliance checking agent."""
    status: ComplianceStatus
    reason_code: str
    reason: str


class FraudAgentSystem:
    """
    5-Agent Fraud Detection System

    Pipeline:
    1. Data Curator - Parse and validate transaction
    2. Risk Profiler - ML-based fraud probability
    3. Compliance Checker - Business rules engine
    4. Robo-Advisor - Action recommendation
    5. Conversational - Human-readable response
    """

    # Amount thresholds (RM)
    SMALL_AMOUNT_LIMIT = 1_000.0
    MEDIUM_AMOUNT_LIMIT = 10_000.0
    HIGH_AMOUNT_LIMIT = 50_000.0
    TRANSFER_SMALL_LIMIT = 200.0

    # Fallback fraud probability when model fails
    MODEL_ERROR_FALLBACK_PROB = 0.5  # Conservative: triggers review

    def __init__(self, model_path: str = "models/qlora_fraud_real_final"):
        print("🔮 INITIALIZING 5-AGENT FRAUD SYSTEM...")

        self._load_model(model_path)
        self._ledger_lock = threading.Lock()
        self._account_balances: Dict[str, float] = {}

    def _load_model(self, adapter_path: str) -> None:
        """Load the base model and LoRA adapter."""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._base_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2,
        )

        try:
            self._model = PeftModel.from_pretrained(
                self._base_model,
                adapter_path,
            )
            print(f"✅ Loaded LoRA adapter from {adapter_path}")
        except Exception as e:
            print(f"⚠️ Failed to load adapter: {e}. Using base model only.")
            self._model = self._base_model

        self._model.to(self._device)
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # ──────────────────────────────────────────────────────────────
    # Agent 1: Data Curator
    # ──────────────────────────────────────────────────────────────

    def _parse_transaction(self, transaction: str) -> ParsedTransaction:
        """
        Parse transaction string into structured data.

        Expected format:
            "TYPE AMOUNT from ACCOUNT to ACCOUNT step:N oldOrg:X newOrg:Y"
        """
        parts = transaction.split()

        if len(parts) < 2:
            raise TransactionValidationError("Transaction string too short")

        tx_type = parts[0].upper()
        if tx_type not in {"PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"}:
            raise TransactionValidationError(f"Unknown transaction type: {tx_type}")

        try:
            amount = float(parts[1])
            if amount <= 0:
                raise TransactionValidationError("Amount must be positive")
        except ValueError:
            raise TransactionValidationError(f"Invalid amount: {parts[1]}")

        # Parse named fields
        fields = self._extract_fields(parts)

        if not fields.get("from"):
            raise TransactionValidationError("Missing 'from' account")
        if not fields.get("to"):
            raise TransactionValidationError("Missing 'to' account")
        if fields.get("step") is None:
            raise TransactionValidationError("Missing step number")
        if not (1 <= fields["step"] <= 1000):
            raise TransactionValidationError("Step must be between 1 and 1000")

        return ParsedTransaction(
            raw_text=transaction,
            tx_type=tx_type,
            amount=amount,
            name_orig=fields["from"],
            name_dest=fields["to"],
            step=fields["step"],
            old_balance=fields.get("old_org"),
            new_balance=fields.get("new_org"),
        )

    def _extract_fields(self, parts: list) -> Dict[str, Any]:
        """Extract named fields from transaction parts."""
        fields: Dict[str, Any] = {}

        for i, token in enumerate(parts):
            lower = token.lower()

            if lower == "from" and i + 1 < len(parts):
                fields["from"] = parts[i + 1]
            elif lower == "to" and i + 1 < len(parts):
                fields["to"] = parts[i + 1]
            elif lower.startswith("step:"):
                try:
                    fields["step"] = int(token.split(":", 1)[1])
                except ValueError:
                    pass
            elif lower.startswith("oldorg:"):
                try:
                    fields["old_org"] = float(token.split(":", 1)[1])
                except ValueError:
                    pass
            elif lower.startswith("neworg:"):
                try:
                    fields["new_org"] = float(token.split(":", 1)[1])
                except ValueError:
                    pass

        return fields

    # ──────────────────────────────────────────────────────────────
    # Agent 2: Risk Profiler
    # ──────────────────────────────────────────────────────────────

    def _assess_risk(self, transaction: ParsedTransaction) -> RiskAssessment:
        """
        Compute fraud probability using the ML model.
        """
        try:
            inputs = self._tokenizer(
                transaction.raw_text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
                fraud_prob = probs[1].item()

            risk_score = fraud_prob * (transaction.amount / 1e6)

            return RiskAssessment(
                fraud_probability=fraud_prob,
                risk_score=risk_score,
            )

        except Exception as e:
            # Use conservative fallback on model error
            return RiskAssessment(
                fraud_probability=self.MODEL_ERROR_FALLBACK_PROB,
                risk_score=self.MODEL_ERROR_FALLBACK_PROB * (transaction.amount / 1e6),
                model_error=str(e),
            )

    # ──────────────────────────────────────────────────────────────
    # Agent 3: Compliance Checker
    # ──────────────────────────────────────────────────────────────

    def _check_compliance(
            self,
            transaction: ParsedTransaction,
            risk: RiskAssessment,
            thresholds: RiskThresholds,
    ) -> ComplianceDecision:
        """
        Apply business rules to determine transaction disposition.
        """
        prob = risk.fraud_probability
        amount = transaction.amount
        tx_type = transaction.tx_type

        # Derived flags
        high_fraud = prob >= thresholds.global_risk
        very_high_fraud = prob >= thresholds.very_high_risk
        is_small = amount < self.SMALL_AMOUNT_LIMIT
        is_medium = self.SMALL_AMOUNT_LIMIT <= amount < self.MEDIUM_AMOUNT_LIMIT
        is_high = amount >= self.HIGH_AMOUNT_LIMIT

        # Special handling for small transfers
        if tx_type == "TRANSFER" and amount < self.TRANSFER_SMALL_LIMIT:
            return self._handle_small_transfer(prob)

        # General rules (ordered by severity)
        if very_high_fraud and not is_small:
            return ComplianceDecision(
                status=ComplianceStatus.BLOCKED,
                reason_code="G1",
                reason="Very high fraud probability on non-trivial amount",
            )

        if very_high_fraud and is_small:
            return ComplianceDecision(
                status=ComplianceStatus.REVIEW,
                reason_code="G2",
                reason="Very high fraud probability but small amount - review required",
            )

        if high_fraud and is_high:
            return ComplianceDecision(
                status=ComplianceStatus.BLOCKED,
                reason_code="G3",
                reason="High fraud probability combined with large amount",
            )

        if high_fraud and is_medium:
            return ComplianceDecision(
                status=ComplianceStatus.REVIEW,
                reason_code="G4",
                reason="High fraud probability on medium-value transaction",
            )

        if high_fraud and is_small:
            return ComplianceDecision(
                status=ComplianceStatus.PASS,
                reason_code="G5",
                reason="High fraud probability but low financial impact - approved with monitoring",
            )

        if is_high:
            return ComplianceDecision(
                status=ComplianceStatus.REVIEW,
                reason_code="G6",
                reason="Large amount requires precautionary review",
            )

        return ComplianceDecision(
            status=ComplianceStatus.PASS,
            reason_code="G0",
            reason="Within risk tolerance",
        )

    def _handle_small_transfer(self, fraud_prob: float) -> ComplianceDecision:
        """Special relaxed rules for small transfers."""
        if fraud_prob >= 0.98:
            return ComplianceDecision(
                status=ComplianceStatus.REVIEW,
                reason_code="T1",
                reason="Extremely high fraud probability on small transfer",
            )
        if fraud_prob >= 0.95:
            return ComplianceDecision(
                status=ComplianceStatus.REVIEW,
                reason_code="T2",
                reason="High fraud probability on small transfer - manual review",
            )
        if fraud_prob >= 0.70:
            return ComplianceDecision(
                status=ComplianceStatus.PASS,
                reason_code="T3",
                reason="Moderate risk but low impact - approved with monitoring",
            )
        return ComplianceDecision(
            status=ComplianceStatus.PASS,
            reason_code="T0",
            reason="Small transfer within tolerance",
        )

    # ──────────────────────────────────────────────────────────────
    # Agent 4: Robo-Advisor
    # ──────────────────────────────────────────────────────────────

    def _get_recommendation(self, decision: ComplianceDecision) -> str:
        """Convert compliance decision to action recommendation."""
        recommendations = {
            ComplianceStatus.PASS: "✅ APPROVE transaction",
            ComplianceStatus.REVIEW: "⚠️ HOLD for manual review",
            ComplianceStatus.BLOCKED: "🚨 FREEZE account + Alert compliance",
            ComplianceStatus.ERROR: "❌ Transaction rejected due to error",
        }
        return recommendations.get(decision.status, "❓ Unknown status")

    # ──────────────────────────────────────────────────────────────
    # Agent 5: Conversational Response
    # ──────────────────────────────────────────────────────────────

    def _format_response(
            self,
            transaction: ParsedTransaction,
            risk: RiskAssessment,
            decision: ComplianceDecision,
    ) -> str:
        """Generate human-readable response."""
        status_text = {
            ComplianceStatus.PASS: "approved ✅",
            ComplianceStatus.REVIEW: "held for review ⚠️",
            ComplianceStatus.BLOCKED: "BLOCKED 🚨",
            ComplianceStatus.ERROR: "rejected ❌",
        }


        lines = [
            f"{transaction.tx_type} of RM{transaction.amount:,.2f} is {status_text[decision.status]}.",
            f"🤖 [Agent 2 - AI Model] Fraud Probability: {risk.fraud_probability:.2%}",
            f"📜 [Agent 3 - Rules Engine] Compliance: {decision.status.value}",
            f"📝 [Agent 3 - Reason] {decision.reason}",
        ]


        if risk.had_model_error:
            lines.append(f"⚠️ Model error occurred: {risk.model_error}")

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────
    # Ledger Management (Thread-Safe)
    # ──────────────────────────────────────────────────────────────

    def _update_ledger(
            self,
            transaction: ParsedTransaction,
            decision: ComplianceDecision,
    ) -> Dict[str, Optional[float]]:
        """Update internal balance ledger (thread-safe)."""
        with self._ledger_lock:
            account = transaction.name_orig

            # Get current balance
            current = self._account_balances.get(account)
            if current is None:
                current = transaction.old_balance or 0.0

            before = current

            # Only update if approved
            if decision.status == ComplianceStatus.PASS:
                if transaction.tx_type in {"PAYMENT", "TRANSFER", "CASH_OUT"}:
                    current = max(current - transaction.amount, 0.0)
                elif transaction.tx_type == "CASH_IN":
                    current += transaction.amount

                self._account_balances[account] = current

            return {"before": before, "after": current}

    # ──────────────────────────────────────────────────────────────
    # Main Pipeline
    # ──────────────────────────────────────────────────────────────

    def process_transaction(
            self,
            transaction_str: str,
            profile: str = "Normal",
    ) -> Dict[str, Any]:
        """
        Process a transaction through the full 5-agent pipeline.

        Thread-safe: thresholds are passed as parameters, not instance state.
        """
        start = time.perf_counter()
        thresholds = RiskThresholds.from_profile(profile)
        trace: Dict[str, Any] = {"profile": profile}

        # Agent 1: Parse and validate
        try:
            transaction = self._parse_transaction(transaction_str)
            trace["parsed"] = {
                "type": transaction.tx_type,
                "amount": transaction.amount,
                "from": transaction.name_orig,
                "to": transaction.name_dest,
            }
        except TransactionValidationError as e:
            return self._error_response(str(e), start, trace)

        # Agent 2: Risk assessment
        risk = self._assess_risk(transaction)
        trace["risk"] = {
            "fraud_prob": risk.fraud_probability,
            "risk_score": risk.risk_score,
            "model_error": risk.model_error,
        }

        # Agent 3: Compliance check
        decision = self._check_compliance(transaction, risk, thresholds)
        trace["compliance"] = {
            "status": decision.status.value,
            "reason_code": decision.reason_code,
        }

        # Agent 4: Recommendation
        recommendation = self._get_recommendation(decision)
        trace["recommendation"] = recommendation

        # Agent 5: Response
        response = self._format_response(transaction, risk, decision)

        # Update ledger
        ledger = self._update_ledger(transaction, decision)

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "final_message": response,
            "status": decision.status.value,
            "reason_code": decision.reason_code,
            "reason": decision.reason,
            "fraud_prob": risk.fraud_probability,
            "risk_score": risk.risk_score,
            "latency_ms": latency_ms,
            "tx_type": transaction.tx_type,
            "amount": transaction.amount,
            "name_orig": transaction.name_orig,
            "name_dest": transaction.name_dest,
            "step": transaction.step,
            "ledger_before": ledger["before"],
            "ledger_after": ledger["after"],
            "trace": trace,
        }

    def _error_response(
            self,
            error_msg: str,
            start_time: float,
            trace: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate standardized error response."""
        trace["error"] = error_msg
        return {
            "final_message": f"Rejected: {error_msg}",
            "status": "ERROR",
            "reason_code": "INVALID_INPUT",
            "reason": error_msg,
            "fraud_prob": 0.0,
            "risk_score": 0.0,
            "latency_ms": (time.perf_counter() - start_time) * 1000,
            "tx_type": "UNKNOWN",
            "amount": 0.0,
            "name_orig": None,
            "name_dest": None,
            "step": None,
            "ledger_before": None,
            "ledger_after": None,
            "trace": trace,
        }


# ══════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    system = FraudAgentSystem()

    test_cases = [
        ("CASH_OUT 25000 from C1399554611 to C988696172 step:207 oldOrg:38427 newOrg:13427", "Normal"),
        ("PAYMENT 50.00 from C1234567890 to M4455667788 step:887 oldOrg:5000 newOrg:4950", "Normal"),
        ("TRANSFER 170.00 from C1234567890 to C9988776655 step:200 oldOrg:4950 newOrg:4900", "Conservative"),
    ]

    print("\n🎬 5-AGENT FRAUD DETECTION SYSTEM")
    print("=" * 60)

    for i, (tx, profile) in enumerate(test_cases, 1):
        print(f"\n📱 CASE {i} (Profile: {profile}):")
        print(f"   Input: {tx[:60]}...")
        result = system.process_transaction(tx, profile)
        print(f"   Status: {result['status']}")
        print(f"   Fraud Prob: {result['fraud_prob']:.2%}")
        print(f"   Latency: {result['latency_ms']:.1f}ms")
        print(f"   Reason: {result['reason']}")

    print("\n🏆 Demo complete!")