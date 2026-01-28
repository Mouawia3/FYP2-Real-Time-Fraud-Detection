import time
import json
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

from five_agent_system import FraudAgentSystem


# ============================================================
# Session state initialisation
# ============================================================
if "wallet_balances" not in st.session_state:
    # Dict: wallet_id -> balance
    st.session_state.wallet_balances = {}

if "tx_history" not in st.session_state:
    # List of transaction dicts (customer-facing + backend)
    st.session_state.tx_history = []

if "last_transaction" not in st.session_state:
    # Most recent transaction made from the Customer App
    st.session_state.last_transaction = None


# ============================================================
# Cached loaders
# ============================================================
@st.cache_resource
def load_model():
    """
    Load DistilBERT + LoRA adapter once.
    """
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
    )
    model = PeftModel.from_pretrained(base_model, "models/qlora_fraud_real_final")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model.eval()
    return model, tokenizer


@st.cache_resource
def load_agent_system():
    """
    Instantiate the 5-agent fraud system once.
    """
    return FraudAgentSystem()

@st.cache_data
def load_eval_metrics_json():
    """
    Load evaluation metrics produced by validate_model_accuracy.py
    """
    try:
        with open("results/eval_all.json", "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Helpers
# ============================================================
def build_transaction_text(
    tx_type: str,
    amount: float,
    name_orig: str,
    name_dest: str,
    step: int,
    oldbalance_org: float,
    newbalance_org: float,
) -> str:
    """
    Build transaction description string in the same format used for training:

    "{type} {amount:.2f} from {nameOrig} to {nameDest} "
    "step:{step} oldOrg:{oldbalanceOrg:.0f} newOrg:{newbalanceOrg:.0f}"
    """
    return (
        f"{tx_type} {amount:.2f} from {name_orig} to {name_dest} "
        f"step:{step} oldOrg:{oldbalance_org:.0f} newOrg:{newbalance_org:.0f}"
    )


def get_step_risk(step: int):
    """
    Map step (1–1000) to a simple time-risk category.
    """
    if step <= 300:
        return "Low", "🟢", "Early in the simulation timeline where most behaviour is normal."
    elif step <= 700:
        return "Medium", "🟠", "Mid simulation period where unusual behaviours start to appear."
    else:
        return "High", "🔴", "Late in the simulation timeline where fraud scenarios are more frequent."


def render_step_risk(step: int, context_label: str = "Time risk window"):
    """
    Render a small time-risk badge and a progress bar for the given step.
    """
    risk_level, emoji, explanation = get_step_risk(step)
    st.markdown(
        f"**{context_label}:** {emoji} **{risk_level}** "
        f"(step {step} / 1000)  \n"
        f"<span style='font-size: 0.9em; color: #666;'>{explanation}</span>",
        unsafe_allow_html=True,
    )
    st.progress(min(max(step / 1000.0, 0.0), 1.0))


def parse_status_from_response(response_text: str) -> str:
    """
    Interpret FraudAgentSystem's final conversational response into a status:
    APPROVED / HOLD / BLOCKED
    """
    txt = response_text.lower()
    if "approved" in txt and "hold" not in txt and "blocked" not in txt:
        return "APPROVED"
    if "hold" in txt or "review" in txt:
        return "HOLD"
    return "BLOCKED"


# ============================================================
# App setup
# ============================================================
st.set_page_config(page_title="FYP2 Fraud Detection", layout="wide")

st.title("🏦 FYP2 Real-time Fraud Detection System")
st.caption(
    "Customer App + Fraud Operations Console · DistilBERT + LoRA + Policy Engine · "
    "Balanced Val F1 ≈ 0.996 | Real-world Imbalanced Test F1 ≈ 0.443 (precision≈0.285) · "
    "Real-time scoring target ≤ 400 ms"
)


model, tokenizer = load_model()
agent_system = load_agent_system()

# Global view selector
mode = st.sidebar.radio(
    "Select view",
    ["Customer App", "Fraud Ops Console"],
)


# ============================================================
# VIEW 1 – Customer App (E-wallet style)
# ============================================================
def render_customer_app():
    st.header("💳 Customer Wallet App")

    # ---------------- Wallet Selection ----------------
    st.subheader("Wallet Profile")

    col_profile_1, col_profile_2 = st.columns([2, 1])

    with col_profile_1:
        wallet_id = st.text_input(
            "Your Wallet ID",
            value="C1234567890",
            key="cust_wallet_id",
        )

    with col_profile_2:
        starting_balance = st.number_input(
            "Set / Update Balance (RM)",
            min_value=0.0,
            value=5000.0,
            step=50.0,
            key="cust_start_balance",
        )

    if st.button("💾 Save Wallet Balance"):
        st.session_state.wallet_balances[wallet_id] = float(starting_balance)
        st.success(
            f"Wallet {wallet_id} balance set to RM{starting_balance:,.2f}"
        )

    current_balance = st.session_state.wallet_balances.get(wallet_id, 0.0)

    # Wallet summary card
    st.markdown("---")
    col_card_1, col_card_2 = st.columns([2, 1])
    with col_card_1:
        st.markdown("#### Wallet Summary")
        st.markdown(
            f"""
            **Wallet ID:** `{wallet_id}`  
            **Current Balance:** 💰 **RM{current_balance:,.2f}**
            """
        )

    with col_card_2:
        last_tx = st.session_state.last_transaction
        st.markdown("#### Last Transaction Status")
        if last_tx is None:
            st.caption("No transactions yet.")
        else:
            st.markdown(
                f"- Time: **{last_tx['time']}**  \n"
                f"- Type: **{last_tx['type']}**  \n"
                f"- Amount: **RM{last_tx['amount']:,.2f}**  \n"
                f"- Status: **{last_tx['status']}**"
            )

    st.markdown("---")

    # ---------------- Action Selection ----------------
    st.subheader("Make a Transaction")

    profile = st.selectbox(
        "Risk Profile (Personalised Finance)",
        ["Conservative", "Normal", "Aggressive"],
        index=1,
        help="Personalises decision thresholds (Conservative=stricter, Aggressive=more permissive).",
    )


    if profile == "Conservative":
        st.info(
            "🛡️ **Profile: CONSERVATIVE**\n\nSystem will flag fraud aggressively (Threshold: 60%). You prioritize security over convenience.")
    elif profile == "Aggressive":
        st.warning(
            "⚡ **Profile: AGGRESSIVE**\n\nSystem allows riskier transactions (Threshold: 80%). You prioritize speed.")
    else:
        st.caption("⚖️ **Profile: NORMAL** (Threshold: 70%) - Balanced security.")


    action = st.radio(
        "Choose action",
        ["Pay Merchant", "Transfer to Friend", "Cash-out to Bank"],
        key="cust_action",
        horizontal=True,
    )

    if action == "Pay Merchant":
        tx_type = "PAYMENT"
    elif action == "Transfer to Friend":
        tx_type = "TRANSFER"
    else:
        tx_type = "CASH_OUT"

    col_form_1, col_form_2 = st.columns([2, 1])

    with col_form_1:
        if tx_type == "PAYMENT":
            to_label = "Merchant ID"
            default_dest = "M4455667788"
        else:
            to_label = "Recipient ID"
            default_dest = "C9988776655"

        to_wallet = st.text_input(
            to_label,
            value=default_dest,
            key="cust_to_wallet",
        )

        amount = st.number_input(
            "Amount (RM)",
            min_value=1.0,
            step=10.0,
            value=100.0,
            key="cust_amount",
        )

        # Simulation step (time index)
        step_val = st.slider(
            "Simulation time index (step)",
            1,
            1000,
            250,
            key="cust_step",
        )
        render_step_risk(step_val, context_label="Time risk (simulation step)")

    with col_form_2:
        st.markdown("##### Transaction Preview")

        # Assume new balance if approved
        if tx_type in ["PAYMENT", "TRANSFER", "CASH_OUT"]:
            new_balance = max(current_balance - amount, 0.0)
        else:
            new_balance = current_balance + amount

        st.write(f"From: `{wallet_id}`")
        st.write(f"To: `{to_wallet}`")
        st.write(f"Type: **{tx_type}** (from action: {action})")
        st.write(f"Amount: **RM{amount:,.2f}**")
        st.write(f"Current balance: **RM{current_balance:,.2f}**")
        st.write(
            f"Balance if approved: **RM{new_balance:,.2f}**"
        )

    st.markdown("")

    # ---------------- Execute transaction ----------------
    if st.button("✅ Confirm & Send"):
        # HARD VALIDATION: balance
        if current_balance <= 0:
            st.error(
                "❌ Payment failed: insufficient balance (RM0). "
                "Please set or top up your wallet balance above."
            )
            st.stop()

        if amount > current_balance:
            st.error(
                f"❌ Payment failed: amount RM{amount:.2f} exceeds "
                f"current balance RM{current_balance:.2f}. "
                "Transaction not sent to fraud engine."
            )
            st.stop()

        # Build model description
        description = build_transaction_text(
            tx_type,
            amount,
            wallet_id,
            to_wallet,
            step_val,
            current_balance,
            new_balance,
        )

        # Call 5-agent system and get structured result
        result = agent_system.process_transaction(description, profile=profile)

        response_text = result["final_message"]          # text from Agent 5
        engine_status = result["status"]                 # PASS / REVIEW / BLOCKED
        reason = result["reason"]                        # explanation from Agent 3
        fraud_prob = result["fraud_prob"]
        latency_ms = result["latency_ms"]

        # Map PASS/REVIEW/BLOCKED → customer-facing status
        if engine_status == "PASS":
            status = "APPROVED"
        elif engine_status == "REVIEW":
            status = "HOLD"
        else:
            status = "BLOCKED"

        # Update wallet balance only if APPROVED
        if status == "APPROVED":
            st.session_state.wallet_balances[wallet_id] = float(new_balance)
            final_balance = new_balance
            st.success("✅ Transaction Successful")
        elif status == "HOLD":
            final_balance = current_balance
            st.warning("⚠️ Transaction On Hold for Review")
            st.info("Your balance has not been deducted yet.")
        else:
            final_balance = current_balance
            st.error("🚨 Transaction Blocked")
            st.info("Your balance remains unchanged.")

        st.markdown("**Summary (customer view):**")
        st.write(response_text)
        st.metric("End-to-end decision latency (ms)", f"{latency_ms:.2f}")
        with st.expander("Show 5-Agent Trace (for viva)"):
            st.json(result.get("trace", {}))


        # Log transaction in shared history INCLUDING REASON & ENGINE STATUS
        tx_record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "profile": profile,
            "from": wallet_id,
            "to": to_wallet,
            "type": tx_type,
            "amount": float(amount),
            "balance_before": float(current_balance),
            "balance_after": float(final_balance),
            "status": status,                # APPROVED / HOLD / BLOCKED (customer view)
            "engine_status": engine_status,  # PASS / REVIEW / BLOCKED (agent 3)
            "reason": reason,                # reason from compliance agent
            "fraud_prob": float(fraud_prob),
            "latency_ms": float(latency_ms),
            "description": description,
            "step": int(step_val),
        }
        st.session_state.tx_history.append(tx_record)
        st.session_state.last_transaction = tx_record


    # ---------------- Customer View History ----------------
    st.markdown("---")
    st.subheader("Recent Transactions (Customer View)")

    if st.session_state.tx_history:
        df = pd.DataFrame(st.session_state.tx_history)
        df = df.sort_values("time", ascending=False)
        st.dataframe(
            df[["time", "type", "amount", "status", "balance_before", "balance_after"]],
            use_container_width=True,
        )
    else:
        st.caption("No transactions yet. Make a transaction to see history.")


# ============================================================
# VIEW 2 – Fraud Ops Console (backend / examiner)
# ============================================================
def render_fraud_console():
    st.header("🧠 Fraud Operations Console")

    subtab_cases, subtab_analysis, subtab_metrics, subtab_agents, subtab_investigator = st.tabs(
        [
            "📂 Cases",
            "🔍 Single Transaction Analysis",
            "📊 Model Performance",
            "🎬 5-Agent Demo",
            "🕵️ Fraud Investigator",
        ]
    )
    # ------------------------------------
    # FRAUD INVESTIGATOR VIEW
    # ------------------------------------
    with subtab_investigator:
        st.subheader("🕵️ Fraud Investigator – Review HOLD/BLOCKED Cases")

        if not st.session_state.tx_history:
            st.caption("No transactions yet. Use the Customer App to generate activity.")
        else:
            df_all = pd.DataFrame(st.session_state.tx_history)

            # Only show transactions where customer-facing status is HOLD or BLOCKED
            mask_risky = df_all["status"].isin(["HOLD", "BLOCKED"])
            df_risky = df_all[mask_risky].copy()

            if df_risky.empty:
                st.info("Currently there are no HOLD or BLOCKED transactions to review.")
            else:
                # Show summary table for investigators
                st.markdown("### Risky Transactions Queue")
                st.dataframe(
                    df_risky[
                        [
                            "time",
                            "from",
                            "to",
                            "type",
                            "amount",
                            "status",          # customer status
                            "engine_status",   # PASS / REVIEW / BLOCKED
                            "reason",          # explanation from Agent 3
                            "fraud_prob",
                            "latency_ms",
                        ]
                    ].sort_values("time", ascending=False),
                    use_container_width=True,
                )

                # Selection for deep inspection
                st.markdown("### Inspect a Case in Detail")

                idx_options_inv = df_risky.index.tolist()
                selected_idx_inv = st.selectbox(
                    "Select case",
                    options=idx_options_inv,
                    format_func=lambda i: (
                        f"[{df_risky.loc[i, 'time']}] "
                        f"{df_risky.loc[i, 'type']} RM{df_risky.loc[i, 'amount']:.2f} "
                        f"({df_risky.loc[i, 'status']}, prob={df_risky.loc[i, 'fraud_prob']:.2%})"
                    ),
                )

                case = df_risky.loc[selected_idx_inv]

                # High-level summary
                st.markdown("#### Case Summary")
                st.write(
                    f"- Time: **{case['time']}**  \n"
                    f"- From: **{case['from']}**  \n"
                    f"- To: **{case['to']}**  \n"
                    f"- Type: **{case['type']}**  \n"
                    f"- Amount: **RM{case['amount']:,.2f}**  \n"
                    f"- Customer status: **{case['status']}**  \n"
                    f"- Engine status: **{case.get('engine_status', 'N/A')}**  \n"
                    f"- Model fraud probability: **{case.get('fraud_prob', 0.0):.2%}**  \n"
                    f"- Decision reason (note for analyst): **{case.get('reason', 'N/A')}**  \n"
                    f"- Balance before: **RM{case['balance_before']:,.2f}**  \n"
                    f"- Balance after (customer view): **RM{case['balance_after']:,.2f}**  \n"
                    f"- Inference latency: **{case.get('latency_ms', 0.0):.2f} ms**"
                )

                # Step / time risk visual
                step_val = case.get("step", None)
                if step_val is not None:
                    render_step_risk(int(step_val), context_label="Time risk (this transaction)")

                # Raw model input
                st.markdown("#### Model Input Description")
                st.code(case["description"], language="text")

                # investigator label
                st.markdown("#### Investigator Verdict (for demonstration)")
                verdict = st.radio(
                    "Mark this decision as:",
                    ["Unreviewed", "True Fraud", "False Positive"],
                    horizontal=True,
                )
                st.caption(
                    "In a production system, this manual label would be stored and later used "
                    "for audit, feedback to the model, and policy refinement."
                )


    # ------------------------------------
    # CASES VIEW
    # ------------------------------------
    with subtab_cases:
        st.subheader("Fraud Cases & Transaction Log")

        if not st.session_state.tx_history:
            st.caption("No transactions yet. Use the Customer App to generate activity.")
            return

        df = pd.DataFrame(st.session_state.tx_history)
        df = df.sort_values("time", ascending=False)

        # Filters
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            status_filter = st.multiselect(
                "Filter by status",
                options=sorted(df["status"].unique().tolist()),
                default=sorted(df["status"].unique().tolist()),
            )
        with col_f2:
            type_filter = st.multiselect(
                "Filter by transaction type",
                options=sorted(df["type"].unique().tolist()),
                default=sorted(df["type"].unique().tolist()),
            )

        mask = df["status"].isin(status_filter) & df["type"].isin(type_filter)
        filtered_df = df[mask]

        st.dataframe(
            filtered_df[
                [
                    "time",
                    "from",
                    "to",
                    "type",
                    "amount",
                    "status",
                    "balance_before",
                    "balance_after",
                    "latency_ms",
                ]
            ],
            use_container_width=True,
        )

        # Select a transaction to inspect
        st.markdown("### Inspect a Specific Transaction")
        idx_options = filtered_df.index.tolist()
        if idx_options:
            selected_idx = st.selectbox(
                "Select row index",
                options=idx_options,
                format_func=lambda i: f"Row {i} – {filtered_df.loc[i, 'time']} | "
                                      f"{filtered_df.loc[i, 'type']} RM{filtered_df.loc[i, 'amount']:.2f} "
                                      f"({filtered_df.loc[i, 'status']})",
            )
            tx_row = filtered_df.loc[selected_idx]

            st.markdown("#### Transaction Details")
            st.write(
                f"- Time: **{tx_row['time']}**  \n"
                f"- From: **{tx_row['from']}**  \n"
                f"- To: **{tx_row['to']}**  \n"
                f"- Type: **{tx_row['type']}**  \n"
                f"- Amount: **RM{tx_row['amount']:,.2f}**  \n"
                f"- Status: **{tx_row['status']}**  \n"
                f"- Balance before: **RM{tx_row['balance_before']:,.2f}**  \n"
                f"- Balance after: **RM{tx_row['balance_after']:,.2f}**"
            )

            step_val = tx_row.get("step", None)
            if step_val is not None:
                render_step_risk(int(step_val), context_label="Time risk (this transaction)")

            st.markdown("**Model input description:**")
            st.code(tx_row["description"], language="text")

            if st.button("🔍 Analyze this transaction with 5-Agent Engine", key="cases_analyze"):
                with st.spinner("Running full 5-agent pipeline..."):
                    start_time = time.time()
                    result = agent_system.process_transaction(tx_row["description"])
                    latency_ms = (time.time() - start_time) * 1000.0

                col_c1, col_c2, col_c3 = st.columns(3)
                col_c1.metric("Model Fraud Probability", f"{result['fraud_prob']:.2%}")
                col_c2.metric("Compliance Status", result["status"])  # PASS / REVIEW / BLOCKED
                col_c3.metric("End-to-end latency (ms)", f"{latency_ms:.2f}")

                st.markdown("**Final decision (Agent 5):**")
                st.info(result["final_message"])

                st.markdown("**Policy Reason (Agent 3):**")
                st.write(f"**Reason code:** `{result['reason_code']}`")
                st.write(f"**Reason text:** {result['reason']}")

                with st.expander("Show 5-Agent Trace (for viva)"):
                    st.json(result.get("trace", {}))

    # ------------------------------------
    # SINGLE TRANSACTION ANALYSIS
    # ------------------------------------
    with subtab_analysis:
        st.subheader("🔍 Single Transaction Analysis (Manual Input) – Model + Policy")

        col_a1, col_a2 = st.columns(2)

        with col_a1:
            tx_type_e = st.selectbox(
                "Transaction type",
                ["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER"],
                key="engine_type",
            )
            amount_e = st.number_input(
                "Amount (RM)",
                min_value=0.0,
                value=10000.0,
                step=100.0,
                key="engine_amount",
            )
            name_orig_e = st.text_input(
                "From Account",
                value="C1234567890",
                key="engine_name_orig",
            )
            name_dest_e = st.text_input(
                "To Account",
                value="M9876543210",
                key="engine_name_dest",
            )
            step_e = st.slider(
                "Step (time index)",
                1,
                1000,
                500,
                key="engine_step",
            )
            render_step_risk(step_e, context_label="Time risk (simulation step)")
            old_org_e = st.number_input(
                "Origin balance BEFORE (oldOrg)",
                min_value=0.0,
                value=100000.0,
                step=1000.0,
                key="engine_old_org",
            )
            new_org_e = st.number_input(
                "Origin balance AFTER (newOrg)",
                min_value=0.0,
                value=90000.0,
                step=1000.0,
                key="engine_new_org",
            )

            engine_text = build_transaction_text(
                tx_type_e,
                amount_e,
                name_orig_e,
                name_dest_e,
                step_e,
                old_org_e,
                new_org_e,
            )
            st.markdown(f"**Model input description:** `{engine_text}`")
            # ===================================

            # ( Error Handling code )

            with st.expander("🛠️ QA & Error Handling Simulation (Panel Req)"):
                st.write("Demonstration of system resilience against invalid inputs.")
                col_err1, col_err2 = st.columns(2)

                with col_err1:
                    if st.button("🧪 Simulate Negative Amount"):
                        # Intentionally bad input
                        res = agent_system.process_transaction(
                            f"PAYMENT -500.00 from {name_orig_e} to {name_dest_e} step:100 oldOrg:0 newOrg:0")
                        if res["status"] == "ERROR":
                            st.success(f"✅ CAUGHT: {res['final_message']}")
                        else:
                            st.error("Failed to catch error.")

                with col_err2:
                    if st.button("🧪 Simulate Garbage Text"):
                        res = agent_system.process_transaction("THIS IS NOT A VALID TRANSACTION STRING")
                        if res["status"] == "ERROR":
                            st.success(f"✅ CAUGHT: {res['reason']}")
                        else:
                            st.error("Failed to catch error.")

        with col_a2:
            if st.button("🔍 Analyze with 5-Agent Engine", key="engine_analyze"):
                with st.spinner("Running full 5-agent pipeline..."):
                    result = agent_system.process_transaction(engine_text)

                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Model Fraud Probability", f"{result['fraud_prob']:.2%}")
                col_m2.metric("Compliance Status", result["status"])
                col_m3.metric("End-to-end latency (ms)", f"{result['latency_ms']:.2f}")

                st.markdown("### Final decision (Agent 5)")
                st.info(result["final_message"])

                st.markdown("### Policy Reason (Agent 3)")
                st.write(f"**Reason code:** `{result['reason_code']}`")
                st.write(f"**Reason text:** {result['reason']}")

                with st.expander("Show 5-Agent Trace (for viva)"):
                    st.json(result.get("trace", {}))

        # ---------------- Last customer transaction quick analysis ----------------
        st.markdown("---")
        st.subheader("Last Transaction from Customer App")

        last_tx = st.session_state.last_transaction
        if last_tx is None:
            st.caption("No customer transactions yet.")
        else:
            st.write(
                f"- Time: **{last_tx['time']}**  \n"
                f"- From: **{last_tx['from']}**  \n"
                f"- To: **{last_tx['to']}**  \n"
                f"- Type: **{last_tx['type']}**  \n"
                f"- Amount: **RM{last_tx['amount']:,.2f}**  \n"
                f"- Status (customer view): **{last_tx['status']}**"
            )
            if "step" in last_tx:
                render_step_risk(last_tx["step"], context_label="Time risk (last transaction)")
            st.code(last_tx["description"], language="text")

            if st.button("🔍 Analyze last customer transaction with 5-Agent Engine", key="analyze_last"):
                with st.spinner("Running full 5-agent pipeline on last transaction..."):
                    result_last = agent_system.process_transaction(last_tx["description"])

                col_la, col_lb, col_lc = st.columns(3)
                col_la.metric("Model Fraud Probability", f"{result_last['fraud_prob']:.2%}")
                col_lb.metric("Compliance Status", result_last["status"])
                col_lc.metric("End-to-end latency (ms)", f"{result_last['latency_ms']:.2f}")

                st.markdown("### Final decision (Agent 5)")
                st.info(result_last["final_message"])

                st.markdown("### Policy Reason (Agent 3)")
                st.write(f"**Reason code:** `{result_last['reason_code']}`")
                st.write(f"**Reason text:** {result_last['reason']}")

    # ------------------------------------
    # MODEL PERFORMANCE
    # ------------------------------------
    with subtab_metrics:
        st.subheader("📊 Model Evaluation Results (Balanced vs Real-World vs OOD Shift)")

        eval_json = load_eval_metrics_json()

        if "error" in eval_json:
            st.error("Could not load results/eval_all.json")
            st.write(f"Error: {eval_json['error']}")
            st.info("Run: python validate_model_accuracy.py")
        else:
            evals = eval_json.get("evaluations", {})

            def show_eval_card(key: str, title: str):
                block = evals.get(key, {})
                ds = block.get("dataset", {})
                met = block.get("metrics", {})
                cm = block.get("confusion_matrix", {})

                st.markdown(f"### {title}")
                st.write(
                    f"- Size: **{ds.get('size', 0):,}**  \n"
                    f"- Fraud count: **{ds.get('n_fraud', 0):,}**  \n"
                    f"- Fraud rate: **{ds.get('fraud_rate', 0.0):.4%}**"
                )

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("F1", f"{met.get('f1', 0.0):.4f}")
                col2.metric("Accuracy", f"{met.get('accuracy', 0.0):.4f}")
                col3.metric("Precision", f"{met.get('precision', 0.0):.4f}")
                col4.metric("Recall", f"{met.get('recall', 0.0):.4f}")

                st.markdown("**Confusion Matrix (counts):**")
                st.write(
                    f"TN={cm.get('tn', 0):,} | FP={cm.get('fp', 0):,} | "
                    f"FN={cm.get('fn', 0):,} | TP={cm.get('tp', 0):,}"
                )

                # ---  REALISTIC METRICS DISPLAY ---
                st.markdown("---")
                st.error("🚨 **REALITY CHECK: Training vs. Production Performance**")
                st.caption(
                    "Panel Feedback Address: The 99.6% score reflects the *balanced training baseline*. The *Real-World* score below reflects actual production performance with 0.1% fraud rate.")

                col_real1, col_real2 = st.columns(2)

                with col_real1:
                    st.success("✅ **Theoretical Baseline (Balanced)**")
                    show_eval_card("balanced_val", "Ideal Training Conditions")

                with col_real2:
                    st.warning("⚠️ **Actual Production (Imbalanced)**")
                    show_eval_card("imbalanced_test", "Real-World Scenario (0.1% Fraud)")

                st.markdown("---")
                show_eval_card("ood_test", "C) OOD Shift Test (late-time window)")

            #  chart for quick comparison
            rows = []
            for k, name in [
                ("balanced_val", "Balanced"),
                ("imbalanced_test", "Imbalanced"),
                ("ood_test", "OOD"),
            ]:
                met = evals.get(k, {}).get("metrics", {})
                rows.append({"Dataset": name, "F1": met.get("f1", 0.0), "Precision": met.get("precision", 0.0),
                             "Recall": met.get("recall", 0.0)})

            df_plot = pd.DataFrame(rows).melt(id_vars="Dataset", var_name="Metric", value_name="Value")
            fig = px.bar(df_plot, x="Dataset", y="Value", color="Metric", barmode="group",
                         title="Balanced vs Imbalanced vs OOD (F1 / Precision / Recall)")
            st.plotly_chart(fig, use_container_width=True)

            st.caption("Metrics loaded from results/eval_all.json generated by validate_model_accuracy.py")

    # ------------------------------------
    # 5-AGENT DEMO
    # ------------------------------------
    with subtab_agents:
        st.subheader("5-Agent Hybrid Cloud–Edge System Demo")

        st.markdown(
            "Enter a transaction description in the same format used during training, "
            "for example:\n\n"
            "`CASH_OUT 38427.47 from C1399554611 to C988696172 step:207 oldOrg:38427 newOrg:0`"
        )

        demo_tx = st.text_area(
            "Transaction description",
            "CASH_OUT 38427.47 from C1399554611 to C988696172 "
            "step:207 oldOrg:38427 newOrg:0",
            height=80,
        )

        if st.button("🚀 Run 5-Agent Analysis", key="agent_run"):
            if not demo_tx.strip():
                st.warning("Please enter a transaction description first.")
            else:
                with st.spinner("Running full 5-agent pipeline..."):
                    start_time = time.time()
                    result_demo = agent_system.process_transaction(demo_tx)
                    pipeline_latency_ms = (time.time() - start_time) * 1000.0

                col_d1, col_d2, col_d3 = st.columns(3)
                col_d1.metric("Model Fraud Probability", f"{result_demo['fraud_prob']:.2%}")
                col_d2.metric("Compliance Status", result_demo["status"])
                col_d3.metric("5-Agent latency (ms)", f"{pipeline_latency_ms:.2f}")

                st.markdown("### Final decision (Agent 5)")
                st.info(result_demo["final_message"])

                st.markdown("### Policy Reason (Agent 3)")
                st.write(f"**Reason code:** `{result_demo['reason_code']}`")
                st.write(f"**Reason text:** {result_demo['reason']}")

                with st.expander("Show 5-Agent Trace (for viva)"):
                    st.json(result_demo.get("trace", {}))


                st.markdown("### Parsed transaction data")
                st.json(
                    {
                        "type": result_demo["tx_type"],
                        "amount": result_demo["amount"],
                        "name_orig": result_demo["name_orig"],
                        "name_dest": result_demo["name_dest"],
                        "step": result_demo["step"],
                    }
                )

                st.info(
                    "Detailed logs for Agent 1–4 (data parsing, risk, compliance, advisor) "
                    "are printed in the backend console for audit and reporting."
                )


# ============================================================
# MAIN ROUTER
# ============================================================
if mode == "Customer App":
    render_customer_app()
else:
    render_fraud_console()

st.markdown("---")
st.markdown(
    "*FYP2 by Mouawia · DistilBERT + LoRA adapter · Balanced PaySim dataset · "
    "Hybrid cloud–edge fraud detection & advisory system*"
)