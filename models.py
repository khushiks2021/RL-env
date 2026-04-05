from pydantic import BaseModel
from typing import Optional


# ─────────────────────────────────────────
# ACTION — what the LLM agent decides
# ─────────────────────────────────────────
class FraudAction(BaseModel):
    action_type: Optional[str] = "submit_decision"  # "investigate" | "submit_decision"
    is_fraud: bool  # True = fraud, False = legitimate
    fraud_type: str  # "card_fraud" | "account_takeover" | "money_mule" | "bust_out" | "legitimate"
    confidence: float  # 0.0 to 1.0 — how sure is the agent
    evidence: list[str]  # list of signals agent found
    attack_vector: Optional[str]  # "geo_impossible" | "velocity" | "credential_stuffing" | "sim_swap" | "none"
    action: str  # "block_card" | "freeze_account" | "allow" | "file_SAR" | "hold_for_review" | "escalate"
    flagged_accounts: list[str]  # for task 3 — which accounts are in the network
    hub_account: Optional[str]  # for task 3 — the central account money flows to
    regulatory_action: str  # "SAR" | "law_enforcement" | "none"
    reasoning: str  # step-by-step explanation from LLM


# ─────────────────────────────────────────
# OBSERVATION — what the agent sees
# ─────────────────────────────────────────
class Transaction(BaseModel):
      txn_id: str
      amount: float
      merchant: str
      category: str
      location: str
      timestamp: str
      card_present: bool


class LoginEvent(BaseModel):
    timestamp: str
    device: str
    ip_address: str
    location: str
    success: bool
    note: Optional[str]


class AccountEvent(BaseModel):
    timestamp: str
    event_type: str  # "password_change" | "email_change" | "phone_change" | "address_change"
    old_value: Optional[str]
    new_value: str


class AccountProfile(BaseModel):
    account_id: str
    name: str
    email: str
    location: str
    account_age_days: int
    avg_monthly_spend: float
    usual_merchants: list[str]
    credit_limit: Optional[float]
    ssn_last4: Optional[str]


class FraudObservation(BaseModel):
    # Core fields
    case_id: str
    task: str                         # "task_easy" | "task_medium" | "task_hard"
    step: int                         # which step in the episode

    # Case data
    account: AccountProfile
    transactions: list[Transaction]
    login_events: list[LoginEvent]
    account_events: list[AccountEvent]
    linked_accounts: list[AccountProfile]   # for task 3
    additional_signals: dict                # extra flags from system

    # RL fields
    reward: float                     # reward from last action (0.0 if first step)
    done: bool                        # is episode over
    feedback: str                     # what was right/wrong (shown after action)


# ─────────────────────────────────────────
# STATE — internal environment state
# ─────────────────────────────────────────
class FraudState(BaseModel):
    episode_id: str
    task: str
    case_id: str
    step_count: int
    total_reward: float
    is_complete: bool
