import os
import json
import time
import sys
from openai import OpenAI
from dotenv import load_dotenv
from client import FraudEnvClient
from models import FraudAction

load_dotenv()

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
API_KEY      = GROQ_API_KEY or HF_TOKEN

API_BASE_URL      = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME        = os.environ.get("MODEL_NAME",   "llama-3.1-8b-instant")
ENV_URL           = os.environ.get("ENV_URL",       "http://localhost:8000")
EPISODES_PER_TASK = int(os.environ.get("EPISODES_PER_TASK", "2"))

ENV_NAME = "fraud-investigation"

llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ─────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────
STEP1_SYSTEM = """You are a senior fraud analyst. Analyze the account and transactions for fraud indicators.

CHECK EACH transaction against the account profile:
- Location mismatch: txn location differs from account's home location?
- Amount spike: txn amount much higher than avg_monthly_spend?
- New merchant category: not in usual_merchants list?
- Card not present for large amounts?
- Multiple rapid transactions?
- Foreign country transaction?

If ANY red flags exist → is_fraud=true. Do NOT default to legitimate without checking.

Return ONLY valid JSON with action_type="investigate":
{"action_type":"investigate","is_fraud":true,"fraud_type":"card_fraud|account_takeover|money_mule|bust_out|legitimate","confidence":0.85,"attack_vector":"geo_impossible|velocity|card_not_present|credential_stuffing|sim_swap|credential_compromise|synthetic_identity_network|organized_bust_out|none","evidence":["location_mismatch","high_amount"],"action":"block_card|freeze_account|allow|file_SAR|hold_for_review|escalate","flagged_accounts":[],"hub_account":null,"regulatory_action":"none","reasoning":"2 sentence explanation of red flags found"}"""

STEP2_SYSTEM = """You are a senior fraud analyst making a FINAL decision with full evidence.

FRAUD PATTERNS (memorize):
- card_fraud: geo_impossible travel, velocity spike, CNP high value, foreign txn
- account_takeover: new device + credential changes + immediate transfers
- money_mule: linked accounts with same SSN/device, structured deposits, coordinated drain
- bust_out: slow credit buildup → sudden maxout → disappear
- legitimate: consistent behavior, context explains anomalies (travel booking, support call)

DECISION RULES:
- geo_impossible speed (>1000 km/h between txns) → card_fraud, block_card
- 5+ txns in 10 mins all CNP → velocity fraud, block_card
- New device + password/email/phone change + large wire → account_takeover, freeze_account
- Multiple accounts same SSN/device, structured deposits → money_mule, freeze_account + SAR
- Flight booking found + destination matches txn → legitimate, allow

IMPORTANT: Use ONLY account IDs from the input. Do NOT invent IDs.

Return ONLY valid JSON with action_type="submit_decision":
{"action_type":"submit_decision","is_fraud":true,"fraud_type":"card_fraud|account_takeover|money_mule|bust_out|legitimate","confidence":0.0-1.0,"attack_vector":"geo_impossible|velocity|card_not_present|credential_stuffing|sim_swap|credential_compromise|synthetic_identity_network|organized_bust_out|none","evidence":["signal1","signal2","signal3"],"action":"block_card|freeze_account|allow|file_SAR|hold_for_review|escalate","flagged_accounts":["ACC-XXXX"],"hub_account":null,"regulatory_action":"SAR|law_enforcement|none","reasoning":"detailed explanation of all evidence and decision"}"""

def debug(msg: str):
    print(msg, file=sys.stderr, flush=True)

# ─────────────────────────────────────────
# Prompt builders — pre-flag anomalies
# ─────────────────────────────────────────
def _fmt_txns(txns: list) -> str:
    lines = []
    for t in txns:
        lines.append(
            f"  [{t['timestamp']}] ₹{t['amount']:,.0f} @ {t['merchant']}"
            f" | {t['category']} | {t['location']} | card_present={t['card_present']}"
        )
    return "\n".join(lines) if lines else "  none"


def _flag_anomalies(obs: dict) -> str:
    """Pre-compute obvious anomalies to help the model."""
    acc  = obs["account"]
    txns = obs["transactions"]
    flags = []

    home = acc["location"].lower()
    avg  = acc["avg_monthly_spend"]
    usual_cats = [m.lower() for m in acc["usual_merchants"]]

    for t in txns:
        loc = t["location"].lower()
        amt = t["amount"]

        # Location mismatch
        if loc not in ("online", "atm") and home.split(",")[0].strip() not in loc:
            flags.append(f"LOCATION MISMATCH: txn in '{t['location']}' but account is '{acc['location']}'")

        # Amount spike
        if amt > avg * 1.5:
            flags.append(f"AMOUNT SPIKE: ₹{amt:,.0f} vs avg monthly ₹{avg:,.0f} (x{amt/max(avg,1):.1f})")

        # CNP high value
        if not t["card_present"] and amt > 5000:
            flags.append(f"CARD NOT PRESENT: ₹{amt:,.0f} online at {t['merchant']}")

        # New category
        cat = t["category"].lower()
        if not any(cat in m or m in cat for m in usual_cats):
            flags.append(f"NEW CATEGORY: '{t['category']}' not in usual merchants")

    # Velocity check
    if len(txns) >= 3:
        flags.append(f"VELOCITY: {len(txns)} transactions in this window")

    return "\n".join(f"  ⚠ {f}" for f in flags) if flags else "  No obvious anomalies detected"


def build_step1_prompt(obs: dict) -> str:
    acc = obs["account"]
    anomalies = _flag_anomalies(obs)

    credit_limit = (
        f"₹{acc['credit_limit']:,.0f}"
        if acc.get("credit_limit") is not None
        else "N/A"
    )

    return (
        f"CASE {obs['case_id']} | Task: {obs['task']}\n\n"
        f"ACCOUNT PROFILE:\n"
        f"  ID: {acc['account_id']} | Name: {acc['name']}\n"
        f"  Home: {acc['location']} | Age: {acc['account_age_days']} days\n"
        f"  Avg Monthly Spend: ₹{acc['avg_monthly_spend']:,.0f}\n"
        f"  Usual Merchants: {', '.join(acc['usual_merchants'])}\n"
        f"  Credit Limit: {credit_limit}\n\n"
        f"TRANSACTIONS:\n{_fmt_txns(obs['transactions'])}\n\n"
        f"PRE-ANALYSIS (auto-detected anomalies):\n{anomalies}\n\n"
        "Based on the above, form your initial hypothesis. action_type MUST be 'investigate'.\n"
        "If anomalies exist, set is_fraud=true."
    )

def build_step2_prompt(obs: dict, hyp: FraudAction) -> str:
    acc = obs["account"]
    prompt = (
        f"CASE {obs['case_id']} | Task: {obs['task']}\n\n"
        f"ACCOUNT: {acc['account_id']} | {acc['name']} | Home: {acc['location']}\n"
        f"Avg Monthly: ₹{acc['avg_monthly_spend']:,.0f} | "
        f"Merchants: {', '.join(acc['usual_merchants'])}\n\n"
        f"TRANSACTIONS:\n{_fmt_txns(obs['transactions'])}\n"
    )

    if obs["login_events"]:
        prompt += "\nLOGIN EVENTS:\n"
        for e in obs["login_events"]:
            prompt += (
                f"  [{e['timestamp']}] Device={e['device']} "
                f"IP={e['ip_address']} Loc={e['location']} "
                f"Success={e['success']} | {e.get('note', '')}\n"
            )

    if obs["account_events"]:
        prompt += "\nACCOUNT CHANGES:\n"
        for e in obs["account_events"]:
            prompt += f"  [{e['timestamp']}] {e['event_type']}: {e.get('old_value', '')} → {e['new_value']}\n"

    if obs["linked_accounts"]:
        prompt += "\nLINKED ACCOUNTS (use ONLY these IDs):\n"
        for a in obs["linked_accounts"]:
            prompt += (
                f"  ID={a['account_id']} | {a['name']} | "
                f"SSN4={a.get('ssn_last4', 'N/A')} | Age={a['account_age_days']}d\n"
            )

    if obs["additional_signals"]:
        prompt += "\nSYSTEM SIGNALS:\n"
        for k, v in obs["additional_signals"].items():
            prompt += f"  {k}: {v}\n"

    prompt += (
        f"\nSTEP-1 HYPOTHESIS: fraud={hyp.is_fraud}, type={hyp.fraud_type}, "
        f"vector={hyp.attack_vector}\n"
        f"Initial evidence: {hyp.evidence}\n\n"
        f"Now review ALL signals above and submit your FINAL decision.\n"
        f"action_type MUST be 'submit_decision'.\n"
        f"flagged_accounts: use ONLY IDs shown above (ACC-XXXX or BIZ-XXX format)."
    )
    return prompt


# ─────────────────────────────────────────
# LLM call with rate-limit retry
# ─────────────────────────────────────────
def call_llm(system: str, user: str, max_tokens: int) -> dict:
    for attempt in range(3):
        try:
            resp = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user}
                ],
                temperature=0.1,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                wait = 30 * (attempt + 1)
                debug(f"[DEBUG] rate limit, sleeping {wait}s", flush=True)
                time.sleep(wait)
            else:
                debug(f"[DEBUG] LLM error: {err}", flush=True)
                return _safe_default()
    return _safe_default()


def _safe_default() -> dict:
    return {
        "action_type": "submit_decision",
        "is_fraud": False,
        "fraud_type": "legitimate",
        "confidence": 0.5,
        "evidence": [],
        "attack_vector": "none",
        "action": "allow",
        "flagged_accounts": [],
        "hub_account": None,
        "regulatory_action": "none",
        "reasoning": "Defaulted due to error"
    }


def _make_action(data: dict, action_type: str) -> FraudAction:
    data["action_type"] = action_type
    for field in ("fraud_type","action","attack_vector","regulatory_action"):
        val= str(data.get(field, ""))
        if "|" in val:
            data[field] = val.split("|")[0].strip()
    data.setdefault("is_fraud", False)
    data.setdefault("fraud_type", "legitimate")
    data.setdefault("confidence", 0.5)
    data.setdefault("evidence", [])
    data.setdefault("attack_vector", "none")
    data.setdefault("action", "allow")
    data.setdefault("flagged_accounts", [])
    data.setdefault("hub_account", None)
    data.setdefault("regulatory_action", "none")
    data.setdefault("reasoning", "")
    return FraudAction(**data)


# ─────────────────────────────────────────
# Episode runner — 2-step RL loop
# ─────────────────────────────────────────
def run_episode(env: FraudEnvClient, task: str, episode_num: int) -> float:
    obs = env.reset(task=task)
    print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    step_rewards = []
    error1 = "null"
    error2 = "null"
    success = True

    # ── Step 1: investigate ────────────────
    try:
        data1   = call_llm(STEP1_SYSTEM, build_step1_prompt(obs.model_dump()), max_tokens=350)
        hyp     = _make_action(data1, "investigate")
        obs2    = env.step(hyp)
        hyp_str = f"investigate(hyp={'fraud' if hyp.is_fraud else 'legit'},type={hyp.fraud_type})"
    except Exception as e:
        error1  = str(e)[:60]
        success = False
        hyp     = _make_action(_safe_default(), "investigate")
        hyp_str = "investigate(error)"
        try:
            obs2 = env.step(hyp)
        except Exception:
            step_rewards.append(0.0)
            print(f"[STEP] step=1 action={hyp_str} reward=0.00 done=false error={error1}", flush=True)
            env.close()
            print(f"[END] success=false steps=1 rewards=0.00", flush=True)
            print()
            return 0.0

    step_rewards.append(0.0)
    print(f"[STEP] step=1 action={hyp_str} reward=0.00 done=false error={error1}", flush=True)
    time.sleep(1)

    # ── Step 2: submit_decision ────────────
    try:
        data2   = call_llm(STEP2_SYSTEM, build_step2_prompt(obs2.model_dump(), hyp), max_tokens=512)
        final   = _make_action(data2, "submit_decision")
        result  = env.step(final)
        reward  = result.reward
        act_str = (
            f"submit_decision(fraud={'true' if final.is_fraud else 'false'},"
            f"type={final.fraud_type},"
            f"act={final.action},"
            f"conf={final.confidence:.2f})"
        )
    except Exception as e:
        error2  = str(e)[:60]
        success = False
        reward  = 0.0
        act_str = "submit_decision(error)"

    step_rewards.append(reward)
    print(f"[STEP] step=2 action={act_str} reward={reward:.2f} done=true error={error2}", flush=True)

    env.close()
    rewards_str = ",".join(f"{r:.2f}" for r in step_rewards)
    print(f"[END] success={'true' if success else 'false'} steps=2 rewards={rewards_str}", flush=True)
    print()
    return reward


# ─────────────────────────────────────────
# Task runner
# ─────────────────────────────────────────
def run_task(env: FraudEnvClient, task: str, n_episodes: int) -> dict:
    rewards = []
    print(f"\n{'='*56}")
    debug(f"[DEBUG] TASK: {task.upper()} | episodes={n_episodes}")
    print(f"{'='*56}\n")

    for i in range(n_episodes):
        r = run_episode(env, task, episode_num=i + 1)
        rewards.append(r)
        if i < n_episodes - 1:
            time.sleep(2)

    avg   = sum(rewards) / len(rewards)
    best  = max(rewards)
    worst = min(rewards)
    debug(f"[DEBUG] {task} | avg={avg:.2f} best={best:.2f} worst={worst:.2f}")
    return {"task": task, "episodes": n_episodes, "rewards": rewards,
            "avg_reward": round(avg, 2), "best": best, "worst": worst}


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    print("=" * 56)
    debug("[DEBUG] FRAUD INVESTIGATION — 2-STEP RL INFERENCE")
    debug(f"[DEBUG] model={MODEL_NAME} env={ENV_URL} episodes={EPISODES_PER_TASK}/task")
    print("=" * 56)

    if not API_KEY:
        print("[ERROR] Set HF_TOKEN or GROQ_API_KEY")
        sys.exit(1)

    try:
        env = FraudEnvClient(base_url=ENV_URL)
        debug(f"[DEBUG] Connected to {ENV_URL}\n")
    except ConnectionError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    results = []
    for i, task in enumerate(["task_easy", "task_medium", "task_hard"]):
        result = run_task(env, task, n_episodes=EPISODES_PER_TASK)
        results.append(result)
        if i < 2:
            time.sleep(3)

    print("\n" + "=" * 56)
    debug("[DEBUG] FINAL SUMMARY")
    print("=" * 56)
    all_rewards = []
    for r in results:
        debug(f"[DEBUG] {r['task']:<15} avg={r['avg_reward']:.2f}  best={r['best']:.2f}  worst={r['worst']:.2f}")
        all_rewards.extend(r["rewards"])

    overall = sum(all_rewards) / len(all_rewards)
    debug(f"[DEBUG] overall_avg={overall:.2f}  total_episodes={len(all_rewards)}")
    print("=" * 56)


if __name__ == "__main__":
    main()