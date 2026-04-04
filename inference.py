import os
import json
import sys
from openai import OpenAI
from dotenv import load_dotenv
from client import FraudEnvClient
from models import FraudAction

load_dotenv()

# ─────────────────────────────────────────
# Config from environment variables (mandatory)
# ─────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "llama-3.3-70b-versatile")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
ENV_URL      = os.environ.get("ENV_URL",       "http://localhost:8000")
EPISODES_PER_TASK = int(os.environ.get("EPISODES_PER_TASK", "3"))

# ─────────────────────────────────────────
# LLM client (OpenAI-compatible)
# ─────────────────────────────────────────
llm = OpenAI(
    base_url=API_BASE_URL,
    api_key=GROQ_API_KEY
)

# ─────────────────────────────────────────
# System prompt — teaches LLM how to think
# ─────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert financial fraud analyst with 10+ years of experience.

You will receive a fraud case with:

* Account profile
* Transactions
* Login events
* Account changes
* Linked accounts (if any)
* System signals

Analyze all evidence and return ONLY this JSON:

{
"is_fraud": true/false,
"fraud_type": "card_fraud" | "account_takeover" | "money_mule" | "bust_out" | "legitimate",
"confidence": 0.0-1.0,
"evidence": ["signal1", "signal2"],
"attack_vector": "geo_impossible" | "velocity" | "card_not_present" | "credential_stuffing" | "sim_swap" | "credential_compromise" | "synthetic_identity_network" | "organized_bust_out" | "none",
"action": "block_card" | "freeze_account" | "allow" | "file_SAR" | "hold_for_review" | "escalate",
"flagged_accounts": ["ACC-XXXX"],
"hub_account": "EXT-XXXX" or null,
"regulatory_action": "SAR" | "law_enforcement" | "none",
"reasoning": "Detailed explanation (min 50 words)"
}

Rules:

* Output ONLY valid JSON (no extra text)
* Avoid false positives; consider legitimate explanations
* Evidence = specific signals (e.g., new_device, geo_impossible, velocity_spike)

Fraud patterns:

* Card fraud → unusual spending, location jumps, odd timing
* Account takeover → new device/IP, credential changes
* Network fraud → shared data, coordinated activity

flagged_accounts rules:

* Use ONLY IDs from input (ACC-XXXX / BIZ-XXX)
* Do NOT invent IDs
* Bust-out → only main account
* Money mule → main + linked accounts
  """



def build_case_prompt(obs: dict) -> str:
    """Convert observation dict into a readable prompt for the LLM."""
    prompt = f"""
=== FRAUD INVESTIGATION CASE ===
Case ID: {obs['case_id']}
Task: {obs['task']}

--- ACCOUNT PROFILE ---
Account ID: {obs['account']['account_id']}       
Name: {obs['account']['name']}
Location: {obs['account']['location']}
Account Age: {obs['account']['account_age_days']} days
Avg Monthly Spend: ₹{obs['account']['avg_monthly_spend']:,.0f}
Usual Merchants: {', '.join(obs['account']['usual_merchants'])}
Credit Limit: {f"₹{obs['account']['credit_limit']:,.0f}" if obs['account']['credit_limit'] is not None else 'N/A'}

--- TRANSACTIONS ---"""

    for txn in obs['transactions']:
        prompt += f"""
  [{txn['timestamp']}] ₹{txn['amount']:,.0f} at {txn['merchant']}
  Category: {txn['category']} | Location: {txn['location']} | Card Present: {txn['card_present']}"""

    if obs['login_events']:
        prompt += "\n\n--- LOGIN EVENTS ---"
        for event in obs['login_events']:
            prompt += f"""
  [{event['timestamp']}] Device: {event['device']}
  IP: {event['ip_address']} | Location: {event['location']}
  Success: {event['success']} | Note: {event.get('note', 'N/A')}"""

    if obs['account_events']:
        prompt += "\n\n--- ACCOUNT CHANGES ---"
        for event in obs['account_events']:
            prompt += f"""
  [{event['timestamp']}] {event['event_type']}
  Old: {event.get('old_value', 'N/A')} → New: {event['new_value']}"""

    if obs['linked_accounts']:
        prompt += "\n\n--- LINKED ACCOUNTS ---"
        for acc in obs['linked_accounts']:
            prompt += f"""
  Account ID: {acc['account_id']}
  Name: {acc['name']} | SSN last4: {acc.get('ssn_last4', 'N/A')}
  Age: {acc['account_age_days']} days | Avg Spend: ₹{acc['avg_monthly_spend']:,.0f}"""

    prompt += "\n\n--- SYSTEM SIGNALS ---"
    for key, value in obs['additional_signals'].items():
        prompt += f"\n  {key}: {value}"

    prompt += "\n\n=== Analyze the above case and return your decision as JSON ==="
    return prompt


def get_llm_decision(obs_dict: dict) -> FraudAction:
    """Call LLM and parse response into FraudAction."""
    prompt = build_case_prompt(obs_dict)

    response = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.1,     # low temp = more consistent decisions
        max_tokens=1024,
        response_format={"type": "json_object"}  # force JSON output
    )

    raw = response.choices[0].message.content

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        print(f"[WARN] JSON parse failed, using safe default")
        data = {
            "is_fraud": False,
            "fraud_type": "legitimate",
            "confidence": 0.5,
            "evidence": [],
            "attack_vector": "none",
            "action": "allow",
            "flagged_accounts": [],
            "hub_account": None,
            "regulatory_action": "none",
            "reasoning": "Failed to parse LLM response"
        }

    return FraudAction(**data)


def run_episode(env: FraudEnvClient, task: str, episode_num: int) -> float:
    """Run one full episode. Returns total reward."""

    # Reset environment
    obs = env.reset(task=task)

    print(f"[START] task={task} episode={episode_num} case={obs.case_id}")
    print(f"[START] account={obs.account.name} txns={len(obs.transactions)}")

    # Get LLM decision
    obs_dict = obs.model_dump()
    action = get_llm_decision(obs_dict)

    print(f"[STEP]  is_fraud={action.is_fraud} type={action.fraud_type} confidence={action.confidence:.2f}")
    print(f"[STEP]  action={action.action} regulatory={action.regulatory_action}")
    print(f"[STEP]  evidence_count={len(action.evidence)}")
    if action.flagged_accounts:
        print(f"[STEP]  flagged_accounts={action.flagged_accounts}")

    # Submit to environment
    result = env.step(action)

    print(f"[STEP]  reward={result.reward}")
    print(f"[STEP]  feedback_preview={result.feedback[:120]}...")
    print(f"[END]   task={task} episode={episode_num} reward={result.reward} done={result.done}")
    print()

    return result.reward


def run_task(env: FraudEnvClient, task: str, n_episodes: int) -> dict:
    """Run multiple episodes for one task. Returns summary."""
    rewards = []

    print(f"\n{'='*60}")
    print(f"TASK: {task.upper()}")
    print(f"Episodes: {n_episodes}")
    print(f"{'='*60}\n")

    for i in range(n_episodes):
        reward = run_episode(env, task, episode_num=i+1)
        rewards.append(reward)

    avg  = sum(rewards) / len(rewards)
    best = max(rewards)
    worst = min(rewards)

    print(f"\n--- {task} SUMMARY ---")
    print(f"Episodes: {n_episodes}")
    print(f"Avg Reward:  {avg:.2f}")
    print(f"Best:        {best:.2f}")
    print(f"Worst:       {worst:.2f}")
    print(f"All rewards: {rewards}")

    return {
        "task": task,
        "episodes": n_episodes,
        "rewards": rewards,
        "avg_reward": round(avg, 2),
        "best": best,
        "worst": worst
    }


def main():
    print("="*60)
    print("FRAUD INVESTIGATION ENVIRONMENT — INFERENCE")
    print(f"Model:    {MODEL_NAME}")
    print(f"Env URL:  {ENV_URL}")
    print(f"Episodes: {EPISODES_PER_TASK} per task")
    print("="*60)

    # Connect to environment
    try:
        env = FraudEnvClient(base_url=ENV_URL)
        print(f"\n[OK] Connected to environment at {ENV_URL}\n")
    except ConnectionError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Run all 3 tasks
    results = []
    for task in ["task_easy", "task_medium", "task_hard"]:
        result = run_task(env, task, n_episodes=EPISODES_PER_TASK)
        results.append(result)

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    overall_rewards = []
    for r in results:
        print(f"{r['task']:<15} avg={r['avg_reward']:.2f}  best={r['best']:.2f}  worst={r['worst']:.2f}")
        overall_rewards.extend(r["rewards"])

    overall_avg = sum(overall_rewards) / len(overall_rewards)
    print(f"\nOverall avg reward: {overall_avg:.2f}")
    print(f"Total episodes run: {len(overall_rewards)}")
    print("="*60)


if __name__ == "__main__":
    main()
