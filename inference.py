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
API_KEY      = HF_TOKEN or GROQ_API_KEY          # HF_TOKEN takes priority

API_BASE_URL      = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME        = os.environ.get("MODEL_NAME",   "llama-3.1-8b-instant")
ENV_URL           = os.environ.get("ENV_URL",       "http://localhost:8000")
EPISODES_PER_TASK = int(os.environ.get("EPISODES_PER_TASK", "2"))

ENV_NAME = "fraud-investigation"

llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ─────────────────────────────────────────
# System prompts — compact to save tokens
# ─────────────────────────────────────────
STEP1_SYSTEM = """You are a fraud analyst. Review account + transactions and form an initial hypothesis.
Return ONLY this JSON:
{"action_type":"investigate","is_fraud":true/false,"fraud_type":"card_fraud|account_takeover|money_mule|bust_out|legitimate","confidence":0.0-1.0,"attack_vector":"geo_impossible|velocity|card_not_present|credential_stuffing|sim_swap|credential_compromise|synthetic_identity_network|organized_bust_out|none","evidence":["signal1","signal2"],"action":"block_card|freeze_account|allow|file_SAR|hold_for_review|escalate","flagged_accounts":[],"hub_account":null,"regulatory_action":"SAR|law_enforcement|none","reasoning":"initial hypothesis in 1-2 sentences"}
Output ONLY valid JSON."""

STEP2_SYSTEM = """You are a fraud analyst. Review ALL evidence and submit your FINAL decision.
Return ONLY this JSON:
{"action_type":"submit_decision","is_fraud":true/false,"fraud_type":"card_fraud|account_takeover|money_mule|bust_out|legitimate","confidence":0.0-1.0,"attack_vector":"geo_impossible|velocity|card_not_present|credential_stuffing|sim_swap|credential_compromise|synthetic_identity_network|organized_bust_out|none","evidence":["signal1","signal2","signal3"],"action":"block_card|freeze_account|allow|file_SAR|hold_for_review|escalate","flagged_accounts":["ACC-XXXX"],"hub_account":"EXT-XXXX or null","regulatory_action":"SAR|law_enforcement|none","reasoning":"detailed explanation min 40 words"}
Rules: Use ONLY account IDs from input. Output ONLY valid JSON."""


# ─────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────
def _fmt_txns(txns: list) -> str:
    lines = []
    for t in txns:
        lines.append(
            f"  {t['timestamp']} ₹{t['amount']:,.0f} @ {t['merchant']} "
            f"[{t['category']}] {t['location']} card_present={t['card_present']}"
        )
    return "\n".join(lines) if lines else "  none"


def build_step1_prompt(obs: dict) -> str:
    acc = obs["account"]
    prompt = (
        f"CASE {obs['case_id']} | Task: {obs['task']}\n"
        f"Account: {acc['account_id']} | {acc['name']} | {acc['location']} | "
        f"age={acc['account_age_days']}d | avg_spend=₹{acc['avg_monthly_spend']:,.0f} | "
        f"limit={f\"₹{acc['credit_limit']:,.0f}\" if acc['credit_limit'] else 'N/A'}\n"
        f"Merchants: {', '.join(acc['usual_merchants'])}\n\n"
        f"TRANSACTIONS:\n{_fmt_txns(obs['transactions'])}\n\n"
        "Form your initial hypothesis. action_type must be 'investigate'."
    )
    return prompt


def build_step2_prompt(obs: dict, hypothesis: FraudAction) -> str:
    acc = obs["account"]
    prompt = (
        f"CASE {obs['case_id']} | Task: {obs['task']}\n"
        f"Account: {acc['account_id']} | {acc['name']} | {acc['location']} | "
        f"age={acc['account_age_days']}d | avg_spend=₹{acc['avg_monthly_spend']:,.0f}\n\n"
        f"TRANSACTIONS:\n{_fmt_txns(obs['transactions'])}\n"
    )

    if obs["login_events"]:
        prompt += "\nLOGIN EVENTS:\n"
        for e in obs["login_events"]:
            prompt += (
                f"  {e['timestamp']} device={e['device']} ip={e['ip_address']} "
                f"loc={e['location']} success={e['success']} note={e.get('note','')}\n"
            )

    if obs["account_events"]:
        prompt += "\nACCOUNT CHANGES:\n"
        for e in obs["account_events"]:
            prompt += f"  {e['timestamp']} {e['event_type']}: {e.get('old_value','')} → {e['new_value']}\n"

    if obs["linked_accounts"]:
        prompt += "\nLINKED ACCOUNTS (USE THESE IDs ONLY):\n"
        for a in obs["linked_accounts"]:
            prompt += (
                f"  ID={a['account_id']} name={a['name']} "
                f"ssn4={a.get('ssn_last4','?')} age={a['account_age_days']}d\n"
            )

    if obs["additional_signals"]:
        prompt += "\nSYSTEM SIGNALS:\n"
        for k, v in obs["additional_signals"].items():
            prompt += f"  {k}: {v}\n"

    prompt += (
        f"\nYour step-1 hypothesis: fraud={hypothesis.is_fraud}, "
        f"type={hypothesis.fraud_type}, vector={hypothesis.attack_vector}\n\n"
        "Now submit your FINAL decision. action_type must be 'submit_decision'."
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
                print(f"[DEBUG] rate limit hit, sleeping {wait}s")
                time.sleep(wait)
            else:
                print(f"[DEBUG] LLM error: {err}")
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
    # Ensure required fields have fallback values
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
      print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}")

      step_rewards = []
      error1 = "null"
      error2 = "null"
      success = True

      # Step 1: investigate
      try:
          prompt1 = build_step1_prompt(obs.model_dump())
          data1   = call_llm(STEP1_SYSTEM, prompt1, max_tokens=300)
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
              print(f"[STEP] step=1 action={hyp_str} reward=0.00 done=false error={error1}")
              env.close()
              rewards_str = ",".join(f"{r:.2f}" for r in step_rewards)
              print(f"[END] success=false steps=1 rewards={rewards_str}")
              return 0.0

      step_rewards.append(0.0)
      print(f"[STEP] step=1 action={hyp_str} reward=0.00 done=false error={error1}")
      time.sleep(1)

      # Step 2: submit_decision
      try:
          prompt2 = build_step2_prompt(obs2.model_dump(), hyp)
          data2   = call_llm(STEP2_SYSTEM, prompt2, max_tokens=512)
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
      print(f"[STEP] step=2 action={act_str} reward={reward:.2f} done=true error={error2}")

      env.close()
      rewards_str = ",".join(f"{r:.2f}" for r in step_rewards)
      print(f"[END] success={'true' if success else 'false'} steps=2 rewards={rewards_str}")
      print()
      return reward



# ─────────────────────────────────────────
# Task runner
# ─────────────────────────────────────────
def run_task(env: FraudEnvClient, task: str, n_episodes: int) -> dict:
    rewards = []

    print(f"\n{'='*56}")
    print(f"[DEBUG] TASK: {task.upper()} | episodes={n_episodes}")
    print(f"{'='*56}\n")

    for i in range(n_episodes):
        r = run_episode(env, task, episode_num=i + 1)
        rewards.append(r)
        if i < n_episodes - 1:
            time.sleep(2)  # avoid rate limit between episodes

    avg   = sum(rewards) / len(rewards)
    best  = max(rewards)
    worst = min(rewards)

    print(f"[DEBUG] {task} | avg={avg:.2f} best={best:.2f} worst={worst:.2f} rewards={rewards}")

    return {"task": task, "episodes": n_episodes, "rewards": rewards,
            "avg_reward": round(avg, 2), "best": best, "worst": worst}


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    print("=" * 56)
    print("[DEBUG] FRAUD INVESTIGATION — 2-STEP RL INFERENCE")
    print(f"[DEBUG] model={MODEL_NAME} env={ENV_URL} episodes={EPISODES_PER_TASK}/task")
    print("=" * 56)

    if not API_KEY:
        print("[ERROR] Set HF_TOKEN or GROQ_API_KEY environment variable")
        sys.exit(1)

    try:
        env = FraudEnvClient(base_url=ENV_URL)
        print(f"[DEBUG] Connected to {ENV_URL}\n")
    except ConnectionError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    results = []
    for i, task in enumerate(["task_easy", "task_medium", "task_hard"]):
        result = run_task(env, task, n_episodes=EPISODES_PER_TASK)
        results.append(result)
        if i < 2:
            time.sleep(3)  # pause between tasks

    print("\n" + "=" * 56)
    print("[DEBUG] FINAL SUMMARY")
    print("=" * 56)
    all_rewards = []
    for r in results:
        print(f"[DEBUG] {r['task']:<15} avg={r['avg_reward']:.2f}  best={r['best']:.2f}  worst={r['worst']:.2f}")
        all_rewards.extend(r["rewards"])

    overall = sum(all_rewards) / len(all_rewards)
    print(f"[DEBUG] overall_avg={overall:.2f}  total_episodes={len(all_rewards)}")
    print("=" * 56)


if __name__ == "__main__":
    main()
