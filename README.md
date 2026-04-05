---
title: Fraud Investigation Env
emoji: 🕵️
colorFrom: red
colorTo: orange
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - fraud-detection
---
# Fraud Investigation RL Environment

An OpenEnv-compatible reinforcement learning environment where LLM agents learn to detect financial fraud across three progressively harder tasks.

---

## Motivation

Financial fraud costs banks billions annually. This environment simulates real investigator workflows: gather evidence, form hypotheses, make decisions. Unlike toy environments, every case mirrors real-world fraud patterns used by fraud ops teams.

---

## Tasks

| Task          | Difficulty | Description                                                                           |
| ------------- | ---------- | ------------------------------------------------------------------------------------- |
| `task_easy`   | Easy       | Credit card fraud — geo-impossible transactions, velocity fraud, legitimate travelers |
| `task_medium` | Medium     | Account takeover — device hijack, credential stuffing, SIM swap                       |
| `task_hard`   | Hard       | Network fraud — money mule rings, bust-out fraud, legitimate businesses               |

---

## Episode Design (2-step)

1. **Step 1 — Investigate**
   Agent sees account profile + transactions only. Calls `investigate` to reveal login events, account changes, and system signals.

2. **Step 2 — Decide**
   Agent sees full data and submits final fraud decision.

---

## Action Space

```json
{
  "action_type": "investigate | submit_decision",
  "is_fraud": true,
  "fraud_type": "card_fraud | account_takeover | money_mule | bust_out | legitimate",
  "confidence": 0.91,
  "evidence": ["geo_impossible", "card_not_present"],
  "attack_vector": "geo_impossible",
  "action": "block_card | freeze_account | allow | file_SAR | hold_for_review | escalate",
  "flagged_accounts": ["ACC-1001"],
  "hub_account": null,
  "regulatory_action": "SAR | law_enforcement | none",
  "reasoning": "Transaction from Lagos 224 mins after Mumbai swipe — physically impossible."
}
```

---

## Observation Space

```json
{
  "case_id": "CC-001",
  "task": "task_easy",
  "step": 1,
  "account": { "account_id": "...", "name": "...", ... },
  "transactions": [...],
  "login_events": [...],
  "account_events": [...],
  "linked_accounts": [...],
  "additional_signals": { "distance_km": 8200, ... },
  "reward": 0.0,
  "done": false,
  "feedback": "..."
}
```

---

## Reward Design

| Signal                   | Weight | Notes                                 |
| ------------------------ | ------ | ------------------------------------- |
| Fraud/legitimate correct | +0.30  | Core decision                         |
| Fraud type correct       | +0.15  | card_fraud vs account_takeover etc    |
| Evidence signals matched | +0.25  | Fuzzy match, partial credit           |
| Correct action taken     | +0.20  | block_card / freeze / allow           |
| Confidence calibrated    | +0.10  | Penalizes overconfident wrong answers |
| False positive           | -0.10  | Flagging legitimate as fraud          |

---

## Baseline Scores

| Task        | Avg Reward |
| ----------- | ---------- |
| task_easy   | ~0.78      |
| task_medium | ~0.73      |
| task_hard   | ~0.71      |

---

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/khushiks2021/RL-env
cd RL-env
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

---

### 2. Start the environment server

```bash
docker build -t fraud-env .
docker run -p 8000:8000 fraud-env
```

Or with docker-compose:

```bash
docker-compose up -d
```

---

### 3. Run inference

```bash
pip install -r requirements.txt
python inference.py
```

---

## Environment Variables

| Variable          | Required      | Description                                                                      |
| ----------------- | ------------- | -------------------------------------------------------------------------------- |
| HF_TOKEN          | Yes (or GROQ) | HuggingFace API token                                                            |
| GROQ_API_KEY      | Yes (or HF)   | Groq API key (free tier works)                                                   |
| API_BASE_URL      | No            | LLM endpoint (default: Groq)                                                     |
| MODEL_NAME        | No            | Model name (default: llama-3.1-8b-instant)                                       |
| ENV_URL           | No            | Environment server URL (default: [http://localhost:8000](http://localhost:8000)) |
| EPISODES_PER_TASK | No            | Episodes per task (default: 2)                                                   |

---

## API Endpoints

| Method | Path    | Description           |
| ------ | ------- | --------------------- |
| GET    | /health | Health check          |
| POST   | /reset  | Start new episode     |
| POST   | /step   | Submit action         |
| GET    | /state  | Current episode state |
| GET    | /tasks  | List all tasks        |
| POST   | /close  | End episode           |

---

