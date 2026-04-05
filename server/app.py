# server/app.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models import FraudAction, FraudObservation, FraudState
from server.environment import FraudEnvironment

# ─────────────────────────────────────────
# App setup
# ─────────────────────────────────────────
app = FastAPI(
    title="Fraud Investigation Environment",
    description="RL environment where LLM agents learn to detect financial fraud",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# One environment instance per server
# (for multi-session support this becomes a dict of session_id → env)
env = FraudEnvironment()


# ─────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────
class ResetRequest(BaseModel):
    task: str = "task_easy"   # default to easy


class StepRequest(BaseModel):
    action: FraudAction


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────

@app.get("/health")
def health():
    """Judges will call this to verify your server is alive."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/reset", response_model=FraudObservation)
def reset(request: ResetRequest):
    """Start a new fraud investigation episode."""
    try:
        obs = env.reset(task=request.task)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=FraudObservation)
def step(request: StepRequest):
    """Submit fraud analysis decision. Returns reward + feedback."""
    try:
        obs = env.step(request.action)
        return obs
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=FraudState)
def state():
    """Get current internal state of the environment."""
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
def list_tasks():
    """List all available tasks with descriptions."""
    return {
        "tasks": [
            {
                "id": "task_easy",
                "name": "Credit Card Fraud Detection",
                "difficulty": "easy",
                "description": "Detect fraud from a single transaction + account history",
                "signals": ["geo_impossible", "velocity", "card_not_present", "legitimate"]
            },
            {
                "id": "task_medium",
                "name": "Account Takeover Detection",
                "difficulty": "medium",
                "description": "Detect account compromise from login events + behavioral signals",
                "signals": ["device_takeover", "credential_stuffing", "sim_swap", "legitimate"]
            },
            {
                "id": "task_hard",
                "name": "Fraud Network Investigation",
                "difficulty": "hard",
                "description": "Detect coordinated fraud across multiple linked accounts",
                "signals": ["money_mule", "bust_out", "legitimate_business"]
            }
        ]
    }

@app.post("/close")
def close():
    try:
        env.close()
        return {"status": "closed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )


if __name__ == "__main__":
    main()
