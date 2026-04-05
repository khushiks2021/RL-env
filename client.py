import requests
from models import FraudAction, FraudObservation, FraudState


class FraudEnvClient:
    """
    HTTP client for the Fraud Investigation Environment.
    Your training code only ever touches this class.
    All HTTP is handled internally.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._verify_connection()

    def _verify_connection(self):
        """Check server is alive on startup."""
        try:
            res = requests.get(f"{self.base_url}/health", timeout=5)
            res.raise_for_status()
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to environment at {self.base_url}\n"
                f"Is the server running? Error: {e}"
            )

    def reset(self, task: str = "task_easy") -> FraudObservation:
        """Start a new episode. Returns first observation."""
        res = requests.post(
            f"{self.base_url}/reset",
            json={"task": task},
            timeout=30
        )
        res.raise_for_status()
        return FraudObservation(**res.json())

    def step(self, action: FraudAction) -> FraudObservation:
        """Submit decision. Returns reward + feedback."""
        res = requests.post(
            f"{self.base_url}/step",
            json={"action": action.model_dump()},
            timeout=30
        )
        res.raise_for_status()
        return FraudObservation(**res.json())

    def state(self) -> FraudState:
        """Get current internal state."""
        res = requests.get(
            f"{self.base_url}/state",
            timeout=10
        )
        res.raise_for_status()
        return FraudState(**res.json())

    def list_tasks(self) -> list[dict]:
        """List all available tasks."""
        res = requests.get(
            f"{self.base_url}/tasks",
            timeout=10
        )
        res.raise_for_status()
        return res.json()["tasks"]

    def close(self):
        """Best-effort close. Never raises."""
        try:
            requests.post(
                f"{self.base_url}/close",
                timeout=10
            )
        except Exception:
            pass
