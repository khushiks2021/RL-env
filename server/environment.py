# server/environment.py
import uuid
import sys
import os

# Make sure models can be imported from parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FraudAction, FraudObservation, FraudState
from server.case_generator import get_case
from server.grader import grade


class FraudEnvironment:

    def __init__(self):
        # Episode state — one environment instance per session
        self.current_case = None   # full case dict {observation, truth}
        self.current_task = None   # which task is running
        self.episode_id = None
        self.step_count = 0
        self.total_reward = 0.0
        self.is_done = False

    # ─────────────────────────────────────────
    # RESET — start a new episode
    # ─────────────────────────────────────────
    def reset(self, task: str = "task_easy") -> FraudObservation:
        """
        Called at the start of every episode.
        Picks a new fraud case and returns the first observation.
        """
        # Validate task name
        valid_tasks = ["task_easy", "task_medium", "task_hard"]
        if task not in valid_tasks:
            raise ValueError(f"Invalid task '{task}'. Choose from {valid_tasks}")

        # Generate a fresh case
        self.current_case = get_case(task)
        self.current_task = task
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.total_reward = 0.0
        self.is_done = False

        # Return the first observation (no reward yet, agent hasn't acted)
        obs = self.current_case["observation"]
        obs.step = 0
        obs.reward = 0.0
        obs.done = False
        obs.feedback = "New case loaded. Analyze the evidence and submit your decision."

        return obs

    # ─────────────────────────────────────────
    # STEP — agent submits a decision
    # ─────────────────────────────────────────
    def step(self, action: FraudAction) -> FraudObservation:
        """
        Agent submits their fraud analysis decision.
        Environment grades it and returns reward + feedback.
        This is a single-step environment (one decision per case).
        """
        if self.is_done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        if self.current_case is None:
            raise RuntimeError("No active episode. Call reset() first.")

        # Grade the action
        truth = self.current_case["truth"]
        reward, feedback = grade(action, truth, self.current_task)

        # Update internal state
        self.step_count += 1
        self.total_reward += reward
        self.is_done = True   # one decision per case (can extend to multi-step)

        # Build response observation
        obs = self.current_case["observation"]
        obs.step = self.step_count
        obs.reward = reward
        obs.done = self.is_done
        obs.feedback = feedback

        return obs

    # ─────────────────────────────────────────
    # STATE — internal state snapshot
    # ─────────────────────────────────────────
    def state(self) -> FraudState:
        """Returns current internal state of the environment."""
        return FraudState(
            episode_id=self.episode_id or "no_episode",
            task=self.current_task or "none",
            case_id=self.current_case["observation"].case_id if self.current_case else "none",
            step_count=self.step_count,
            total_reward=self.total_reward,
            is_complete=self.is_done
        )