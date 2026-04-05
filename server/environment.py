# server/environment.py
import uuid
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FraudAction, FraudObservation, FraudState
from server.case_generator import get_case
from server.grader import grade

MAX_STEPS = 2  # step 1: investigate, step 2: submit_decision


class FraudEnvironment:

    def __init__(self):
        self.current_case = None
        self._full_obs = None   # stores full observation for step 2 reveal
        self.current_task = None
        self.episode_id = None
        self.step_count = 0
        self.total_reward = 0.0
        self.is_done = False

    # ─────────────────────────────────────────
    # RESET — start a new episode
    # ─────────────────────────────────────────
    def reset(self, task: str = "task_easy") -> FraudObservation:
        valid_tasks = ["task_easy", "task_medium", "task_hard"]
        if task not in valid_tasks:
            raise ValueError(f"Invalid task '{task}'. Choose from {valid_tasks}")

        self.current_case = get_case(task)
        self.current_task = task
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.total_reward = 0.0
        self.is_done = False

        full = self.current_case["observation"]
        self._full_obs = full

        # Return PARTIAL observation — account + transactions only
        # Login events, account events, linked accounts, and signals are hidden
        # Agent must call action_type="investigate" to reveal them
        return FraudObservation(
            case_id=full.case_id,
            task=full.task,
            step=0,
            account=full.account,
            transactions=full.transactions,
            login_events=[],
            account_events=[],
            linked_accounts=[],
            additional_signals={},
            reward=0.0,
            done=False,
            feedback=(
                "Step 1 of 2: You see account profile and transactions. "
                "Call action_type='investigate' to reveal login events, "
                "account changes, linked accounts and system signals. "
                "Then submit your final decision."
            )
        )

    # ─────────────────────────────────────────
    # STEP — agent acts
    # ─────────────────────────────────────────
    def step(self, action: FraudAction) -> FraudObservation:
        if self.is_done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        if self.current_case is None:
            raise RuntimeError("No active episode. Call reset() first.")

        self.step_count += 1
        full = self._full_obs

        # Step 1: investigate → reveal full data (no grading yet)
        if action.action_type == "investigate" and self.step_count == 1:
            return FraudObservation(
                case_id=full.case_id,
                task=full.task,
                step=self.step_count,
                account=full.account,
                transactions=full.transactions,
                login_events=full.login_events,
                account_events=full.account_events,
                linked_accounts=full.linked_accounts,
                additional_signals=full.additional_signals,
                reward=0.0,
                done=False,
                feedback=(
                    "Step 2 of 2: Full data revealed — login events, account changes, "
                    "linked accounts and system signals are now visible. "
                    "Submit your final decision with action_type='submit_decision'."
                )
            )

        # Final step: grade the decision
        truth = self.current_case["truth"]
        reward, feedback = grade(action, truth, self.current_task)

        self.total_reward += reward
        self.is_done = True

        return FraudObservation(
            case_id=full.case_id,
            task=full.task,
            step=self.step_count,
            account=full.account,
            transactions=full.transactions,
            login_events=full.login_events,
            account_events=full.account_events,
            linked_accounts=full.linked_accounts,
            additional_signals=full.additional_signals,
            reward=reward,
            done=True,
            feedback=feedback
        )

    # ─────────────────────────────────────────
    # STATE / CLOSE
    # ─────────────────────────────────────────
    def state(self) -> FraudState:
        return FraudState(
            episode_id=self.episode_id or "no_episode",
            task=self.current_task or "none",
            case_id=self._full_obs.case_id if self._full_obs else "none",
            step_count=self.step_count,
            total_reward=self.total_reward,
            is_complete=self.is_done
        )

    def close(self):
        self.current_case = None
        self._full_obs = None
        self.is_done = True
