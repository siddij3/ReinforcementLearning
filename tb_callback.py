from collections import deque
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback


class FraudTensorboardCallback(BaseCallback):
    """
    Tracks fraud-detection outcomes in a rolling deque and writes
    precision / recall / F1 / action-rate metrics to TensorBoard.

    Parameters
    ----------
    window : int
        Number of *episode-terminal* steps to keep in the rolling window.
        Each terminal step contributes one outcome ("tp", "fp", "tn", "fn").
    verbose : int
        0 = silent; 1 = print a summary line each time metrics are flushed.
    """

    def __init__(self, window: int = 100, verbose: int = 1):
        # Must call super().__init__(verbose) — SB3 uses verbose internally.
        super().__init__(verbose)
        self.window = window

        # Rolling window of terminal outcomes across ALL parallel envs.
        # Each entry is one of: "tp", "fp", "tn", "fn"
        self._outcomes: deque = deque(maxlen=window)

        # Rolling window of stages at which episodes ended.
        self._stages: deque = deque(maxlen=window)

        # Counters reset each rollout (for per-rollout action-rate logging)
        self._rollout_actions = {"probe": 0, "flag": 0, "pass": 0}
        self._rollout_steps   = 0

    # ── Lifecycle hooks ───────────────────────────────────────────────────────

    def _on_training_start(self) -> None:
        """Called once before the first rollout. Used to log the graph or
        hyperparameters — left minimal here."""
        if self.verbose:
            print(f"[FraudCB] Training start — rolling window = {self.window}")

    def _on_rollout_start(self) -> None:
        """Reset per-rollout action counters so we can log action rates per
        rollout rather than globally."""
        self._rollout_actions = {"probe": 0, "flag": 0, "pass": 0}
        self._rollout_steps   = 0

    def _on_step(self) -> bool:
        """
        Called by SB3 after EVERY environment step (across all n_envs in
        parallel).  `self.locals` contains:
            "actions"  : np.ndarray shape (n_envs,) — the actions just taken
            "infos"    : list[dict]                  — one dict per env

        We inspect every info dict; only terminal steps carry an outcome.
        Return True to continue training; False would abort training early.
        """
        actions = self.locals["actions"]   # shape (n_envs,)
        infos   = self.locals["infos"]     # list of dicts, one per env

        for action, info in zip(actions, infos):
            # --- Track action distribution for this rollout ----------------
            self._rollout_steps += 1
            if   action == 0: self._rollout_actions["probe"] += 1
            elif action == 1: self._rollout_actions["flag"]  += 1
            elif action == 2: self._rollout_actions["pass"]  += 1

            # --- Collect outcome only from terminal steps ------------------
            outcome = info.get("outcome")   # None on PROBE steps
            if outcome is not None:
                self._outcomes.append(outcome)
                self._stages.append(info.get("stage", 0))

        return True   # returning False would stop training

    def _on_rollout_end(self) -> None:
        """
        Called once the rollout buffer is full, just before PPO updates.
        This is the right place to flush metrics: all n_steps × n_envs
        transitions for this rollout have already passed through _on_step().

        We compute and log:
          Outcome counts   : tp, fp, tn, fn (rolling window)
          Precision        : tp / (tp + fp)   — of flagged, how many real fraud
          Recall           : tp / (tp + fn)   — of all fraud, how many caught
          F1               : harmonic mean of precision + recall
          Action rates     : fraction of steps that were probe/flag/pass
          Mean stage       : average stage at episode end (how deeply we probed)
        """
        if not self._outcomes:
            return   # nothing logged yet

        # ── Outcome counts from the rolling window ────────────────────────
        tp = self._outcomes.count("tp")
        fp = self._outcomes.count("fp")
        tn = self._outcomes.count("tn")
        fn = self._outcomes.count("fn")
        total_terminal = tp + fp + tn + fn

        # ── Classification metrics ────────────────────────────────────────
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # false positive rate
        accuracy  = (tp + tn) / total_terminal if total_terminal > 0 else 0.0

        # ── Action rates for this rollout ─────────────────────────────────
        n = max(self._rollout_steps, 1)
        probe_rate = self._rollout_actions["probe"] / n
        flag_rate  = self._rollout_actions["flag"]  / n
        pass_rate  = self._rollout_actions["pass"]  / n

        # ── Mean decision stage ───────────────────────────────────────────
        mean_stage = (sum(self._stages) / len(self._stages)
                      if self._stages else 0.0)

        # ── Write to TensorBoard via SB3's logger ─────────────────────────
        # self.logger.record() buffers values; SB3 calls logger.dump()
        # after this hook, so everything appears in the same TensorBoard step.
        self.logger.record("fraud/precision",   precision)
        self.logger.record("fraud/recall",      recall)
        self.logger.record("fraud/f1",          f1)
        self.logger.record("fraud/fpr",         fpr)
        self.logger.record("fraud/accuracy",    accuracy)

        self.logger.record("fraud/tp",  tp)
        self.logger.record("fraud/fp",  fp)
        self.logger.record("fraud/tn",  tn)
        self.logger.record("fraud/fn",  fn)

        self.logger.record("actions/probe_rate", probe_rate)
        self.logger.record("actions/flag_rate",  flag_rate)
        self.logger.record("actions/pass_rate",  pass_rate)
        self.logger.record("actions/mean_stage", mean_stage)

        if self.verbose:
            step = self.num_timesteps
            print(
                f"[FraudCB] step={step:>8,}  "
                f"P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}  "
                f"FPR={fpr:.3f}  probe={probe_rate:.2f}  "
                f"stage={mean_stage:.2f}"
            )