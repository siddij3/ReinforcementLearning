"""
tb_callback.py
Custom TensorBoard callback for the fraud-detection PPO agent.

Logs everything SB3 doesn't log by default:
  fraud/precision          — rolling true-positive rate
  fraud/recall             — rolling true-negative rate
  fraud/f1                 — harmonic mean of precision & recall
  fraud/false_positive_rate
  fraud/false_negative_rate
  actions/probe_rate       — fraction of PROBE actions
  actions/flag_rate        — fraction of FLAG actions
  actions/pass_rate        — fraction of PASS actions
  episode/mean_reward      — explicit redundant log (useful for grouping)
  episode/mean_length
"""

from collections import deque
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class FraudTensorboardCallback(BaseCallback):
    """
    Writes fraud-detection metrics into SB3's existing TensorBoard writer
    so everything appears in the same run directory, in the same session.

    Parameters
    ----------
    window : int
        Number of recent episodes used for rolling metric calculation.
    verbose : int
        0 = silent, 1 = print metric summary every log step.
    """

    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window = window

        # ── Rolling episode buffers ───────────────────────────────
        self._rewards = deque(maxlen=window)
        self._lengths = deque(maxlen=window)

        # Outcome counts  (tp / fp / fn / tn)
        self._tp = deque(maxlen=window)
        self._fp = deque(maxlen=window)
        self._fn = deque(maxlen=window)
        self._tn = deque(maxlen=window)

        # Action counts per step
        self._n_probe = deque(maxlen=window * 10)
        self._n_flag  = deque(maxlen=window * 10)
        self._n_pass  = deque(maxlen=window * 10)

    # ── SB3 hook: called after every env.step() ───────────────────
    def _on_step(self) -> bool:

        # ── 1. Episode completions ────────────────────────────────
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if not done:
                continue

            # Episode reward / length via Monitor wrapper
            ep = info.get("episode")
            if ep:
                self._rewards.append(ep["r"])
                self._lengths.append(ep["l"])

            # Fraud outcome — emitted by CandidateEnv.step()
            outcome = info.get("outcome")
            self._tp.append(1 if outcome == "tp" else 0)
            self._fp.append(1 if outcome == "fp" else 0)
            self._fn.append(1 if outcome == "fn" else 0)
            self._tn.append(1 if outcome == "tn" else 0)

        # ── 2. Action distribution ────────────────────────────────
        actions = self.locals.get("actions", [])
        for a in np.array(actions).flatten():
            self._n_probe.append(1 if a == 0 else 0)
            self._n_flag.append( 1 if a == 1 else 0)
            self._n_pass.append( 1 if a == 2 else 0)

        # ── 3. Write to TensorBoard every n_steps (one rollout) ──
        #    self.num_timesteps is updated by SB3 before _on_step
        n = self.model.n_steps * self.model.n_envs   # steps per rollout
        if self.num_timesteps % n != 0:
            return True

        step = self.num_timesteps

        # ── Episode metrics ───────────────────────────────────────
        if self._rewards:
            self.logger.record("episode/mean_reward", np.mean(self._rewards))
            self.logger.record("episode/mean_length", np.mean(self._lengths))

        # ── Fraud metrics ─────────────────────────────────────────
        tp = sum(self._tp);  fp = sum(self._fp)
        fn = sum(self._fn);  tn = sum(self._tn)

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        fpr       = fp / max(fp + tn, 1)
        fnr       = fn / max(fn + tp, 1)
        f1        = (2 * precision * recall) / max(precision + recall, 1e-8)

        self.logger.record("fraud/precision",           precision)
        self.logger.record("fraud/recall",              recall)
        self.logger.record("fraud/f1",                  f1)
        self.logger.record("fraud/false_positive_rate", fpr)
        self.logger.record("fraud/false_negative_rate", fnr)

        # ── Action distribution ───────────────────────────────────
        total = max(sum(self._n_probe) + sum(self._n_flag) + sum(self._n_pass), 1)
        self.logger.record("actions/probe_rate", sum(self._n_probe) / total)
        self.logger.record("actions/flag_rate",  sum(self._n_flag)  / total)
        self.logger.record("actions/pass_rate",  sum(self._n_pass)  / total)

        # Flush writes the scalars to disk immediately
        self.logger.dump(step)

        if self.verbose:
            print(
                f"[TB] step={step:>8,} | "
                f"reward={np.mean(self._rewards) if self._rewards else 0:.3f} | "
                f"prec={precision:.3f}  rec={recall:.3f}  f1={f1:.3f} | "
                f"probe={sum(self._n_probe)/total:.2f}  "
                f"flag={sum(self._n_flag)/total:.2f}  "
                f"pass={sum(self._n_pass)/total:.2f}"
            )

        return True