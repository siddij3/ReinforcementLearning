"""
Reward structure
────────────────
  PROBE at stage s : _PROBE_COSTS[s]  (negative, increasing cost per stage)
  FLAG  correct    : +1.0  (true positive)
  FLAG  wrong      : -1.0  (false positive — wasted a hire, mild)
  PASS  correct    : +1.0  (true negative)
  PASS  wrong      : -2.0  (false negative — hired fraud, expensive)

Stage mapping
──────────────
  Stage 0 : profile signals only (always available)
  Stage 1 : first screening question answered  (L1 surface)
  Stage 2 : second screening question answered (L2 depth)
  Stage 3 : cross-answer consistency signals
  Stage 4 : web/external verification (most expensive, revealed last) - Not implemented

Feature dimensions
───────────────────
  Stage 0 profile    :  5 features
  Stage 1 Q1 screen  :  4 features
  Stage 2 Q2 depth   :  3 features
  Stage 3 consistency:  2 features
  Stage 4 web        :  3 features
  ─────────────────────────────
  Total features     : 17
  Stage one-hot      :  5
  Observation dim    : 22
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CandidateEnv(gym.Env):
    MAX_STAGES = 5
    FEATURE_DIM = 24   # total features at full observation

    # Probe costs — stage 4 entry removed: reaching stage 4 forces a PASS,
    # so the agent never pays a probe cost at stage 4.
    _PROBE_COSTS = {0: -0.05, 1: -0.08, 2: -0.10, 3: -0.15}

    def __init__(self, signal_processor=None):
        super().__init__()
        self.signals = signal_processor

        # ── Observation space ──────────────────────────────────────────────
        # Shape: (17 features) + (5-dim stage one-hot) = 22 floats.
        # Bounds [0, 1] match the actual data range produced by
        # CandidateGenerator: all features are np.clip(v, 0, 1) and the
        # stage one-hot is either 0.0 or 1.0.
        # FIX: was (-3.0, 3.0) — incorrect and misleads normalization wrappers.
        self.observation_space = spaces.Box(
            low   = 0.0,
            high  = 1.0,
            shape = (self.FEATURE_DIM + self.MAX_STAGES,),
            dtype = np.float32,
        )

        # ── Action space ───────────────────────────────────────────────────
        # Discrete(3): 0 = PROBE, 1 = FLAG (fraud), 2 = PASS (genuine)
        self.action_space = spaces.Discrete(3)

        self.candidate = None
        self.stage     = 0
        self.done      = False

    # ── Reset ──────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        """
        Gymnasium-compliant reset.

        `seed` is forwarded to the parent to seed self.np_random (SB3
        passes seeds automatically when using make_vec_env). Sampling
        new candidates uses the CandidateGenerator's own RNG, which is
        independent and set at construction time.

        Returns
        -------
        obs  : np.ndarray shape (22,)  — profile features + stage one-hot
        info : dict                    — empty at reset (required by Gymnasium)
        """
        super().reset(seed=seed)
        self.candidate = self.signals.random_profile()
        self.stage     = 0
        self.done      = False
        return self._obs(), {}

    # ── Step ───────────────────────────────────────────────────────────────

    def step(self, action: int):
        """
        Apply one action and advance the environment.

        Parameters
        ----------
        action : int
            0 PROBE  — pay a cost to reveal the next stage's features.
            1 FLAG   — terminate and declare this candidate as fraud.
            2 PASS   — terminate and accept this candidate as genuine.

        Returns
        -------
        obs        : np.ndarray shape (22,)
        reward     : float
        terminated : bool   — True when FLAG or PASS is chosen
        truncated  : bool   — always False (no time-limit truncation)
        info       : dict
            is_fraud   : bool
            outcome    : "tp" | "fp" | "tn" | "fn" | None
            stage      : int
            stage_name : str
        """
        assert not self.done, "Episode already terminated — call reset()"

        reward     = 0.0
        terminated = False
        is_fraud   = self.candidate["is_fraud"]
        outcome    = None   # only set on terminal steps

        if action == 0:                                 # ── PROBE ──────────
            if self.stage < self.MAX_STAGES - 1:
                # Pay the stage cost and advance to the next information tier.
                reward      = self._PROBE_COSTS[self.stage]
                self.stage += 1
            else:
                # Already at the final stage — no more signals available.
                # Force a PASS so the episode doesn't loop forever.
                # The fall-through to the PASS elif handles the reward.
                action = 2

        if action == 1:                                 # ── FLAG ───────────
            # Declare the candidate as fraudulent.
            reward     =  1.0 if is_fraud else -1.0
            outcome    = "tp" if is_fraud else "fp"
            terminated = True

        elif action == 2:                               # ── PASS ───────────
            # Accept the candidate as genuine.
            reward     =  1.0 if not is_fraud else -2.0
            outcome    = "tn" if not is_fraud else "fn"
            terminated = True

        self.done = terminated

        info = {
            "is_fraud":   is_fraud,
            "outcome":    outcome,       # consumed by FraudTensorboardCallback
            "stage":      self.stage,
            "stage_name": self._stage_name(self.stage),
        }

        return self._obs(), reward, terminated, False, info

    # ── Observation ────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        """
        Concatenate the revealed feature vector with the stage one-hot.
        """
        features = self._revealed_features()
        stage_oh = np.zeros(self.MAX_STAGES, dtype=np.float32)
        stage_oh[self.stage] = 1.0
        return np.concatenate([features, stage_oh])

    def _revealed_features(self) -> np.ndarray:

        out = np.zeros(self.FEATURE_DIM, dtype=np.float32)
        c = self.candidate
        
        # ── Stage 0: Profile signals ───────────────────────────────────────
        stage_zero = self.stage_zero(c)
        out[0] = stage_zero.get("coverage_score",          0.0)
        out[1] = stage_zero.get("idiosyncrasy_score",     0.0)
        out[2] = stage_zero.get("semantic_mirror_score",  0.0)
        out[3] = stage_zero.get("timeline_coherence_fraud_score", 0.0)
        out[4] = stage_zero.get("career_smoothness_fraud_score",      0.0)
        out[5] = stage_zero.get("profile_voice_fraud_score",      0.0)
        out[6] = stage_zero.get("soft_skill_mean_rate",      0.0)
        out[7] = stage_zero.get("soft_skill_cv",      0.0)
        out[8] = stage_zero.get("sentence_uniformity",      0.0)
        out[9] = stage_zero.get("tech_density_cv",      0.0)
        out[10] = stage_zero.get("voice_cluster_penalty",      0.0)

        # ── Stage 1: First screening question (L1 surface) ────────────────
        stage_one = self.stage_one(c)
        if self.stage >= 1:
            out[11] = stage_one.get("causal_span_score", 0.0)
            out[12] = stage_one.get("result_entity_score",       0.0)
            out[13] = stage_one.get("coherence_score", 0.0)
            out[14] = stage_one.get("answer_perplexity_score",    0.0)
            out[15] = stage_one.get("raw_perplexity",    0.0)
            out[16] = stage_one.get("conditioned_perplexity",    0.0)
            out[17] = stage_one.get("mean_log_prob",    0.0)
            out[18] = stage_one.get("log_prob_cv",    0.0)
            out[19] = stage_one.get("high_prob_token_fraction",    0.0)
            out[20] = stage_one.get("low_prob_token_fraction",    0.0)

        # ── Stage 2: Deep follow-up question (L2) ─────────────────────────
        stage_two = self.stage_two(c)
        if self.stage >= 2:
            out[21] = stage_two.get("depth_collapse_delta", 0.0)

        # ── Stage 3: Cross-answer consistency ─────────────────────────────
        stage_three = self.stage_three(c)
        if self.stage >= 3:
            out[22] = stage_three.get("cross_answer_consistency_fraud_score", 0.0)
            out[23] = stage_three.get("combined_consistency_score",          0.0)


        return out

    # ── Utilities ──────────────────────────────────────────────────────────

    @staticmethod
    def _stage_name(stage: int) -> str:
        return {
            0: "profile_review",
            1: "screening_q1",
            2: "screening_q2_depth",
            3: "cross_answer_consistency",
        }.get(stage, "unknown")

    def render(self):
        """Human-readable state summary for interactive debugging."""
        c     = self.candidate
        stage = self._stage_name(self.stage)
        fraud = c.get("is_fraud", "?")
        obs   = self._revealed_features()

        print(f"\n  Stage [{self.stage}] {stage}  |  is_fraud={fraud}")
        feature_names = [
            "coverage_score",
            "idiosyncrasy_score",
            "semantic_mirror_score",
            "timeline_coherence_fraud_score",
            "career_smoothness_fraud_score",
            "profile_voice_fraud_score",
            "soft_skill_mean_rate",
            "soft_skill_cv",
            "sentence_uniformity",
            "tech_density_cv",
            "voice_cluster_penalty",
            "causal_span_score",
            "result_entity_score",
            "coherence_score",
            "answer_perplexity_score",
            "raw_perplexity",
            "conditioned_perplexity",
            "mean_log_prob",
            "log_prob_cv",
            "high_prob_token_fraction",
            "low_prob_token_fraction",
            "depth_collapse_delta",
            "cross_answer_consistency_fraud_score",
            "combined_consistency_score",
        ]
        for i, (name, val) in enumerate(zip(feature_names, obs)):
            if val != 0.0:
                bar_len = int(val * 20)
                bar     = "█" * bar_len + "░" * (20 - bar_len)
                print(f"    [{i:2d}] {name:<32} {bar} {val:.3f}")