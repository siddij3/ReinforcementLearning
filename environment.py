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
  Stage 0 : first screening question answered  (L1 surface)
  Stage 1 : second screening question answered (L2 depth)
  Stage 2 : cross-answer consistency signals
  Stage 3 : profile signals only
  Stage 4 : web/external verification (most expensive, revealed last) - Not implemented

"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CandidateEnv(gym.Env):
    MAX_STAGES = 5
    # 11 (stage_zero + pad) + 10 (stage_one + pads) + 1 (depth) + 12 (profile) — must match _build_feature_cache
    FEATURE_DIM = 34

    # Probe costs — stage 4 entry removed: reaching stage 4 forces a PASS,
    # so the agent never pays a probe cost at stage 4.
    _PROBE_COSTS = {0: -0.05, 
                    1: -0.08, 
                    2: -0.10, 
                    3: -0.15,
                    4: -0.20}

    def __init__(self, signal_processor=None, *, debug: bool = True):
        super().__init__()
        self.signals = signal_processor
        self.debug = debug
        self._feature_cache = np.zeros(self.FEATURE_DIM, dtype=np.float32)

        # ── Observation space ──────────────────────────────────────────────
        # Shape: FEATURE_DIM + MAX_STAGES (one-hot).
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

        `seed` is forwarded to the parent to seed self.np_random. Stable-Baselines3
        passes ``seed + env_index`` on the first reset of each sub-env so parallel
        envs must not share a fixed profile RNG (see ``SignalProcessor.rng``).

        Returns
        -------
        obs  : np.ndarray shape (,)  — profile features + stage one-hot
        info : dict                    — empty at reset (required by Gymnasium)
        """
        super().reset(seed=seed)
        if seed is not None:
            self.signals.rng = np.random.default_rng(int(seed))
        else:
            # VecEnv clears seeds after each reset; derive a fresh stream from the env RNG.
            self.signals.rng = np.random.default_rng(
                int(self.np_random.integers(0, 2**31 - 1, dtype=np.int32))
            )
        self.candidate = self.signals.random_profile()
        self.stage     = 0
        self.done      = False
        self._build_feature_cache()
        if self.debug:
            print("--------------------------------")
            print(self.candidate.summary)
            print("--------------------------------")
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
        obs        : np.ndarray shape (,)
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
        is_fraud   = self.candidate.is_fraud
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

    def _build_feature_cache(self) -> None:
        """
        Run all signal stages once per episode. _revealed_features() only masks
        by current stage — no repeated scoring on every observation.
        """
        c = self.candidate
        cache = self._feature_cache
        cache.fill(0.0)

        stage_zero = self.signals.stage_zero(c)
        print(stage_zero)
        cache[0] = stage_zero.get("causal_span_score", 0.0)
        cache[1] = stage_zero.get("result_entity_score", 0.0)
        cache[2] = stage_zero.get("coherence_score", 0.0)
        cache[3] = stage_zero.get("answer_perplexity_score", 0.0)
        cache[4] = stage_zero.get("raw_perplexity", 0.0)
        cache[5] = stage_zero.get("conditioned_perplexity", 0.0)
        cache[6] = stage_zero.get("mean_log_prob", 0.0)
        cache[7] = stage_zero.get("log_prob_cv", 0.0)
        cache[8] = stage_zero.get("high_prob_token_fraction", 0.0)
        cache[9] = stage_zero.get("low_prob_token_fraction", 0.0)
        cache[10] = 0.0  # pad so block 0 matches out[0:11]

        stage_one = self.signals.stage_one(c)
        print(stage_one)
        cache[11] = stage_one.get("semantic_consistency", 0.0)
        cache[12] = stage_one.get("factual_consistency", 0.0)
        cache[13] = stage_one.get("operational_specificity_score", 0.0)
        cache[14] = stage_one.get("artifact_count", 0.0)
        cache[15] = stage_one.get("weighted_density", 0.0)
        cache[16] = stage_one.get("type_diversity", 0.0)
        cache[17] = 0.0
        cache[18] = 0.0
        cache[19] = 0.0
        cache[20] = 0.0

        stage_two = self.signals.stage_two(c)
        print(stage_two)
        cache[21] = stage_two.get("depth_collapse_delta", 0.0)

        stage_three = self.signals.stage_three(c)
        print(stage_three)
        cache[22] = stage_three.get("coverage_score", 0.0)
        cache[23] = stage_three.get("idiosyncrasy_score", 0.0)
        cache[24] = stage_three.get("semantic_mirror_score", 0.0)
        cache[25] = stage_three.get("timeline_coherence_fraud_score", 0.0)
        cache[26] = stage_three.get("career_smoothness_fraud_score", 0.0)
        cache[27] = stage_three.get("profile_voice_fraud_score", 0.0)
        cache[28] = stage_three.get("soft_skill_mean_rate", 0.0)
        cache[29] = stage_three.get("soft_skill_cv", 0.0)
        cache[30] = stage_three.get("sentence_uniformity", 0.0)
        cache[31] = stage_three.get("tech_density_cv", 0.0)
        cache[32] = stage_three.get("voice_cluster_penalty", 0.0)
        cache[33] = 0.0  # reserved for future use (e.g. stage_four git signal)

    def _revealed_features(self) -> np.ndarray:
        out = np.zeros(self.FEATURE_DIM, dtype=np.float32)
        out[0:11] = self._feature_cache[0:11]
        if self.stage >= 1:
            out[11:21] = self._feature_cache[11:21]
        if self.stage >= 2:
            out[21] = self._feature_cache[21]
        if self.stage >= 3:
            out[22:34] = self._feature_cache[22:34]
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
        fraud = getattr(c, "is_fraud", "?") if c is not None else "?"
        obs   = self._revealed_features()

        print(f"\n  Stage [{self.stage}] {stage}  |  is_fraud={fraud}")
        feature_names = [
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
            "feature_pad_10",
            "semantic_consistency",
            "factual_consistency",
            "operational_specificity_score",
            "artifact_count",
            "weighted_density",
            "type_diversity",
            "pad_17",
            "pad_18",
            "pad_19",
            "pad_20",
            "depth_collapse_delta",
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
            "reserved_33",
        ]
        for i, (name, val) in enumerate(zip(feature_names, obs)):
            if val != 0.0:
                bar_len = int(val * 20)
                bar     = "█" * bar_len + "░" * (20 - bar_len)
                print(f"    [{i:2d}] {name:<32} {bar} {val:.3f}")