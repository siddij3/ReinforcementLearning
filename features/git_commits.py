"""
features/github_commit_archaeology.py
──────────────────────────────────────────────────────────────────────────────
GitHubCommitArchaeologyScorer

Analyses GitHub activity signals for evidence of expertise fabrication.

Two operating modes
───────────────────
  AGGREGATE mode   – uses only the scalar web_signals dict that every
                     LinkedInSyntheticProfile already carries.  No API key
                     required; works entirely on the values your pipeline
                     already collects.

  DETAILED mode    – additionally accepts a `CommitHistory` object built from
                     live GitHub API responses (or a pre-fetched JSON cache).
                     Unlocks commit-pattern, burstiness, language-evolution,
                     and repo-age-vs-tenure sub-scores.

Both modes return the same output schema so the RL feature vector stays stable
regardless of which mode is active.

Fraud signal logic (summary)
─────────────────────────────
  High fraud probability ← low repo count  AND  very stale commits
                         ← claimed tenure predates oldest repo by many years
                         ← overnight language portfolio explosion (no gradual growth)
                         ← commit bursts that are atypically short and dense
                           then go completely dark (LLM-assisted repo stuffing)
                         ← no community surface area (stars, forks, PRs) on repos
                           despite large claimed commit volume
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data containers for DETAILED mode
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CommitRecord:
    """One commit from the GitHub API (or a mock)."""
    timestamp: datetime           # UTC
    repo_name: str
    language: str                 # primary language of the repo at commit time
    additions: int = 0
    deletions: int = 0
    is_merge: bool = False        # merge commits carry less signal


@dataclass
class RepoRecord:
    """Per-repo summary fetched from GitHub API."""
    name: str
    created_at: datetime          # UTC
    primary_language: str
    stargazers: int = 0
    forks: int = 0
    open_issues: int = 0
    last_pushed: Optional[datetime] = None
    is_fork: bool = False         # forked repos inflate counts


@dataclass
class CommitHistory:
    """
    Full detailed payload for DETAILED mode.

    Build this from GitHub REST / GraphQL responses and pass it to
    GitHubCommitArchaeologyScorer alongside the standard web_signals dict.
    """
    commits: List[CommitRecord] = field(default_factory=list)
    repos: List[RepoRecord] = field(default_factory=list)
    profile_github_username: Optional[str] = None

    # ── convenience properties ──────────────────────────────────────────────

    @property
    def original_repos(self) -> List[RepoRecord]:
        return [r for r in self.repos if not r.is_fork]

    @property
    def non_merge_commits(self) -> List[CommitRecord]:
        return [c for c in self.commits if not c.is_merge]

    @property
    def commit_timestamps(self) -> List[datetime]:
        return sorted(c.timestamp for c in self.non_merge_commits)

    @property
    def languages_over_time(self) -> Dict[str, datetime]:
        """First commit timestamp per language – proxies when expertise started."""
        first: Dict[str, datetime] = {}
        for c in self.non_merge_commits:
            lang = c.language or "Unknown"
            if lang not in first or c.timestamp < first[lang]:
                first[lang] = c.timestamp
        return first


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GitHubArchaeologyResult:
    """
    Returned by GitHubCommitArchaeologyScorer.score().

    All sub-scores are in [0, 1] where 1.0 = maximum fraud signal.
    The composite github_archaeology_fraud_score is a weighted combination.
    """
    # ── aggregate-mode sub-scores (always populated) ────────────────────────
    recency_penalty: float = 0.0          # stale last-push relative to claimed tenure
    repo_volume_penalty: float = 0.0      # repo count vs years of experience
    web_signal_composite: float = 0.0     # aggregate-only fraud score

    # ── detailed-mode sub-scores (NaN when detailed data unavailable) ───────
    burstiness_score: float = float("nan")        # short dense bursts then silence
    language_explosion_score: float = float("nan") # overnight polyglot jump
    repo_age_tenure_gap: float = float("nan")     # repo history vs claimed start year
    community_surface_score: float = float("nan") # stars/forks/PRs vs commit volume
    working_hours_score: float = float("nan")     # inhuman commit time distribution

    # ── composite ────────────────────────────────────────────────────────────
    github_archaeology_fraud_score: float = 0.0
    mode: str = "aggregate"               # "aggregate" | "detailed"


# ─────────────────────────────────────────────────────────────────────────────
# Scorer
# ─────────────────────────────────────────────────────────────────────────────

class GitHubCommitArchaeologyScorer:
    """
    Usage (aggregate mode – no GitHub API key needed)
    ─────────────────────────────────────────────────
        scorer = GitHubCommitArchaeologyScorer(
            web_signals=profile.web_signals,
            timeline=profile.timeline,
            claimed_skills=profile.skills,
        )
        result = scorer.score()

    Usage (detailed mode)
    ─────────────────────
        history = CommitHistory(commits=[...], repos=[...])
        scorer = GitHubCommitArchaeologyScorer(
            web_signals=profile.web_signals,
            timeline=profile.timeline,
            claimed_skills=profile.skills,
            commit_history=history,
        )
        result = scorer.score()
    """

    # Hyper-parameters (tunable from RL reward shaping)
    RECENCY_STALE_DAYS: int = 180        # commits older than this are suspicious
    RECENCY_DEAD_DAYS: int = 365         # beyond this → strong fraud signal
    MIN_REPOS_PER_YEAR: float = 1.5      # legitimate practitioners accumulate repos
    BURST_WINDOW_DAYS: int = 14          # look for dense creation in short windows
    BURST_COMMIT_THRESHOLD: int = 30     # commits in BURST_WINDOW_DAYS = suspicious
    LANGUAGE_JUMP_MONTHS: int = 3        # acquiring N new languages in < this = flag
    LANGUAGE_JUMP_COUNT: int = 4         # N new languages
    TENURE_REPO_GAP_YEARS: float = 2.0  # repo history lags claimed start by > this

    # Aggregate-mode score weights
    _AGG_WEIGHTS = {
        "recency": 0.45,
        "volume": 0.35,
        "endorsement_reciprocity": 0.20,
    }

    # Detailed-mode score weights (aggregate sub-scores are still included)
    _DET_WEIGHTS = {
        "recency": 0.20,
        "volume": 0.10,
        "burstiness": 0.25,
        "language_explosion": 0.20,
        "repo_age_tenure_gap": 0.15,
        "community_surface": 0.10,
    }

    def __init__(
        self,
        web_signals: Dict[str, float],
        timeline: list,                        # List[TimelineEntry] from your dataclass
        claimed_skills: List[str],
        commit_history: Optional[CommitHistory] = None,
        reference_date: Optional[datetime] = None,
    ):
        self.web_signals = web_signals
        self.timeline = timeline
        self.claimed_skills = claimed_skills
        self.commit_history = commit_history
        self.ref_date = reference_date or datetime.now(timezone.utc)

    # ── public entry point ───────────────────────────────────────────────────

    def score(self) -> GitHubArchaeologyResult:
        result = GitHubArchaeologyResult()

        # ── always run aggregate sub-scores ─────────────────────────────────
        result.recency_penalty = self._score_recency()
        result.repo_volume_penalty = self._score_repo_volume()
        reciprocity_penalty = self._score_endorsement_reciprocity()
        result.web_signal_composite = (
            self._AGG_WEIGHTS["recency"] * result.recency_penalty
            + self._AGG_WEIGHTS["volume"] * result.repo_volume_penalty
            + self._AGG_WEIGHTS["endorsement_reciprocity"] * reciprocity_penalty
        )

        if self.commit_history is None:
            # ── aggregate-only final score ───────────────────────────────────
            result.github_archaeology_fraud_score = result.web_signal_composite
            result.mode = "aggregate"
            return result

        # ── detailed sub-scores ───────────────────────────────────────────────
        result.mode = "detailed"
        result.burstiness_score = self._score_burstiness()
        result.language_explosion_score = self._score_language_explosion()
        result.repo_age_tenure_gap = self._score_repo_age_tenure_gap()
        result.community_surface_score = self._score_community_surface()
        result.working_hours_score = self._score_working_hours()

        w = self._DET_WEIGHTS
        result.github_archaeology_fraud_score = _clip(
            w["recency"] * result.recency_penalty
            + w["volume"] * result.repo_volume_penalty
            + w["burstiness"] * result.burstiness_score
            + w["language_explosion"] * result.language_explosion_score
            + w["repo_age_tenure_gap"] * result.repo_age_tenure_gap
            + w["community_surface"] * result.community_surface_score
        )
        return result

    # ── aggregate sub-scorers ────────────────────────────────────────────────

    def _score_recency(self) -> float:
        """
        How stale is the most recent GitHub push relative to the candidate's
        claimed current employment?  A practitioner who claims to be actively
        coding should have commits in the last few months.
        """
        recency_days = self.web_signals.get("github_commit_recency_days", None)
        if recency_days is None:
            return 0.5                   # missing signal → neutral

        if recency_days <= 30:
            return 0.0                   # very active → no signal
        if recency_days <= self.RECENCY_STALE_DAYS:
            # linear ramp from 30 → STALE_DAYS
            return _linear_ramp(recency_days, 30, self.RECENCY_STALE_DAYS, 0.0, 0.4)
        if recency_days <= self.RECENCY_DEAD_DAYS:
            # steeper ramp from STALE → DEAD
            return _linear_ramp(recency_days, self.RECENCY_STALE_DAYS, self.RECENCY_DEAD_DAYS, 0.4, 0.85)
        return 1.0                       # > 1 year dark → maximum suspicion

    def _score_repo_volume(self) -> float:
        """
        Legitimate practitioners accumulate repos gradually over their career.
        Very few repos despite claiming many years = fraud signal.
        Very many repos for a short career could be bulk-created = also suspicious.
        """
        repo_count = self.web_signals.get("github_public_repos", None)
        if repo_count is None:
            return 0.5

        career_years = self._career_years()
        if career_years <= 0:
            return 0.5

        expected_repos = career_years * self.MIN_REPOS_PER_YEAR
        ratio = repo_count / max(expected_repos, 1.0)

        if ratio >= 0.8:
            return 0.0                   # healthy accumulation
        if ratio >= 0.4:
            return _linear_ramp(ratio, 0.8, 0.4, 0.0, 0.5)
        if ratio >= 0.1:
            return _linear_ramp(ratio, 0.4, 0.1, 0.5, 0.9)
        return 1.0                       # almost no repos → very suspicious

    def _score_endorsement_reciprocity(self) -> float:
        """
        Low endorsement reciprocity means the candidate endorsed many people
        but received few back — consistent with mass-connection farming.
        """
        er = self.web_signals.get("endorsement_reciprocity", None)
        if er is None:
            return 0.3
        # er ∈ [0, 1]; fraud profiles cluster around 0.05–0.15
        return _clip(1.0 - er)

    # ── detailed sub-scorers ─────────────────────────────────────────────────

    def _score_burstiness(self) -> float:
        """
        LLM-assisted portfolio stuffing produces short, dense commit bursts
        followed by prolonged silence.  We measure the Gini coefficient of
        weekly commit counts — high Gini = highly uneven = suspicious.
        We also penalise a specific pattern: >THRESHOLD commits in any
        BURST_WINDOW_DAYS period with a subsequent dead period ≥ 90 days.
        """
        timestamps = self.commit_history.commit_timestamps
        if len(timestamps) < 5:
            return 0.6                   # too few commits to judge, slight suspicion

        # ── Gini of weekly commit counts ────────────────────────────────────
        gini = _gini_coefficient(self._weekly_commit_counts(timestamps))

        # ── burst-then-silence pattern ───────────────────────────────────────
        burst_penalty = self._detect_burst_silence(timestamps)

        return _clip(0.6 * gini + 0.4 * burst_penalty)

    def _score_language_explosion(self) -> float:
        """
        Genuine polyglots grow their language portfolio gradually.
        Acquiring ≥ LANGUAGE_JUMP_COUNT languages within LANGUAGE_JUMP_MONTHS
        is a strong signal of bulk-created repos (LLM-generated stubs in many
        languages to make a profile look broad).
        """
        lang_first = self.commit_history.languages_over_time
        if len(lang_first) < 2:
            return 0.0

        # sort languages by first appearance
        sorted_langs = sorted(lang_first.items(), key=lambda x: x[1])
        max_explosion = 0.0

        # slide a window of LANGUAGE_JUMP_MONTHS and count new languages
        window = timedelta(days=self.LANGUAGE_JUMP_MONTHS * 30)
        for i, (lang, t_start) in enumerate(sorted_langs):
            count_in_window = sum(
                1 for _, t in sorted_langs[i:]
                if t - t_start <= window
            )
            if count_in_window >= self.LANGUAGE_JUMP_COUNT:
                # score scales with how many languages appeared in the window
                explosion = min(count_in_window / 8.0, 1.0)
                max_explosion = max(max_explosion, explosion)

        return _clip(max_explosion)

    def _score_repo_age_tenure_gap(self) -> float:
        """
        The candidate claims to have started their career at self.timeline[0].start.
        The oldest non-fork repo should roughly match that date (±18 months is fine).
        A large gap where claimed tenure predates any GitHub activity suggests the
        GitHub account was created after-the-fact to support a fabricated history.
        """
        original_repos = self.commit_history.original_repos
        if not original_repos:
            return 0.8                   # no original repos at all → suspicious

        oldest_repo_date = min(r.created_at for r in original_repos)

        # Infer career start from timeline
        if not self.timeline:
            return 0.5
        career_start = min(
            e.start for e in self.timeline
            if hasattr(e, "start") and e.start is not None
        )
        # Normalise timezone
        if oldest_repo_date.tzinfo is None:
            oldest_repo_date = oldest_repo_date.replace(tzinfo=timezone.utc)
        if career_start.tzinfo is None:
            career_start = career_start.replace(tzinfo=timezone.utc)

        gap_years = (oldest_repo_date - career_start).days / 365.25
        # Negative gap = repos predate claimed career start (fine or overachiever)
        # Positive gap = claimed career started years before any repo (suspicious)
        if gap_years <= 1.5:
            return 0.0
        if gap_years <= self.TENURE_REPO_GAP_YEARS:
            return _linear_ramp(gap_years, 1.5, self.TENURE_REPO_GAP_YEARS, 0.0, 0.5)
        if gap_years <= 4.0:
            return _linear_ramp(gap_years, self.TENURE_REPO_GAP_YEARS, 4.0, 0.5, 0.95)
        return 1.0

    def _score_community_surface(self) -> float:
        """
        Real practitioners leave community surface area: stars, forks, PRs merged
        in others' repos, issues opened.  A large claimed commit history with zero
        community signal is consistent with private-and-then-made-public stuffing.
        """
        repos = self.commit_history.original_repos
        if not repos:
            return 0.8

        total_stars = sum(r.stargazers for r in repos)
        total_forks = sum(r.forks for r in repos)
        total_commits = len(self.commit_history.non_merge_commits)

        # stars+forks per 100 commits — legitimates tend to have ≥ 5
        if total_commits == 0:
            return 0.7
        community_ratio = (total_stars + total_forks * 2) / max(total_commits / 100, 1)

        if community_ratio >= 5.0:
            return 0.0
        if community_ratio >= 1.0:
            return _linear_ramp(community_ratio, 5.0, 1.0, 0.0, 0.5)
        return _linear_ramp(community_ratio, 1.0, 0.0, 0.5, 1.0)

    def _score_working_hours(self) -> float:
        """
        Human developers commit across a distribution of hours that roughly
        follows a bimodal pattern (morning focus block + afternoon focus block).
        LLM-assisted bulk creation tends to produce commits clustered in very
        narrow windows or uniformly distributed across all 24 hours (non-human).

        Score = 1 - entropy_normalised; high entropy (uniform) → high fraud signal.
        Low entropy (all at 3 AM) is also suspicious.
        """
        timestamps = self.commit_history.commit_timestamps
        if len(timestamps) < 10:
            return float("nan")

        hours = [t.hour for t in timestamps]
        # Build 24-bin histogram
        hist, _ = np.histogram(hours, bins=24, range=(0, 24))
        hist = hist + 1e-9              # Laplace smooth
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log(hist))

        # Max entropy for 24 bins = ln(24) ≈ 3.178
        # Human normal entropy ≈ 2.1–2.7
        # Uniform (bot) entropy ≈ 3.1
        # All-at-3am entropy ≈ 0.4
        max_entropy = math.log(24)
        norm_entropy = entropy / max_entropy

        # Both extremes are suspicious; penalise deviation from [0.55, 0.85]
        if 0.55 <= norm_entropy <= 0.85:
            return 0.0
        if norm_entropy > 0.85:
            return _linear_ramp(norm_entropy, 0.85, 1.0, 0.0, 0.9)
        return _linear_ramp(norm_entropy, 0.55, 0.0, 0.0, 0.7)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _career_years(self) -> float:
        if not self.timeline:
            return 3.0                   # fallback
        starts = [e.start for e in self.timeline if hasattr(e, "start") and e.start]
        if not starts:
            return 3.0
        earliest = min(starts)
        if earliest.tzinfo is None:
            earliest = earliest.replace(tzinfo=timezone.utc)
        return max((self.ref_date - earliest).days / 365.25, 0.1)

    def _weekly_commit_counts(self, timestamps: List[datetime]) -> np.ndarray:
        if not timestamps:
            return np.array([0])
        t_min, t_max = timestamps[0], timestamps[-1]
        total_weeks = max(int((t_max - t_min).days / 7) + 1, 1)
        counts = np.zeros(total_weeks)
        for ts in timestamps:
            week_idx = min(int((ts - t_min).days / 7), total_weeks - 1)
            counts[week_idx] += 1
        return counts

    def _detect_burst_silence(self, timestamps: List[datetime]) -> float:
        """
        Returns a [0,1] score reflecting how much the commit history looks like
        short bursts followed by long silences.
        """
        if len(timestamps) < 5:
            return 0.0

        window = timedelta(days=self.BURST_WINDOW_DAYS)
        silence_threshold = timedelta(days=90)
        max_burst_score = 0.0

        for i, t_start in enumerate(timestamps):
            # Count commits in window
            burst_commits = [t for t in timestamps if t_start <= t <= t_start + window]
            if len(burst_commits) < self.BURST_COMMIT_THRESHOLD:
                continue

            # Check for silence after the burst
            t_burst_end = max(burst_commits)
            subsequent = [t for t in timestamps if t > t_burst_end]
            if not subsequent:
                silence_after = (self.ref_date - t_burst_end).days
            else:
                silence_after = (min(subsequent) - t_burst_end).days

            if silence_after >= silence_threshold.days:
                # Score proportional to burst density and subsequent silence
                density_score = min(len(burst_commits) / 60.0, 1.0)
                silence_score = min(silence_after / 365.0, 1.0)
                burst_score = 0.5 * density_score + 0.5 * silence_score
                max_burst_score = max(max_burst_score, burst_score)

        return max_burst_score


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _linear_ramp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Linear interpolation clamped to [y0, y1]."""
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return _clip(y0 + t * (y1 - y0), min(y0, y1), max(y0, y1))


def _gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient ∈ [0, 1]. 0 = perfect equality, 1 = maximum inequality."""
    if values.sum() == 0:
        return 0.0
    values = np.sort(np.abs(values))
    n = len(values)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) - (n + 1) * values.sum()) / (n * values.sum()))