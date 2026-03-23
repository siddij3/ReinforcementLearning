import numpy as np
import re
from typing import List
from timeline_coherence import TimelineEntry

SENIORITY_LEVELS = {
    "intern": 0, "junior": 1, "associate": 1, "engineer": 2,
    "senior engineer": 3, "senior": 3, "lead": 4, "staff": 5,
    "principal": 6, "director": 7, "vp": 8, "head of": 7,
}

def _seniority_level(title: str) -> int:
    title_lower = title.lower()
    best = 2  # default: generic engineer
    for key, level in sorted(SENIORITY_LEVELS.items(), key=lambda x: -x[1]):
        if key in title_lower:
            best = level
            break
    return best


class CareerProgressionSmoothnessScorer:
    """
    Detects unnaturally smooth career arcs — a hallmark of
    LLM-generated career histories.

    Genuine careers:  non-monotonic seniority, tool specialization
                      that narrows then broadens, role changes with friction
    Fraudulent:       monotonic seniority, every role adds tools, every
                      role has higher metrics, no lateral moves
    """

    def score(self, timeline: List[TimelineEntry]) -> dict:
        jobs = sorted(
            [e for e in timeline if e.entry_type == "job"],
            key=lambda e: e.start
        )
        if len(jobs) < 3:
            return {"career_smoothness_fraud_score": 0.4,
                    "verdict": "too few roles to detect smoothness artifacts"}

        flags  = []
        scores = []

        # ── 1. Seniority monotonicity ─────────────────────────────
        levels = [_seniority_level(j.title) for j in jobs]
        diffs  = [levels[i+1] - levels[i] for i in range(len(levels)-1)]
        monotonic_fraction = sum(1 for d in diffs if d >= 0) / len(diffs)
        if monotonic_fraction == 1.0:
            scores.append(0.60)
            flags.append({
                "type": "monotonic_seniority",
                "detail": "every role is same or higher seniority — no lateral moves",
            })
        elif monotonic_fraction > 0.85:
            scores.append(0.35)
            flags.append({
                "type": "near_monotonic_seniority",
                "detail": f"{monotonic_fraction:.0%} of transitions are upward",
            })

        # ── 2. Tool count growth monotonicity ─────────────────────
        skill_counts = [len(j.claimed_skills or []) for j in jobs]
        if len(skill_counts) > 2:
            skill_diffs   = [skill_counts[i+1] - skill_counts[i]
                             for i in range(len(skill_counts)-1)]
            skills_always_grow = all(d >= 0 for d in skill_diffs)
            if skills_always_grow:
                scores.append(0.40)
                flags.append({
                    "type": "monotonic_skill_growth",
                    "detail": "skill count increases with every role — no specialization dips",
                })

        # ── 3. Metric inflation monotonicity ──────────────────────
        # Extract numeric claims from role descriptions and check if
        # they monotonically increase (every role better than last)
        metric_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(%|x\b|ms\b)')
        role_metrics   = []
        for job in jobs:
            desc    = " ".join(job.claimed_skills or [])
            metrics = [float(m[0]) for m in metric_pattern.findall(desc)]
            role_metrics.append(np.mean(metrics) if metrics else None)

        valid_metrics = [(i, m) for i, m in enumerate(role_metrics) if m is not None]
        if len(valid_metrics) >= 3:
            vals      = [m for _, m in valid_metrics]
            mon_frac  = sum(1 for i in range(len(vals)-1) if vals[i+1] >= vals[i]) / (len(vals)-1)
            if mon_frac > 0.9:
                scores.append(0.30)
                flags.append({
                    "type": "escalating_metrics",
                    "detail": "numeric claims strictly increase across roles — suspiciously linear",
                })

        # ── 4. Title-skill coherence across roles ─────────────────
        # A genuine senior engineer who moved to a startup may have
        # FEWER listed skills (less time, more focus). Check for this.
        startup_keywords = ["startup", "early-stage", "seed", "series a", "founding"]
        for job in jobs:
            is_startup = any(kw in (job.org or "").lower() for kw in startup_keywords)
            skill_count = len(job.claimed_skills or [])
            if is_startup and skill_count > 15:
                scores.append(0.20)
                flags.append({
                    "type": "startup_skill_inflation",
                    "detail": f"startup role '{job.title}' claims {skill_count} skills — implausible for early-stage",
                })

        final = float(np.clip(np.mean(scores) if scores else 0.1, 0.0, 1.0))
        return {
            "career_smoothness_fraud_score": round(final, 4),
        }

