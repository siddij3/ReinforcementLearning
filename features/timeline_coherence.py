import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class TimelineEntry:
    entry_type:  str          # "job" | "education" | "certification" | "gap"
    title:       str          # role title, degree name, cert name
    org:         str          # employer, university, issuer
    start:       datetime
    end:         Optional[datetime]   # None = present
    claimed_skills: List[str] = None


class TimelineCoherenceScorer:
    """
    Scores how plausible the temporal structure of a candidate's
    career history is. Detects:

      1. Perfect seams   — jobs end exactly when next begins (no gap)
      2. Impossible overlaps — two full-time jobs simultaneously
      3. Credential timing — certs claimed before tools existed
      4. Compression artifacts — multi-year roles with suspiciously
                                  round tenure (exactly 2.0 years)
      5. Education / employment alignment — degree while working is normal;
                                            degree after senior role is odd
    """

    # Tools with known release dates — claimed expertise before release = flag
    TOOL_RELEASE_DATES = {
        "langchain":  datetime(2022, 10, 1),
        "llama":      datetime(2023, 2, 1),
        "gpt-4":      datetime(2023, 3, 1),
        "mistral":    datetime(2023, 9, 1),
        "ray 2":      datetime(2022, 6, 1),
        "feast 0.3":  datetime(2022, 8, 1),
        "dbt 1.0":    datetime(2022, 1, 1),
        "prefect 2":  datetime(2022, 7, 1),
    }

    def score(self, timeline: List[TimelineEntry]) -> dict:
        if len(timeline) < 2:
            return {"timeline_coherence_fraud_score": 0.3,
                    "verdict": "too few entries to analyze"}

        jobs  = [e for e in timeline if e.entry_type == "job"]
        edu   = [e for e in timeline if e.entry_type == "education"]
        certs = [e for e in timeline if e.entry_type == "certification"]

        flags = []
        score = 0.0

        # ── 1. Perfect seam detection ─────────────────────────────
        sorted_jobs = sorted(jobs, key=lambda e: e.start)
        for i in range(len(sorted_jobs) - 1):
            curr, nxt = sorted_jobs[i], sorted_jobs[i + 1]
            if curr.end is None:
                continue
            gap_days = (nxt.start - curr.end).days
            if gap_days == 0:
                score += 0.15
                flags.append({
                    "type": "perfect_seam",
                    "detail": f"{curr.title} ends same day {nxt.title} begins",
                    "weight": 0.15,
                })
            elif -5 <= gap_days < 0:
                # Tiny overlap could be rounding — mild flag
                score += 0.08
                flags.append({
                    "type": "micro_overlap",
                    "detail": f"{abs(gap_days)}-day overlap between {curr.title} and {nxt.title}",
                    "weight": 0.08,
                })

        # ── 2. Impossible full-time overlaps ──────────────────────
        for i, job_a in enumerate(sorted_jobs):
            for job_b in sorted_jobs[i + 1:]:
                end_a = job_a.end or datetime.now()
                if job_b.start < end_a:
                    overlap_days = (end_a - job_b.start).days
                    if overlap_days > 60:   # >2 months overlap = suspicious
                        score += 0.30
                        flags.append({
                            "type": "impossible_overlap",
                            "detail": f"{overlap_days}d overlap: '{job_a.title}' and '{job_b.title}'",
                            "weight": 0.30,
                        })

        # ── 3. Round tenure compression ───────────────────────────
        # Real tenures: 1yr 3mo, 2yr 7mo. LLM tenures: exactly 1yr, 2yr, 3yr
        for job in sorted_jobs:
            if job.end is None:
                continue
            months = (job.end - job.start).days / 30.44
            remainder = months % 12
            if abs(remainder) < 0.5 or abs(remainder - 12) < 0.5:
                score += 0.08
                flags.append({
                    "type": "round_tenure",
                    "detail": f"'{job.title}' tenure suspiciously round: {months:.1f} months",
                    "weight": 0.08,
                })

        # ── 4. Credential timing vs. tool release dates ───────────
        for entry in jobs + certs:
            for skill in (entry.claimed_skills or []):
                skill_lower = skill.lower()
                for tool, release in self.TOOL_RELEASE_DATES.items():
                    if tool in skill_lower and entry.start < release:
                        score += 0.40
                        flags.append({
                            "type": "anachronistic_skill",
                            "detail": f"'{skill}' claimed at {entry.start.year} but released {release.year}",
                            "weight": 0.40,
                        })

        # ── 5. Education / seniority plausibility ─────────────────
        if edu and jobs:
            latest_edu_end = max((e.end or datetime.now()) for e in edu)
            senior_jobs    = [j for j in jobs if any(
                t in j.title.lower() for t in ["senior", "staff", "principal", "lead", "vp", "head"]
            )]
            if senior_jobs:
                earliest_senior = min(j.start for j in senior_jobs)
                years_since_edu = (earliest_senior - latest_edu_end).days / 365
                if years_since_edu < 1.5:
                    score += 0.20
                    flags.append({
                        "type": "seniority_too_fast",
                        "detail": f"Senior role claimed {years_since_edu:.1f}yr after graduation",
                        "weight": 0.20,
                    })

        final = float(np.clip(score, 0.0, 1.0))
        return {
            "timeline_coherence_fraud_score": round(final, 4),
        }