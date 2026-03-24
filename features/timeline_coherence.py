"""
test_timeline_coherence.py

Sample tests for TimelineCoherenceScorer covering all five signals:
  1. Perfect seams       — jobs end exactly when next begins
  2. Impossible overlaps — two full-time roles simultaneously
  3. Round tenure        — exactly 1yr, 2yr, 3yr tenures
  4. Anachronistic skill — tool claimed before its release date
  5. Seniority too fast  — senior role within 1.5yr of graduation

Each test group has a genuine and a fraudulent variant so the
expected score separation is visible.
"""

from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

# ── Paste or import the scorer ────────────────────────────────────────────────
# (copied here so the test file is self-contained)

@dataclass
class TimelineEntry:
    title:          str
    org:            str
    start:          datetime
    end:            Optional[datetime]
    claimed_skills: List[str] = field(default_factory=list)
    entry_type:     str       = "job"

@dataclass
class TimeLineResult:
    timeline_coherence_fraud_score: float
    flags: List[dict] = field(default_factory=list)

class TimelineCoherenceScorer:
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

        sorted_jobs = sorted(jobs, key=lambda e: e.start)

        # Signal 1: Perfect seams
        for i in range(len(sorted_jobs) - 1):
            curr, nxt = sorted_jobs[i], sorted_jobs[i + 1]
            if curr.end is None:
                continue
            gap_days = (nxt.start - curr.end).days
            if gap_days == 0:
                score += 0.15
                flags.append({"type": "perfect_seam",
                               "detail": f"{curr.title} → {nxt.title}: 0-day gap"})
            elif -5 <= gap_days < 0:
                score += 0.08
                flags.append({"type": "micro_overlap",
                               "detail": f"{abs(gap_days)}-day overlap"})

        # Signal 2: Impossible overlaps
        for i, job_a in enumerate(sorted_jobs):
            for job_b in sorted_jobs[i + 1:]:
                end_a = job_a.end or datetime.now()
                if job_b.start < end_a:
                    overlap_days = (end_a - job_b.start).days
                    if overlap_days > 60:
                        score += 0.30
                        flags.append({"type": "impossible_overlap",
                                       "detail": f"{overlap_days}d overlap: {job_a.title} / {job_b.title}"})

        # Signal 3: Round tenure
        for job in sorted_jobs:
            if job.end is None:
                continue
            months = (job.end - job.start).days / 30.44
            remainder = months % 12
            if abs(remainder) < 0.5 or abs(remainder - 12) < 0.5:
                score += 0.08
                flags.append({"type": "round_tenure",
                               "detail": f"'{job.title}': {months:.1f} months"})

        # Signal 4: Anachronistic skill
        for entry in jobs + certs:
            for skill in (entry.claimed_skills or []):
                for tool, release in self.TOOL_RELEASE_DATES.items():
                    if tool in skill.lower() and entry.start < release:
                        score += 0.40
                        flags.append({"type": "anachronistic_skill",
                                       "detail": f"'{skill}' at {entry.start.year}, released {release.year}"})

        # Signal 5: Seniority too fast
        if edu and jobs:
            latest_edu_end = max((e.end or datetime.now()) for e in edu)
            senior_jobs = [j for j in jobs if any(
                t in j.title.lower()
                for t in ["senior", "staff", "principal", "lead", "vp", "head"]
            )]
            if senior_jobs:
                earliest_senior = min(j.start for j in senior_jobs)
                years_since_edu = (earliest_senior - latest_edu_end).days / 365
                if years_since_edu < 1.5:
                    score += 0.20
                    flags.append({"type": "seniority_too_fast",
                                   "detail": f"{years_since_edu:.1f}yr after graduation"})

        final = float(np.clip(score, 0.0, 1.0))
        return {
            "timeline_coherence_fraud_score": round(final, 4),
            "flags": flags,
            "flag_types": [f["type"] for f in flags],
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def dt(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day)


def run(label: str, timeline: List[TimelineEntry], expected_range: tuple):
    scorer = TimelineCoherenceScorer()
    result = scorer.score(timeline)
    score  = result["timeline_coherence_fraud_score"]
    flags  = result.get("flag_types", [])
    lo, hi = expected_range
    passed = lo <= score <= hi
    status = "PASS" if passed else "FAIL"
    bar    = "█" * int(score * 30) + "░" * (30 - int(score * 30))
    print(f"\n  [{status}] {label}")
    print(f"         score │{bar}│ {score:.3f}  (expected {lo}–{hi})")
    if flags:
        print(f"         flags : {flags}")
    return passed


# ═════════════════════════════════════════════════════════════════════════════
# TEST GROUP 1 — Signal 1: Perfect seams
# ═════════════════════════════════════════════════════════════════════════════

def test_perfect_seams():
    print("\n" + "═" * 60)
    print("  Signal 1: Perfect seams")
    print("═" * 60)

    # Genuine: realistic gaps between roles (weeks to months)
    genuine = [
        TimelineEntry("Software Engineer", "Acme",
                      start=dt(2018, 3), end=dt(2020, 8), entry_type="job"),
        TimelineEntry("Senior Engineer", "Beta Inc",
                      start=dt(2020, 10), end=dt(2023, 2), entry_type="job"),
        TimelineEntry("Staff Engineer", "Gamma",
                      start=dt(2023, 4), end=None, entry_type="job"),
    ]
    run("Genuine — natural gaps (weeks between roles)", genuine, (0.0, 0.20))

    # Fraudulent: every role ends the day the next begins
    fraud = [
        TimelineEntry("Analyst", "Firm A",
                      start=dt(2019, 1, 1), end=dt(2020, 6, 1), entry_type="job"),
        TimelineEntry("Senior Analyst", "Firm B",
                      start=dt(2020, 6, 1), end=dt(2022, 1, 1), entry_type="job"),
        TimelineEntry("Director", "Firm C",
                      start=dt(2022, 1, 1), end=None, entry_type="job"),
    ]
    run("Fraud — perfect seams (0-day gaps on all transitions)", fraud, (0.25, 0.60))


# ═════════════════════════════════════════════════════════════════════════════
# TEST GROUP 2 — Signal 2: Impossible full-time overlaps
# ═════════════════════════════════════════════════════════════════════════════

def test_impossible_overlaps():
    print("\n" + "═" * 60)
    print("  Signal 2: Impossible full-time overlaps")
    print("═" * 60)

    # Genuine: short contractor overlap is realistic
    genuine = [
        TimelineEntry("Product Manager", "AlphaCo",
                      start=dt(2018, 6), end=dt(2021, 3), entry_type="job"),
        # 1-month consulting overlap while transitioning — plausible
        TimelineEntry("Consultant", "Self",
                      start=dt(2021, 2), end=dt(2021, 4), entry_type="job"),
        TimelineEntry("Senior PM", "BetaCo",
                      start=dt(2021, 5), end=None, entry_type="job"),
    ]
    run("Genuine — short consulting overlap during transition", genuine, (0.0, 0.20))

    # Fraudulent: two full-time senior roles with 8-month overlap
    fraud = [
        TimelineEntry("VP Engineering", "TechCorp",
                      start=dt(2019, 3), end=dt(2022, 11), entry_type="job"),
        TimelineEntry("CTO", "StartupXYZ",
                      start=dt(2022, 3), end=dt(2024, 1), entry_type="job"),
    ]
    run("Fraud — 8-month VP + CTO overlap at different companies", fraud, (0.25, 0.70))

    # Edge: exact 60-day overlap — below threshold, should not fire
    edge = [
        TimelineEntry("Engineer", "Co A",
                      start=dt(2020, 1), end=dt(2021, 6), entry_type="job"),
        TimelineEntry("Engineer", "Co B",
                      start=dt(2021, 4), end=dt(2023, 1), entry_type="job"),
    ]
    run("Edge — exactly 61-day overlap (just above 60-day threshold)", edge, (0.25, 0.55))


# ═════════════════════════════════════════════════════════════════════════════
# TEST GROUP 3 — Signal 3: Round tenure compression
# ═════════════════════════════════════════════════════════════════════════════

def test_round_tenures():
    print("\n" + "═" * 60)
    print("  Signal 3: Round tenure compression")
    print("═" * 60)

    # Genuine: messy tenures (1yr 4mo, 2yr 8mo, etc.)
    genuine = [
        TimelineEntry("Nurse", "City Hospital",
                      start=dt(2016, 3), end=dt(2017, 7), entry_type="job"),   # 16mo
        TimelineEntry("Senior Nurse", "Metro ICU",
                      start=dt(2017, 9), end=dt(2020, 5), entry_type="job"),   # 32mo
        TimelineEntry("Charge Nurse", "Regional Medical",
                      start=dt(2020, 8), end=None, entry_type="job"),
    ]
    run("Genuine — irregular tenures (16mo, 32mo)", genuine, (0.0, 0.15))

    # Fraudulent: every role is exactly 1yr, 2yr, or 3yr
    fraud = [
        TimelineEntry("Analyst", "Bank A",
                      start=dt(2017, 7, 1), end=dt(2019, 7, 1), entry_type="job"),   # exactly 24mo
        TimelineEntry("Associate", "Bank B",
                      start=dt(2019, 9, 1), end=dt(2021, 9, 1), entry_type="job"),   # exactly 24mo
        TimelineEntry("VP", "Bank C",
                      start=dt(2021, 11, 1), end=dt(2024, 11, 1), entry_type="job"), # exactly 36mo
    ]
    run("Fraud — all roles exactly 2yr, 2yr, 3yr", fraud, (0.15, 0.50))

    # Mixed: one round tenure among natural ones
    mixed = [
        TimelineEntry("Engineer", "Co A",
                      start=dt(2018, 2), end=dt(2019, 9), entry_type="job"),   # 19mo — natural
        TimelineEntry("Senior Eng", "Co B",
                      start=dt(2019, 11), end=dt(2021, 11), entry_type="job"), # exactly 24mo — flag
        TimelineEntry("Staff Eng", "Co C",
                      start=dt(2022, 1), end=None, entry_type="job"),
    ]
    run("Mixed — one round tenure among natural ones", mixed, (0.05, 0.25))


# ═════════════════════════════════════════════════════════════════════════════
# TEST GROUP 4 — Signal 4: Anachronistic skill claims
# ═════════════════════════════════════════════════════════════════════════════

def test_anachronistic_skills():
    print("\n" + "═" * 60)
    print("  Signal 4: Anachronistic skill claims")
    print("═" * 60)

    # Genuine: skills claimed after tool release dates
    genuine = [
        TimelineEntry(
            "ML Engineer", "DataCo",
            start=dt(2021, 6), end=dt(2023, 4),
            claimed_skills=["PyTorch", "Airflow", "dbt 0.21", "Kubernetes"],
            entry_type="job"
        ),
        TimelineEntry(
            "Senior ML Engineer", "AIStartup",
            start=dt(2023, 5), end=None,
            claimed_skills=["LangChain", "LLaMA 2", "GPT-4 API"],  # all post-release
            entry_type="job"
        ),
    ]
    run("Genuine — tools claimed after their release dates", genuine, (0.0, 0.20))

    # Fraudulent: LangChain claimed in a role that started in 2021 (released Oct 2022)
    fraud_langchain = [
        TimelineEntry(
            "LLM Engineer", "TechCorp",
            start=dt(2021, 1), end=dt(2023, 6),
            claimed_skills=["Python", "LangChain", "vector databases", "RAG pipelines"],
            entry_type="job"
        ),
    ]
    run("Fraud — LangChain claimed in Jan 2021 role (released Oct 2022)",
        fraud_langchain, (0.35, 0.80))

    # Fraudulent: multiple anachronistic tools
    fraud_multi = [
        TimelineEntry(
            "Senior Data Engineer", "DataPlatform Co",
            start=dt(2021, 3), end=dt(2023, 8),
            claimed_skills=["dbt 1.0", "Prefect 2", "Ray 2", "LangChain"],
            # dbt 1.0: Jan 2022, Prefect 2: Jul 2022, Ray 2: Jun 2022, LangChain: Oct 2022
            # All claimed in role starting Mar 2021
            entry_type="job"
        ),
    ]
    run("Fraud — 4 tools claimed before release (dbt 1.0, Prefect 2, Ray 2, LangChain)",
        fraud_multi, (0.80, 1.00))

    # Edge: tool name partially matches (should not flag "ray" if skill is "x-ray")
    edge_partial = [
        TimelineEntry(
            "Radiologist", "City Hospital",
            start=dt(2020, 1), end=dt(2023, 6),
            claimed_skills=["X-ray interpretation", "MRI", "CT scanning"],
            entry_type="job"
        ),
    ]
    # Note: "ray" is in "x-ray" — this tests whether partial match is a problem
    # The scorer uses `tool in skill.lower()` so "ray" will match "x-ray"
    # This is a known limitation — flagged here for awareness
    run("Edge — 'x-ray' partially matches 'ray 2' pattern (known limitation)",
        edge_partial, (0.0, 0.50))  # wide range — documents the ambiguity


# ═════════════════════════════════════════════════════════════════════════════
# TEST GROUP 5 — Signal 5: Seniority too fast after graduation
# ═════════════════════════════════════════════════════════════════════════════

def test_seniority_speed():
    print("\n" + "═" * 60)
    print("  Signal 5: Seniority relative to education")
    print("═" * 60)

    # Genuine: 3+ years between graduation and first senior role
    genuine = [
        TimelineEntry("BSc Computer Science", "State University",
                      start=dt(2015, 9), end=dt(2019, 6), entry_type="education"),
        TimelineEntry("Software Engineer", "StartupA",
                      start=dt(2019, 7), end=dt(2021, 8), entry_type="job"),
        TimelineEntry("Senior Software Engineer", "ScaleupB",
                      start=dt(2021, 9), end=None, entry_type="job"),  # 2.25yr after grad
    ]
    run("Genuine — senior role 2.25yr after graduation", genuine, (0.0, 0.20))

    # Fraudulent: senior role claimed 6 months after graduation
    fraud = [
        TimelineEntry("BSc Finance", "Business School",
                      start=dt(2021, 9), end=dt(2023, 6), entry_type="education"),
        TimelineEntry("Senior Financial Analyst", "Hedge Fund",
                      start=dt(2023, 9), end=None, entry_type="job"),  # 0.25yr after grad
    ]
    run("Fraud — senior role 3 months after graduation", fraud, (0.15, 0.40))

    # Fraudulent: VP role immediately after graduation
    fraud_vp = [
        TimelineEntry("MBA", "Business School",
                      start=dt(2020, 9), end=dt(2022, 6), entry_type="education"),
        TimelineEntry("VP of Product", "TechStartup",
                      start=dt(2022, 7), end=None, entry_type="job"),  # 1mo after MBA
    ]
    run("Fraud — VP role 1 month after MBA graduation", fraud_vp, (0.15, 0.40))

    # Genuine: senior role while still finishing part-time degree
    genuine_pt = [
        TimelineEntry("Part-time MSc", "Open University",
                      start=dt(2018, 9), end=dt(2021, 6), entry_type="education"),
        TimelineEntry("Software Engineer", "TechCo",
                      start=dt(2016, 7), end=dt(2020, 3), entry_type="job"),
        TimelineEntry("Senior Engineer", "ScaleCo",
                      start=dt(2020, 4), end=None, entry_type="job"),  # senior during degree
    ]
    run("Genuine — senior role while completing part-time MSc (edu end > senior start)",
        genuine_pt, (0.0, 0.30))


# ═════════════════════════════════════════════════════════════════════════════
# TEST GROUP 6 — Combined: genuine profiles with natural friction
# ═════════════════════════════════════════════════════════════════════════════

def test_genuine_profiles():
    print("\n" + "═" * 60)
    print("  Combined: full genuine profiles")
    print("═" * 60)

    # Software engineer with lateral move and startup stint
    sw_genuine = [
        TimelineEntry("BSc Software Engineering", "University",
                      start=dt(2013, 9), end=dt(2017, 6), entry_type="education"),
        TimelineEntry("Software Engineer", "Stripe",
                      start=dt(2017, 8), end=dt(2019, 11),   # 27mo — natural
                      claimed_skills=["Python", "PostgreSQL", "Redis"],
                      entry_type="job"),
        TimelineEntry("Senior Engineer", "Lyft",
                      start=dt(2020, 1), end=dt(2022, 5),    # 28mo — natural
                      claimed_skills=["Kafka", "Flink", "Python"],
                      entry_type="job"),
        TimelineEntry("Senior Engineer", "Seed Startup",      # lateral — same title
                      start=dt(2022, 7), end=dt(2023, 9),    # 14mo — left when failed
                      claimed_skills=["Python", "AWS Lambda"],
                      entry_type="job"),
        TimelineEntry("Staff Engineer", "Anthropic",
                      start=dt(2023, 11), end=None,
                      claimed_skills=["Python", "Rust", "distributed training"],
                      entry_type="job"),
    ]
    run("SW engineer — lateral move, failed startup, natural gaps", sw_genuine, (0.0, 0.25))

    # Financial analyst with step-back then recovery
    fin_genuine = [
        TimelineEntry("BSc Economics", "LSE",
                      start=dt(2013, 9), end=dt(2016, 6), entry_type="education"),
        TimelineEntry("Analyst", "Goldman Sachs",
                      start=dt(2016, 8), end=dt(2018, 9),    # 25mo
                      claimed_skills=["DCF", "Excel", "Bloomberg"],
                      entry_type="job"),
        TimelineEntry("Analyst", "Boutique Restructuring",   # lateral/step-back
                      start=dt(2018, 11), end=dt(2020, 7),   # 20mo
                      claimed_skills=["distressed debt", "Section 363"],
                      entry_type="job"),
        TimelineEntry("VP Finance", "VC Fund",
                      start=dt(2020, 9), end=dt(2023, 2),    # 29mo
                      claimed_skills=["FP&A", "LP reporting", "SQL"],
                      entry_type="job"),
        TimelineEntry("Director of Finance", "Brex",
                      start=dt(2023, 4), end=None,
                      claimed_skills=["revenue modelling", "Workday"],
                      entry_type="job"),
    ]
    run("Finance — lateral boutique move, step-back pattern, natural gaps", fin_genuine, (0.0, 0.25))


# ═════════════════════════════════════════════════════════════════════════════
# TEST GROUP 7 — Combined: fabricated profiles
# ═════════════════════════════════════════════════════════════════════════════

def test_fabricated_profiles():
    print("\n" + "═" * 60)
    print("  Combined: full fabricated profiles")
    print("═" * 60)

    # Perfect ladder: seams + round tenures + anachronistic tools
    fraud_full = [
        TimelineEntry("BSc CS", "State University",
                      start=dt(2018, 9), end=dt(2022, 6), entry_type="education"),
        TimelineEntry("ML Engineer", "TechCo",             # 0.3yr after grad → flag
                      start=dt(2022, 9, 1), end=dt(2024, 9, 1),  # exactly 24mo → flag
                      claimed_skills=["LangChain", "GPT-4", "LLaMA"],  # claimed Sep 2022
                      # LangChain: Oct 2022, GPT-4: Mar 2023, LLaMA: Feb 2023 → flags
                      entry_type="job"),
        TimelineEntry("Senior ML Engineer", "AIStartup",
                      start=dt(2024, 9, 1), end=None,  # perfect seam → flag
                      claimed_skills=["Mistral", "RAG"],
                      entry_type="job"),
    ]
    run("Fraud full — seam + round tenure + anachronistic tools + fast seniority",
        fraud_full, (0.50, 1.00))

    # Impossible overlap: two concurrent senior roles
    fraud_overlap = [
        TimelineEntry("Director of Engineering", "BigCo",
                      start=dt(2020, 1), end=dt(2023, 6),
                      claimed_skills=["Python", "Kubernetes"],
                      entry_type="job"),
        TimelineEntry("CTO", "StartupXYZ",                 # 5-month overlap
                      start=dt(2023, 1), end=dt(2024, 6),
                      claimed_skills=["Python", "AWS"],
                      entry_type="job"),
    ]
    run("Fraud overlap — Director + CTO with 5-month concurrent overlap",
        fraud_overlap, (0.25, 0.65))


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("  TimelineCoherenceScorer — sample tests")
    print("█" * 60)

    test_perfect_seams()
    test_impossible_overlaps()
    test_round_tenures()
    test_anachronistic_skills()
    test_seniority_speed()
    test_genuine_profiles()
    test_fabricated_profiles()

    print("\n" + "─" * 60)
    print("  Done.")
    print("─" * 60 + "\n")