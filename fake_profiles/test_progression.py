"""

# Faker

# sdv (Synthetic Data Vault) lets you fit a multivariate
#  model to whatever ground-truth distribution you 

career_progression_profiles.py

Synthetic genuine and fabricated career timelines for testing
CareerProgressionSmoothnessScorer across multiple domains.

Each profile is designed to trigger (or deliberately avoid triggering)
specific flags:
  - monotonic_seniority       : every role same/higher level
  - monotonic_skill_growth    : skill count grows every role
  - escalating_metrics        : numeric claims strictly increase
  - startup_skill_inflation   : startup role with 15+ skills

Profile roster
──────────────
Genuine (5):
  G1 — Software engineer, lateral move into startup, dip in seniority
  G2 — Financial analyst, moved to smaller firm (step back), then recovered
  G3 — Marketing manager, went from manager to IC at a new company
  G4 — Data scientist, specialization narrowing arc (fewer tools per role)
  G5 — Product manager, long stint → short stint → long stint (messy)

Fabricated (5):
  F1 — Software, perfect monotonic climb, tool list grows every role
  F2 — Finance, every role has higher AUM/deal size, clean ladder
  F3 — Marketing, CAC/conversion metrics strictly improve every role
  F4 — Data scientist, every role adds new frameworks, no setbacks
  F5 — Multi-domain, implausibly complete skill sets at every level
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from ..features.timeline_coherence import TimelineEntry


def dt(year: int, month: int = 1) -> datetime:
    return datetime(year, month, 1)


# ─────────────────────────────────────────────────────────────────────────────
# GENUINE PROFILES
# ─────────────────────────────────────────────────────────────────────────────

def genuine_g1_software_lateral_move() -> dict:
    """
    Software engineer who took a lateral/downward move to join an early
    startup, then recovered. Natural arc: Senior → (Senior at startup but
    with narrower scope) → Staff. Tool count DROPS at startup role because
    they were wearing fewer hats and going deeper.
    Flags expected: NONE
    """
    return {
        "label":  "G1 — Software, lateral startup move",
        "domain": "software",
        "timeline": [
            TimelineEntry(
                entry_type="job", org="Stripe", title="Software Engineer",
                start=dt(2016, 3), end=dt(2018, 8),
                claimed_skills=[
                    "Python", "Go", "PostgreSQL", "Redis",
                    "REST API design", "gRPC", "internal payment SDK",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="Lyft", title="Senior Software Engineer",
                start=dt(2018, 9), end=dt(2021, 2),
                claimed_skills=[
                    "Python", "Kafka", "Flink", "Presto",
                    "distributed tracing", "on-call rotation",
                    "service mesh", "Envoy",
                ]
            ),
            # Lateral move — same title, smaller scope, FEWER skills listed
            TimelineEntry(
                entry_type="job", org="Ditto (Series A)", title="Senior Software Engineer",
                start=dt(2021, 3), end=dt(2022, 11),
                claimed_skills=[
                    "Python", "Postgres", "AWS Lambda",
                    "product thinking", "wore many hats",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="Anthropic", title="Staff Software Engineer",
                start=dt(2022, 12), end=None,
                claimed_skills=[
                    "Python", "Rust", "distributed training infra",
                    "CUDA", "systems design", "technical leadership",
                    "cross-team coordination",
                ]
            ),
        ]
    }


def genuine_g2_finance_step_back() -> dict:
    """
    Financial analyst who left a large bank for a boutique (step back in
    prestige/comp), then moved to a VC-backed fintech as Director.
    Non-monotonic: Associate → Analyst (lateral) → VP → Director.
    Deal sizes and AUM do NOT strictly increase.
    Flags expected: NONE
    """
    return {
        "label":  "G2 — Finance, deliberate step back then recovery",
        "domain": "finance",
        "timeline": [
            TimelineEntry(
                entry_type="job", org="Goldman Sachs", title="Analyst",
                start=dt(2015, 7), end=dt(2017, 9),
                claimed_skills=[
                    "DCF modelling", "comparable company analysis",
                    "pitchbook preparation", "Excel", "Bloomberg terminal",
                    "100-hour weeks", "M&A deal support",
                ]
            ),
            # Step back: left Goldman for boutique (smaller firm, similar title)
            TimelineEntry(
                entry_type="job", org="Lazard (restructuring)", title="Analyst",
                start=dt(2017, 10), end=dt(2019, 6),
                claimed_skills=[
                    "restructuring modelling", "distressed debt", "creditor negotiations",
                    "plan of reorganisation", "Section 363 sales",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="Andreessen Horowitz", title="VP Finance",
                start=dt(2019, 7), end=dt(2022, 3),
                claimed_skills=[
                    "portfolio company support", "FP&A", "board reporting",
                    "LP reporting", "fund accounting", "SQL", "Looker",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="Brex", title="Director of Finance",
                start=dt(2022, 4), end=None,
                claimed_skills=[
                    "FP&A", "revenue modelling", "headcount planning",
                    "three-statement model", "variance analysis", "Workday Adaptive",
                ]
            ),
        ]
    }


def genuine_g3_marketing_manager_to_ic() -> dict:
    """
    Marketing manager who went from managing a team back to individual
    contributor at a new company (took a pay cut for equity). Seniority
    goes: Manager → Senior Manager → Senior Specialist (IC step down).
    Flags expected: NONE (non-monotonic seniority)
    """
    return {
        "label":  "G3 — Marketing, manager to IC step-down",
        "domain": "marketing",
        "timeline": [
            TimelineEntry(
                entry_type="job", org="HubSpot", title="Marketing Manager",
                start=dt(2017, 4), end=dt(2019, 11),
                claimed_skills=[
                    "paid search", "Google Ads", "Meta Ads",
                    "email nurture sequences", "HubSpot", "Salesforce",
                    "team management (2 direct reports)",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="Shopify", title="Senior Marketing Manager",
                start=dt(2019, 12), end=dt(2022, 5),
                claimed_skills=[
                    "lifecycle marketing", "Braze", "Segment", "dbt",
                    "cohort analysis", "LTV modelling",
                    "managed $4M annual paid budget",
                ]
            ),
            # IC step-down — joined early stage company for equity
            TimelineEntry(
                entry_type="job", org="Cortex (Seed)", title="Senior Growth Specialist",
                start=dt(2022, 6), end=dt(2023, 8),
                claimed_skills=[
                    "growth experiments", "A/B testing", "Amplitude",
                    "no team to manage — hands-on execution",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="Rippling", title="Growth Marketing Lead",
                start=dt(2023, 9), end=None,
                claimed_skills=[
                    "demand generation", "ABM", "Marketo", "Salesforce",
                    "cross-channel attribution", "budget ownership $6M",
                ]
            ),
        ]
    }


def genuine_g4_data_scientist_specialization() -> dict:
    """
    Data scientist whose skill set NARROWS over time as they specialize.
    Early roles: broad generalist. Later roles: deep NLP specialist.
    Skill count goes 12 → 9 → 6 → 7 (non-monotonic).
    Flags expected: NONE
    """
    return {
        "label":  "G4 — Data scientist, specialization narrows then deepens",
        "domain": "data science",
        "timeline": [
            TimelineEntry(
                entry_type="job", org="Accenture", title="Data Analyst",
                start=dt(2016, 6), end=dt(2018, 3),
                claimed_skills=[
                    "Python", "R", "SQL", "Tableau", "Excel", "PowerPoint",
                    "logistic regression", "A/B testing", "client presentations",
                    "Hadoop", "Hive", "SAS",  # broad generalist
                ]
            ),
            TimelineEntry(
                entry_type="job", org="Spotify", title="Data Scientist",
                start=dt(2018, 4), end=dt(2020, 9),
                claimed_skills=[
                    "Python", "Spark", "causal inference", "recommendation systems",
                    "experiment design", "XGBoost", "SHAP", "Luigi", "BigQuery",
                    # slightly fewer — starting to focus
                ]
            ),
            # Specialization: moved into NLP specifically, skill list SHRINKS
            TimelineEntry(
                entry_type="job", org="Cohere", title="Senior Research Scientist",
                start=dt(2020, 10), end=dt(2023, 2),
                claimed_skills=[
                    "transformer fine-tuning", "RLHF", "evaluation frameworks",
                    "PyTorch", "academic writing", "internal research agenda",
                    # narrower — deep NLP, dropped SQL/Tableau/Hadoop etc.
                ]
            ),
            TimelineEntry(
                entry_type="job", org="OpenAI", title="Research Scientist",
                start=dt(2023, 3), end=None,
                claimed_skills=[
                    "RLHF", "reward modelling", "constitutional AI",
                    "large-scale distributed training", "safety evals",
                    "red-teaming", "mechanistic interpretability",
                ]
            ),
        ]
    }


def genuine_g5_product_manager_messy() -> dict:
    """
    Product manager with a long tenure, then a short failed startup stint,
    then back to a large company. The short stint is a career 'dip'.
    Skill count doesn't monotonically grow. Metrics don't strictly improve
    (the startup had bad metrics — that's why it failed).
    Flags expected: NONE
    """
    return {
        "label":  "G5 — PM, long-short-long with failed startup",
        "domain": "product",
        "timeline": [
            TimelineEntry(
                entry_type="job", org="Microsoft", title="Product Manager",
                start=dt(2014, 8), end=dt(2018, 11),
                claimed_skills=[
                    "roadmap planning", "stakeholder alignment", "user research",
                    "SQL", "A/B testing", "PRDs", "Azure DevOps",
                    "shipped 3 features to 40M users",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="Microsoft", title="Senior Product Manager",
                start=dt(2018, 12), end=dt(2021, 5),
                claimed_skills=[
                    "growth strategy", "monetisation", "pricing analysis",
                    "cross-functional leadership", "OKR facilitation",
                    "enterprise customer interviews",
                ]
            ),
            # Short failed startup — bad metrics, honest about it
            TimelineEntry(
                entry_type="job", org="Nomad (failed, Seed)", title="Head of Product",
                start=dt(2021, 6), end=dt(2022, 4),
                claimed_skills=[
                    "0-to-1 product work", "wearing all hats",
                    "we didn't find product-market fit",
                    "company shut down after 10 months",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="Figma", title="Senior Product Manager",
                start=dt(2022, 5), end=None,
                claimed_skills=[
                    "developer tools product", "plugin ecosystem",
                    "community-led growth", "Figma API surface",
                    "design-engineer collaboration", "north star metrics",
                ]
            ),
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# FABRICATED PROFILES
# ─────────────────────────────────────────────────────────────────────────────

def fabricated_f1_software_perfect_ladder() -> dict:
    """
    Perfect monotonic climb: Intern → Junior → Engineer → Senior → Staff.
    Skill count grows every single role. Metrics (latency, throughput)
    strictly improve. All roles at prestigious companies.
    Flags expected: monotonic_seniority, monotonic_skill_growth, escalating_metrics
    """
    return {
        "label":  "F1 — Software, perfect monotonic ladder",
        "domain": "software",
        "timeline": [
            TimelineEntry(
                entry_type="job", org="Google", title="Software Engineering Intern",
                start=dt(2016, 6), end=dt(2016, 9),
                claimed_skills=[
                    "Python", "Java", "SQL", "improved intern project latency by 10%",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="Google", title="Junior Software Engineer",
                start=dt(2017, 7), end=dt(2019, 6),
                claimed_skills=[
                    "Python", "Java", "SQL", "Kubernetes", "gRPC",
                    "reduced service latency by 25%", "on-call experience",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="Meta", title="Software Engineer",
                start=dt(2019, 7), end=dt(2021, 6),
                claimed_skills=[
                    "Python", "Java", "C++", "SQL", "Kubernetes", "gRPC",
                    "Kafka", "Spark", "reduced latency by 40%",
                    "improved throughput by 3x", "led 2 junior engineers",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="OpenAI", title="Senior Software Engineer",
                start=dt(2021, 7), end=dt(2023, 6),
                claimed_skills=[
                    "Python", "Java", "C++", "Rust", "SQL", "Kubernetes",
                    "gRPC", "Kafka", "Spark", "Ray", "Triton",
                    "reduced latency by 60%", "improved throughput by 5x",
                    "led team of 5", "designed distributed training infra",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="Anthropic", title="Staff Software Engineer",
                start=dt(2023, 7), end=None,
                claimed_skills=[
                    "Python", "Java", "C++", "Rust", "Go", "SQL", "Kubernetes",
                    "gRPC", "Kafka", "Spark", "Ray", "Triton", "CUDA", "JAX",
                    "reduced latency by 80%", "improved throughput by 10x",
                    "led team of 12", "org-wide technical strategy",
                    "patent holder", "published 3 papers",
                ]
            ),
        ]
    }


def fabricated_f2_finance_escalating_deals() -> dict:
    """
    Every role at progressively more prestigious firms with strictly
    larger deal sizes and AUM. No lateral moves, no boutique stints,
    skill list grows every role.
    Flags expected: monotonic_seniority, monotonic_skill_growth, escalating_metrics
    """
    return {
        "label":  "F2 — Finance, escalating deal sizes perfectly",
        "domain": "finance",
        "timeline": [
            TimelineEntry(
                entry_type="job", org="JPMorgan", title="Analyst",
                start=dt(2016, 7), end=dt(2018, 6),
                claimed_skills=[
                    "DCF", "LBO modelling", "Excel", "Bloomberg",
                    "closed $200M M&A deal", "2 year analyst programme",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="Goldman Sachs", title="Associate",
                start=dt(2018, 7), end=dt(2020, 6),
                claimed_skills=[
                    "DCF", "LBO modelling", "merger modelling", "Excel",
                    "Bloomberg", "FactSet", "pitchbook",
                    "closed $500M cross-border M&A", "managed 2 analysts",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="Blackstone", title="Vice President",
                start=dt(2020, 7), end=dt(2022, 6),
                claimed_skills=[
                    "DCF", "LBO", "merger modelling", "portfolio monitoring",
                    "Excel", "Bloomberg", "FactSet", "Pitchbook", "Tableau",
                    "SQL", "led $1.2B buyout transaction",
                    "managed deal team of 6", "investor relations",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="KKR", title="Director",
                start=dt(2022, 7), end=None,
                claimed_skills=[
                    "DCF", "LBO", "merger modelling", "portfolio monitoring",
                    "fund accounting", "LP reporting", "Excel", "Bloomberg",
                    "FactSet", "Pitchbook", "Tableau", "SQL", "Python",
                    "led $3.5B flagship fund deal", "built deal team of 10",
                    "board observer at 4 portfolio companies",
                ]
            ),
        ]
    }


def fabricated_f3_marketing_perfect_metrics() -> dict:
    """
    Every role shows strictly improving marketing metrics. CAC drops,
    ROAS improves, conversion rate improves — every single role.
    Skill list grows perfectly. No budget constraints, no channel failures.
    Flags expected: monotonic_seniority, monotonic_skill_growth, escalating_metrics
    """
    return {
        "label":  "F3 — Marketing, strictly improving metrics",
        "domain": "marketing",
        "timeline": [
            TimelineEntry(
                entry_type="job", org="HubSpot", title="Marketing Coordinator",
                start=dt(2017, 6), end=dt(2019, 5),
                claimed_skills=[
                    "Google Ads", "Meta Ads", "email marketing", "HubSpot",
                    "reduced CAC by 15%", "improved CTR by 20%",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="Shopify", title="Marketing Manager",
                start=dt(2019, 6), end=dt(2021, 5),
                claimed_skills=[
                    "Google Ads", "Meta Ads", "LinkedIn Ads", "email marketing",
                    "HubSpot", "Salesforce", "Tableau", "A/B testing",
                    "reduced CAC by 30%", "improved ROAS by 2x", "improved CTR by 35%",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="Stripe", title="Senior Marketing Manager",
                start=dt(2021, 6), end=dt(2023, 5),
                claimed_skills=[
                    "Google Ads", "Meta Ads", "LinkedIn Ads", "programmatic",
                    "email marketing", "HubSpot", "Salesforce", "Tableau",
                    "Segment", "dbt", "SQL", "A/B testing", "attribution modelling",
                    "reduced CAC by 50%", "improved ROAS by 4x",
                    "improved conversion rate by 45%", "managed $8M budget",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="Figma", title="Head of Growth Marketing",
                start=dt(2023, 6), end=None,
                claimed_skills=[
                    "Google Ads", "Meta Ads", "LinkedIn Ads", "programmatic",
                    "CTV", "podcast advertising", "email marketing", "HubSpot",
                    "Salesforce", "Tableau", "Segment", "dbt", "SQL", "Python",
                    "A/B testing", "MMM", "attribution modelling", "Braze",
                    "reduced CAC by 65%", "improved ROAS by 6x",
                    "improved conversion rate by 60%", "managed $20M budget",
                ]
            ),
        ]
    }


def fabricated_f4_data_scientist_framework_collector() -> dict:
    """
    Data scientist who adds every new ML framework to their profile with
    each role. Skill count grows: 6 → 10 → 15 → 22.
    No specialization, no narrowing. Perfect breadth at every stage.
    Flags expected: monotonic_seniority, monotonic_skill_growth
    """
    return {
        "label":  "F4 — Data scientist, framework collector",
        "domain": "data science",
        "timeline": [
            TimelineEntry(
                entry_type="job", org="IBM", title="Junior Data Scientist",
                start=dt(2018, 7), end=dt(2020, 6),
                claimed_skills=[
                    "Python", "R", "SQL", "scikit-learn",
                    "pandas", "Tableau",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="Netflix", title="Data Scientist",
                start=dt(2020, 7), end=dt(2022, 6),
                claimed_skills=[
                    "Python", "R", "SQL", "scikit-learn", "pandas",
                    "Tableau", "PyTorch", "TensorFlow", "XGBoost",
                    "Spark", "Airflow",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="Waymo", title="Senior Data Scientist",
                start=dt(2022, 7), end=dt(2023, 12),
                claimed_skills=[
                    "Python", "R", "SQL", "scikit-learn", "pandas",
                    "Tableau", "PyTorch", "TensorFlow", "XGBoost",
                    "Spark", "Airflow", "JAX", "MLflow", "dbt",
                    "Kubeflow", "Ray", "ONNX",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="Google DeepMind", title="Staff Data Scientist",
                start=dt(2024, 1), end=None,
                claimed_skills=[
                    "Python", "R", "SQL", "scikit-learn", "pandas",
                    "Tableau", "PyTorch", "TensorFlow", "XGBoost", "LightGBM",
                    "CatBoost", "Spark", "Airflow", "JAX", "MLflow", "dbt",
                    "Kubeflow", "Ray", "ONNX", "Triton", "LangChain",
                    "vector databases", "Pinecone", "Weaviate",
                ]
            ),
        ]
    }


def fabricated_f5_multi_domain_implausible() -> dict:
    """
    Multi-domain 'expert' — has been simultaneously expert in software,
    finance, and marketing. Every role is at a top-tier company.
    Every transition is upward. Skill counts: 8 → 14 → 20 → 28.
    Flags expected: monotonic_seniority, monotonic_skill_growth
    """
    return {
        "label":  "F5 — Multi-domain, implausibly complete",
        "domain": "multi",
        "timeline": [
            TimelineEntry(
                entry_type="job", org="McKinsey", title="Associate",
                start=dt(2016, 9), end=dt(2018, 8),
                claimed_skills=[
                    "financial modelling", "market sizing", "Excel", "PowerPoint",
                    "Python", "SQL", "client management", "strategy consulting",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="Goldman Sachs", title="Vice President",
                start=dt(2018, 9), end=dt(2020, 8),
                claimed_skills=[
                    "DCF", "LBO", "M&A", "Excel", "Bloomberg", "Python",
                    "SQL", "financial modelling", "deal origination",
                    "stakeholder management", "team leadership",
                    "closed $800M deal", "managed 4 analysts",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="Airbnb", title="Director of Product",
                start=dt(2020, 9), end=dt(2022, 8),
                claimed_skills=[
                    "product strategy", "roadmap", "OKRs", "user research",
                    "A/B testing", "SQL", "Python", "Tableau", "Looker",
                    "Figma", "Jira", "stakeholder alignment",
                    "P&L ownership", "managed 8 PMs", "cross-functional leadership",
                    "grew GMV by 40%", "launched in 12 new markets",
                    "launched 5 major features",
                ]
            ),
            TimelineEntry(
                entry_type="job", org="OpenAI", title="VP of Strategy",
                start=dt(2022, 9), end=None,
                claimed_skills=[
                    "corporate strategy", "M&A", "partnership development",
                    "financial modelling", "DCF", "LBO", "Excel", "Bloomberg",
                    "Python", "SQL", "machine learning", "LLM deployment",
                    "product strategy", "roadmap", "OKRs", "user research",
                    "A/B testing", "Tableau", "Figma", "Jira",
                    "P&L ownership", "managed 20 person team",
                    "closed $2B strategic partnership",
                    "led Series D fundraise", "board presentations",
                    "built 0-to-1 enterprise product",
                ]
            ),
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    from career_smoothness import CareerProgressionSmoothnessScorer

    scorer = CareerProgressionSmoothnessScorer()

    genuine_profiles    = [
        genuine_g1_software_lateral_move(),
        genuine_g2_finance_step_back(),
        genuine_g3_marketing_manager_to_ic(),
        genuine_g4_data_scientist_specialization(),
        genuine_g5_product_manager_messy(),
    ]
    fabricated_profiles = [
        fabricated_f1_software_perfect_ladder(),
        fabricated_f2_finance_escalating_deals(),
        fabricated_f3_marketing_perfect_metrics(),
        fabricated_f4_data_scientist_framework_collector(),
        fabricated_f5_multi_domain_implausible(),
    ]

    print(f"\n{'═'*66}")
    print("  GENUINE PROFILES  (expected: low fraud score, no flags)")
    print(f"{'═'*66}")
    for profile in genuine_profiles:
        result = scorer.score(profile["timeline"])
        _print_result(profile, result)

    print(f"\n{'═'*66}")
    print("  FABRICATED PROFILES  (expected: high fraud score, multiple flags)")
    print(f"{'═'*66}")
    for profile in fabricated_profiles:
        result = scorer.score(profile["timeline"])
        _print_result(profile, result)


def _print_result(profile: dict, result: dict):
    label        = profile["label"]
    score        = result["career_smoothness_fraud_score"]
    flags        = result.get("flags", [])
    levels       = result.get("seniority_levels", [])
    skill_counts = [len(e.claimed_skills) for e in profile["timeline"]
                    if e.entry_type == "job"]
    verdict      = result.get("verdict", "")

    flag_types   = [f["type"] for f in flags]

    bar_len = int(score * 30)
    bar     = "█" * bar_len + "░" * (30 - bar_len)

    print(f"\n  {label}")
    print(f"  fraud signal │{bar}│ {score:.3f}")
    print(f"  seniority    : {levels}")
    print(f"  skill counts : {skill_counts}")
    if flag_types:
        print(f"  flags        : {flag_types}")
    else:
        print(f"  flags        : none")
    print(f"  verdict      : {verdict}")


if __name__ == "__main__":
    run_all()