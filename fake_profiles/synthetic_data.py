from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from features import timeline_coherence
import hf_token

hf_token.ensure_hf_environment()

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np

from features.timeline_coherence import TimelineEntry, TimelineCoherenceScorer, TimeLineResult
from features.voice_consistency import ProfileVoiceConsistencyScorer
from features.career_smoothness import CareerProgressionSmoothnessScorer
from features.structural_organization import StructuralOrganizationScorer
from features.operational_specificity import OperationalSpecificityScorer
from features.cross_answer_consistency import CrossAnswerConsistencyScorer
from features.depth_collapse import DepthCollapseDeltaScorer
from features.narrative_causality import NarrativeCausalityScorer
from features.answer_perplexity import AnswerPerplexityScorer
from features.skill_taxonomy import SkillTaxonomyScorer

@dataclass
class SignalProcessor:
    fraud_rate: float = 0.30
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))

    def __post_init__(self) -> None:
        
        self.skill_taxonomy_scorer = None

        self.timeline_scorer = None
        self.voice_scorer = None
        self.career_scorer = None
        self.structure_scorer = None
        self.operational_scorer = None
        self.cross_scorer = None
        self.depth_scorer = None
        self.narrative_scorer = None
        self.answer_perplexity_scorer = None

    def random_profile(self) -> LinkedInSyntheticProfile:
        is_fraud = self.rng.random() < self.fraud_rate
        profile_template = self.rng.choice(ALL_PROFILES)
        jd_text = profile_template.jd_text
        summary = profile_template.summary
        experiences = profile_template.experiences
        skills = profile_template.skills
        timeline = profile_template.timeline
        screening_questions = profile_template.screening_questions
        screening_answers = profile_template.screening_answers
        web_signals = profile_template.web_signals

        return LinkedInSyntheticProfile(
            is_fraud=is_fraud,
            jd_text=jd_text,
            summary=summary,
            experiences=experiences,
            skills=skills,
            timeline=timeline,
            screening_questions=screening_questions,
            screening_answers=screening_answers,
            web_signals=web_signals
        )

    def stage_zero(self, profile: LinkedInSyntheticProfile) -> dict:

        # Stage 0
        self.skill_taxonomy_scorer = SkillTaxonomyScorer(profile_text=profile.profile_text, jd_text=profile.jd_text) if not self.skill_taxonomy_scorer else self.skill_taxonomy_scorer
        self.timeline_scorer =  TimelineCoherenceScorer() if not self.timeline_scorer else self.timeline_scorer
        self.career_scorer = CareerProgressionSmoothnessScorer() if not self.career_scorer else self.career_scorer
        self.voice_scorer = ProfileVoiceConsistencyScorer() if not self.voice_scorer else self.voice_scorer

        skill_taxonomy_result = self.skill_taxonomy_scorer.score()
        skill_taxonomy = {
            "coverage_score": skill_taxonomy_result.coverage_score,
            "idiosyncrasy_score": skill_taxonomy_result.idiosyncrasy_score,
            "semantic_mirror_score": skill_taxonomy_result.semantic_mirror_score,
        }

        print(skill_taxonomy)
        timeline_coherence_result = self.timeline_scorer.score(profile.timeline)
        timeline_coherence = {
            "timeline_coherence_fraud_score": timeline_coherence_result["timeline_coherence_fraud_score"],
        }

        print(skill_taxonomy)
        career_smoothness_result = self.career_scorer.score(profile.timeline)
        career_smoothness = {
            "career_smoothness_fraud_score": career_smoothness_result["career_smoothness_fraud_score"],
        }
        print(career_smoothness)
        voice_score_result = self.voice_scorer.score(profile.sections)
        voice_score = {
            "profile_voice_fraud_score": voice_score_result["profile_voice_fraud_score"],
            "soft_skill_mean_rate": voice_score_result["soft_skill_mean_rate"],
            "soft_skill_cv": voice_score_result["soft_skill_cv"],
            "sentence_uniformity": voice_score_result["sentence_uniformity"],
            "tech_density_cv": voice_score_result["tech_density_cv"],
            "voice_cluster_penalty": voice_score_result["voice_cluster_penalty"]
        }
        print(voice_score)
        return {**skill_taxonomy, **timeline_coherence, **career_smoothness, **voice_score}

    def stage_one(self, profile: LinkedInSyntheticProfile) -> dict:
        # Stage 1
        a1 = profile.screening_answers[0]
        
        self.narrative_scorer = NarrativeCausalityScorer() if not self.narrative_scorer else self.narrative_scorer
        self.answer_perplexity_scorer = AnswerPerplexityScorer() if not self.answer_perplexity_scorer else self.answer_perplexity_scorer

        # q1_op_fraud = self._operational_specificity_fraud(a1, profile.is_fraud)
        
        narrative_result = self.narrative_scorer.score(a1) 
        narrative = {
            "causal_span_score": narrative_result.causal_span_score,
            "result_entity_score": narrative_result.result_entity_score,
            "coherence_score": narrative_result.coherence_score,
        }

        answer_perplexity = self.answer_perplexity_scorer.score(a1)
        asnwer_perplexity_score = {
            "answer_perplexity_score": answer_perplexity["answer_perplexity_score"],
            "raw_perplexity": answer_perplexity["raw_perplexity"],
            "conditioned_perplexity": answer_perplexity["conditioned_perplexity"],
            "mean_log_prob": answer_perplexity["mean_log_prob"],
            "log_prob_cv": answer_perplexity["log_prob_cv"],
            "high_prob_token_fraction": answer_perplexity["high_prob_token_fraction"],
            "low_prob_token_fraction": answer_perplexity["low_prob_token_fraction"],
        }

        # structural = self._structural_over_org(a1, profile.is_fraud)
        # response_div = self._response_divergence(q1, a1, profile.is_fraud)
        # token_burst = self._token_burstiness_fraud(a1_) answer perplexity.

        return {**narrative, **asnwer_perplexity_score}

        # Stage 2
    def stage_two(self, profile: LinkedInSyntheticProfile) -> dict:
        a1 = profile.screening_answers[0]
        q1 = profile.screening_questions[0]
        a2 = profile.screening_answers[1]
        q2 = profile.screening_questions[1]

        self.depth_scorer = DepthCollapseDeltaScorer() if not self.depth_scorer else self.depth_scorer
        
        depth_delta = self.depth_scorer.compute_delta(a1, q1, a2, q2)["depth_collapse_delta"]
        return {"depth_collapse_delta": depth_delta}

        # Stage 3
    def stage_three(self, profile: LinkedInSyntheticProfile) -> dict:

        self.cross_scorer = CrossAnswerConsistencyScorer() if not self.cross_scorer else self.cross_scorer

        cross = self.cross_scorer.score(profile.screening_answers)
        cross_answer_score = {
            "cross_answer_consistency_fraud_score": cross["cross_answer_consistency_fraud_score"],
            "combined_consitency_score": cross["combined_consistency"],
        }
        # q2_op_fraud = self._operational_specificity_fraud(a2, profile.is_fraud)
        # operational_agg = clip(0.5 * q1_op_fraud + 0.5 * q2_op_fraud)

        return cross_answer_score

       
@dataclass
class LinkedInSyntheticProfile:
    is_fraud: bool
    jd_text: str
    summary: str
    experiences: List[Dict[str, str]]
    skills: List[str]
    timeline: List[TimelineEntry]
    screening_questions: List[str]
    screening_answers: List[str]
    web_signals: Dict[str, float]

    @property
    def sections(self) -> Dict[str, str]:
        sections = {"summary": self.summary, "skills": ", ".join(self.skills)}
        for i, exp in enumerate(self.experiences):
            sections[f"role_{i+1}"] = exp["description"]
        return sections

    @property
    def profile_text(self) -> str:
        role_text = "\n".join(
            f"{e['title']} at {e['org']}: {e['description']}" for e in self.experiences
        )
        return f"{self.summary}\n\nSkills: {', '.join(self.skills)}\n\n{role_text}"


# ===========================================================================
# ── PROFILE 1 ── Senior Software Engineer (Legitimate) ─────────────────────
# ===========================================================================
p1 = LinkedInSyntheticProfile(
    is_fraud=False,
    jd_text=(
        "We are looking for a Senior Software Engineer with 5+ years of experience "
        "building distributed backend systems in Python or Go. Experience with Kubernetes, "
        "Postgres, and CI/CD pipelines required. Strong system-design instincts."
    ),
    summary=(
        "Backend engineer with 8 years turning ambiguous requirements into reliable, "
        "observable services. Spent the first half of my career at mid-size SaaS shops "
        "learning to move fast; the second half at a Series C fintech learning to move "
        "carefully. I care about on-call ergonomics as much as feature velocity."
    ),
    experiences=[
        {
            "title": "Senior Software Engineer",
            "org": "Meridian Payments",
            "description": (
                "Owned the real-time ledger service processing $2 B/month in ACH transactions. "
                "Migrated the monolith to a 12-service Kubernetes mesh, cutting p99 latency from "
                "420 ms to 38 ms. Introduced structured logging and distributed tracing (Jaeger), "
                "reducing MTTR from 47 min to 9 min. Mentored two mid-level engineers who both "
                "received promotions within 18 months."
            ),
        },
        {
            "title": "Software Engineer II",
            "org": "Stackline Analytics",
            "description": (
                "Built the data-ingestion pipeline for 600 retail clients, processing 4 M events/day "
                "via Kafka and writing to Redshift. Designed idempotent retry logic that eliminated "
                "a class of duplicate-row bugs costing ~3 engineering days/month to investigate. "
                "Led adoption of pre-commit hooks and mypy across the backend org."
            ),
        },
        {
            "title": "Software Engineer",
            "org": "Waverly Digital",
            "description": (
                "Full-stack feature work on a SaaS CRM (Django + React). Rewrote the PDF report "
                "generator, cutting generation time from 14 s to 1.1 s. On-call rotation for a "
                "300 k-user product."
            ),
        },
    ],
    skills=[
        "Python", "Go", "Kubernetes", "PostgreSQL", "Kafka", "Redis",
        "Terraform", "Jaeger", "Prometheus", "System Design", "CI/CD",
    ],
    timeline=[
        TimelineEntry("Software Engineer",    "Waverly Digital",    datetime(2016, 6, 1),  datetime(2019, 3, 1)),
        TimelineEntry("Software Engineer II", "Stackline Analytics",datetime(2019, 4, 1),  datetime(2021, 8, 1)),
        TimelineEntry("Senior Software Engineer","Meridian Payments",datetime(2021, 9, 1),  None),
    ],
    screening_questions=[
        "Describe a system you designed from scratch and the trade-offs you made.",
        "Tell me about a time you reduced operational toil significantly.",
    ],
    screening_answers=[
        (
            "At Meridian I designed the ledger service from scratch. The core trade-off was "
            "consistency vs. availability: ACH requires exactly-once semantics, so I chose "
            "Postgres with advisory locks over a Kafka-only approach, accepting slightly higher "
            "write latency (~5 ms overhead) in exchange for a simple, auditable transaction log. "
            "I prototyped both approaches in a one-week spike, ran load tests at 3× peak volume, "
            "and presented the results to the team before committing."
        ),
        (
            "On-call at Stackline was burning people out because we had no structured logging—"
            "every incident required grepping raw files. I spent three Fridays adding structured "
            "JSON logs via structlog, wiring them into CloudWatch Insights with saved queries for "
            "the top five alert types. Within six weeks, the median incident investigation dropped "
            "from 35 minutes to 8 minutes, measured across 22 incidents."
        ),
    ],
    web_signals={
        "github_public_repos": 18.0,
        "github_commit_recency_days": 12.0,
        "stackoverflow_reputation": 2340.0,
        "linkedin_connections": 487.0,
        "endorsement_reciprocity": 0.72,
    },
)


# ===========================================================================
# ── PROFILE 2 ── Data Scientist, ML Platform (Legitimate) ──────────────────
# ===========================================================================
p2 = LinkedInSyntheticProfile(
    is_fraud=False,
    jd_text=(
        "ML Engineer / Data Scientist for a growth-stage e-commerce platform. "
        "Must have hands-on experience with recommender systems, A/B testing at scale, "
        "and productionizing models via MLflow or similar. Python, SQL, Spark required."
    ),
    summary=(
        "Data scientist who prefers shipping models over publishing notebooks. "
        "Background in statistics (M.S., University of Waterloo) with 6 years applying "
        "that foundation to real product decisions—primarily in e-commerce and marketplace "
        "ranking. I get uncomfortable when a model can't explain itself."
    ),
    experiences=[
        {
            "title": "Senior Data Scientist",
            "org": "Cartwell Commerce",
            "description": (
                "Led the homepage personalization project: rebuilt a rule-based carousel into a "
                "two-tower neural retrieval model (TensorFlow Recommenders). Lifted add-to-cart "
                "rate +11% in a 90-day A/B test (n = 1.4 M users, p < 0.001). Set up the feature "
                "store in Feast, reducing feature-serving latency from 180 ms to 22 ms. "
                "Owned ML platform reliability SLA (99.9%)."
            ),
        },
        {
            "title": "Data Scientist",
            "org": "Arborview Marketplace",
            "description": (
                "Built fraud propensity model (XGBoost) reducing chargebacks by $1.2 M/year. "
                "Ran 30+ A/B tests, establishing a shared testing framework adopted company-wide. "
                "Mentored two analysts in causal inference methods."
            ),
        },
        {
            "title": "Data Analyst",
            "org": "Syntek Research",
            "description": (
                "Automated weekly reporting suite (Airflow + dbt), saving 12 analyst-hours/week. "
                "Built customer churn logistic regression that informed a $500 k retention "
                "campaign with 18% response rate."
            ),
        },
    ],
    skills=[
        "Python", "SQL", "Spark", "TensorFlow", "XGBoost", "MLflow",
        "Feast", "Airflow", "dbt", "A/B Testing", "Causal Inference", "Statistics",
    ],
    timeline=[
        TimelineEntry("Data Analyst",         "Syntek Research",     datetime(2017, 9, 1),  datetime(2019, 12, 1)),
        TimelineEntry("Data Scientist",        "Arborview Marketplace",datetime(2020, 1, 1), datetime(2022, 5, 1)),
        TimelineEntry("Senior Data Scientist", "Cartwell Commerce",   datetime(2022, 6, 1),  None),
    ],
    screening_questions=[
        "Walk me through how you would build and validate a recommendation model.",
        "How do you ensure a model stays healthy after it goes to production?",
    ],
    screening_answers=[
        (
            "I start with offline baselines: collaborative filtering and a popularity baseline. "
            "For Cartwell's recs, I defined the task as next-click prediction and evaluated with "
            "recall@10 and NDCG on a held-out time split. I built the two-tower architecture "
            "iteratively—first just item embeddings, then adding user context features, measuring "
            "lift at each step to justify added complexity. Before A/B testing, I ran a replay "
            "evaluation against logged traffic to confirm the offline gains translated."
        ),
        (
            "I instrument three layers: data quality (Great Expectations checks on feature "
            "distributions), model performance (daily precision/recall vs. a rolling baseline "
            "window), and business KPIs (revenue per impression). I set paging thresholds at 2σ "
            "degradation and a non-paging drift alert at 1.5σ to catch gradual concept drift "
            "before it becomes an incident. At Cartwell, this caught a feature-store schema "
            "change 11 days before it would have visibly impacted conversion."
        ),
    ],
    web_signals={
        "github_public_repos": 9.0,
        "github_commit_recency_days": 31.0,
        "kaggle_rank_percentile": 0.88,
        "linkedin_connections": 612.0,
        "endorsement_reciprocity": 0.68,
    },
)


# ===========================================================================
# ── PROFILE 3 ── "Full-Stack AI Lead" (FRAUD) ──────────────────────────────
# ===========================================================================
p3 = LinkedInSyntheticProfile(
    is_fraud=True,
    jd_text=(
        "AI Engineering Lead for a Series B healthtech startup. Must have experience "
        "with LLM fine-tuning, RAG pipelines, clinical NLP, and leading a small team. "
        "Python, PyTorch, and cloud (AWS/GCP) required."
    ),
    summary=(
        "Passionate AI leader with expertise in leveraging cutting-edge machine learning "
        "solutions to drive transformative business outcomes. Proven track record of "
        "delivering innovative AI-powered products at scale. Strong communicator who "
        "bridges technical and business stakeholders seamlessly. Results-driven and "
        "committed to excellence in every endeavor."
    ),
    experiences=[
        {
            "title": "AI Engineering Lead",
            "org": "NeuroPath Health",
            "description": (
                "Led AI initiatives leveraging large language models to improve clinical workflows. "
                "Collaborated cross-functionally to deliver impactful solutions. Managed a team "
                "of talented engineers. Drove significant improvements in key metrics. Presented "
                "technical roadmaps to C-suite leadership on a regular basis."
            ),
        },
        {
            "title": "Senior Machine Learning Engineer",
            "org": "Datatron Systems",
            "description": (
                "Developed machine learning models to solve complex business problems. "
                "Worked closely with product and data teams to ensure alignment. Delivered "
                "projects on time and under budget. Utilized best practices in model development "
                "and deployment. Contributed to a culture of innovation and continuous improvement."
            ),
        },
        {
            "title": "Machine Learning Engineer",
            "org": "Innovex Solutions",
            "description": (
                "Built and deployed machine learning pipelines. Collaborated with stakeholders "
                "to understand requirements. Ensured high-quality deliverables. Applied "
                "state-of-the-art techniques to real-world datasets."
            ),
        },
    ],
    skills=[
        "Python", "PyTorch", "LLMs", "RAG", "NLP", "AWS", "GCP",
        "Machine Learning", "Deep Learning", "Leadership", "Strategy",
        "Communication", "Agile", "Innovation",
    ],
    timeline=[
        # Implausible: overlapping dates and suspiciously short tenures
        TimelineEntry("Machine Learning Engineer",        "Innovex Solutions", datetime(2020, 1, 1), datetime(2021, 2, 1)),
        TimelineEntry("Senior Machine Learning Engineer", "Datatron Systems",  datetime(2020, 11, 1), datetime(2022, 3, 1)),
        TimelineEntry("AI Engineering Lead",              "NeuroPath Health",  datetime(2022, 4, 1), None),
    ],
    screening_questions=[
        "Describe a RAG pipeline you built end-to-end in production.",
        "How did you fine-tune an LLM for a clinical task?",
    ],
    screening_answers=[
        (
            "I built a comprehensive RAG pipeline that leveraged state-of-the-art language "
            "models to retrieve and generate highly relevant responses. The system significantly "
            "improved information retrieval accuracy and was well-received by stakeholders. "
            "I coordinated with the team to ensure seamless deployment and monitored performance "
            "continuously. The outcome was a transformative improvement in the product experience."
        ),
        (
            "I fine-tuned a large language model on clinical notes, which resulted in greatly "
            "improved performance on downstream tasks. I worked with the data team to curate "
            "high-quality training data and applied best practices in model evaluation. "
            "The model was successfully deployed and received positive feedback from clinicians. "
            "It was a challenging but rewarding project that demonstrated the power of AI in healthcare."
        ),
    ],
    web_signals={
        "github_public_repos": 2.0,
        "github_commit_recency_days": 290.0,
        "linkedin_connections": 1843.0,
        "endorsement_reciprocity": 0.14,
    },
)


# ===========================================================================
# ── PROFILE 4 ── Product Manager, Growth (Legitimate) ──────────────────────
# ===========================================================================
p4 = LinkedInSyntheticProfile(
    is_fraud=False,
    jd_text=(
        "Senior Product Manager for a B2C fintech app. You'll own the growth and "
        "activation funnel, working with engineering, design, and data. 4+ years PM "
        "experience, strong experimentation background required."
    ),
    summary=(
        "PM focused on the messy middle of the funnel—where users drop off before "
        "they ever see the product's real value. I've spent five years running "
        "experiments at two consumer fintech companies and have a high tolerance for "
        "counterintuitive results. I track decisions in writing so future teams don't "
        "repeat mistakes."
    ),
    experiences=[
        {
            "title": "Senior Product Manager, Growth",
            "org": "Finlo App",
            "description": (
                "Owned onboarding and activation (D1/D7 retention). Ran 41 experiments over "
                "18 months; 17 shipped. Most impactful: redesigned the bank-link step based on "
                "support ticket analysis, lifting bank-link completion from 54% to 71% (+6 pp on "
                "D7 retention). Maintained a living decision log of 80+ entries used by PMs "
                "who joined after me."
            ),
        },
        {
            "title": "Product Manager",
            "org": "ClearSpend",
            "description": (
                "Launched spend-categorization feature from 0 to GA in 7 months, becoming the "
                "second most-used feature in the app. Partnered with compliance to ship a "
                "redesigned KYC flow that cut document-rejection rate from 22% to 9%."
            ),
        },
        {
            "title": "Associate Product Manager",
            "org": "Nuvia Bank",
            "description": (
                "Supported three senior PMs on the savings product. Wrote 40+ PRDs, ran weekly "
                "user interviews, and built the internal NPS dashboard that surfaced insight "
                "used to reprioritize the Q3 roadmap."
            ),
        },
    ],
    skills=[
        "Product Strategy", "A/B Testing", "SQL", "Funnel Analysis",
        "User Research", "PRD Writing", "Roadmapping", "Amplitude", "Figma",
    ],
    timeline=[
        TimelineEntry("Associate Product Manager", "Nuvia Bank",  datetime(2018, 7, 1),  datetime(2020, 6, 1)),
        TimelineEntry("Product Manager",            "ClearSpend",  datetime(2020, 7, 1),  datetime(2022, 9, 1)),
        TimelineEntry("Senior Product Manager, Growth", "Finlo App", datetime(2022, 10, 1), None),
    ],
    screening_questions=[
        "Tell me about an experiment that failed and what you learned.",
        "How do you decide what to put on the roadmap when everything is a priority?",
    ],
    screening_answers=[
        (
            "At Finlo we ran an in-app savings nudge we were confident would lift activation—"
            "it was based on solid behavioral research and tested well in usability sessions. "
            "It hurt D7 retention by 2.1 pp. Post-hoc analysis showed we'd primed users to "
            "focus on a goal they hadn't set yet, which created anxiety rather than motivation. "
            "We killed the feature and I documented the exact user-state model that led us "
            "astray. That doc has prevented two similar mistakes since."
        ),
        (
            "I force a stack-rank with a simple rubric: reach × impact × confidence ÷ effort, "
            "all scored 1–5. But the real discipline is making the score public before the "
            "discussion, not after, so it can't be retrofitted to justify a pre-decided winner. "
            "At ClearSpend, running this process quarterly surfaced that we'd been over-investing "
            "in power-user features while ignoring a drop-off that was costing us 30% of signups."
        ),
    ],
    web_signals={
        "linkedin_connections": 534.0,
        "endorsement_reciprocity": 0.61,
        "medium_articles_published": 4.0,
        "speaker_event_hits": 1.0,
    },
)


# ===========================================================================
# ── PROFILE 5 ── DevOps / Platform Engineer (Legitimate) ───────────────────
# ===========================================================================
p5 = LinkedInSyntheticProfile(
    is_fraud=False,
    jd_text=(
        "Platform Engineer to own our internal developer platform on AWS. "
        "Must have deep Terraform, GitHub Actions, and EKS experience. "
        "Security posture ownership (SOC 2 / ISO 27001) a plus."
    ),
    summary=(
        "Platform engineer who thinks a good internal platform is a product, not "
        "a cost center. I've built CI/CD systems, secret-management pipelines, and "
        "self-service infrastructure tooling for teams ranging from 8 to 200 engineers. "
        "I'm happiest when developers stop opening Slack tickets and start deploying themselves."
    ),
    experiences=[
        {
            "title": "Staff Platform Engineer",
            "org": "Lumen Health",
            "description": (
                "Built a self-service Terraform catalog (25 modules) that reduced new-service "
                "setup from 3 days to 45 minutes. Migrated secrets from hardcoded env vars to "
                "AWS Secrets Manager + Vault, closing 14 SOC 2 findings. Reduced CI pipeline "
                "average runtime from 18 min to 6 min via parallelisation and remote caching "
                "(Buildkite + S3). Interviewed and hired 3 engineers."
            ),
        },
        {
            "title": "Senior DevOps Engineer",
            "org": "Portside Logistics",
            "description": (
                "Owned Kubernetes cluster operations (EKS, 40 microservices). Implemented "
                "Karpenter for node autoscaling, cutting EC2 spend by $180 k/year. Led "
                "incident response for three P1 outages; authored runbooks that cut MTTR "
                "by 55%. Introduced Atlantis for Terraform plan/apply via pull requests."
            ),
        },
        {
            "title": "DevOps Engineer",
            "org": "Brightfield Software",
            "description": (
                "Migrated a monolithic Jenkins setup to GitHub Actions, halving pipeline "
                "maintenance burden. Containerised 8 legacy services and deployed on ECS. "
                "Wrote internal wiki documentation used by 40 engineers."
            ),
        },
    ],
    skills=[
        "Terraform", "AWS", "Kubernetes", "GitHub Actions", "Vault",
        "Buildkite", "Python", "Bash", "SOC 2", "Karpenter", "EKS",
    ],
    timeline=[
        TimelineEntry("DevOps Engineer",        "Brightfield Software", datetime(2017, 3, 1),  datetime(2019, 11, 1)),
        TimelineEntry("Senior DevOps Engineer", "Portside Logistics",   datetime(2019, 12, 1), datetime(2022, 2, 1)),
        TimelineEntry("Staff Platform Engineer","Lumen Health",         datetime(2022, 3, 1),  None),
    ],
    screening_questions=[
        "How would you design a secrets management system for a 150-engineer org?",
        "Walk me through an incident where the platform itself caused an outage.",
    ],
    screening_answers=[
        (
            "I'd start with Vault in HA mode behind an internal load balancer, with AWS KMS "
            "as the auto-unseal backend. Secrets are namespaced by service and environment; "
            "apps get short-lived dynamic credentials via AppRole or Kubernetes auth rather "
            "than static keys. Rotation is automated via Vault's dynamic secrets engine for "
            "RDS and IAM. At Lumen I layered AWS Secrets Manager for non-Vault consumers, with "
            "a sync job keeping the two stores consistent so we didn't lock out teams that "
            "hadn't migrated yet."
        ),
        (
            "At Portside, a Terraform plan in CI accidentally targeted production because "
            "a workspace variable was unset and fell back to a default. Atlantis ran the apply "
            "automatically. Three EKS node groups were deleted mid-business-day. We restored "
            "within 40 minutes from a node template snapshot, but 22 minutes of API errors hit "
            "customers. The fix was mandatory explicit workspace declaration and a prod-branch "
            "protection rule requiring two approvals on any Atlantis apply."
        ),
    ],
    web_signals={
        "github_public_repos": 22.0,
        "github_commit_recency_days": 8.0,
        "linkedin_connections": 390.0,
        "endorsement_reciprocity": 0.74,
        "speaker_event_hits": 2.0,
    },
)


# ===========================================================================
# ── PROFILE 6 ── UX / Product Designer (Legitimate) ────────────────────────
# ===========================================================================
p6 = LinkedInSyntheticProfile(
    is_fraud=False,
    jd_text=(
        "Senior UX Designer for a B2B SaaS analytics tool. You'll own end-to-end design "
        "for our reporting suite—research, wireframes, prototypes, and design system "
        "contributions. Figma proficiency and experience with data-dense interfaces required."
    ),
    summary=(
        "Designer who spent three years doing pure research before switching to product "
        "design. That order was deliberate: I believe the research leg is usually skipped, "
        "and its absence is visible in shipped products. I specialise in data-dense "
        "dashboards and enterprise tables—the kind of UI most designers avoid."
    ),
    experiences=[
        {
            "title": "Senior UX Designer",
            "org": "Clairo Analytics",
            "description": (
                "Redesigned the report builder used by 4,000 enterprise clients. Ran 28 "
                "moderated user sessions and 3 card-sort studies before touching Figma. "
                "Shipped a redesigned filter/group-by experience that reduced task completion "
                "time from 4.2 min to 1.8 min (SUS score improved from 62 to 81). "
                "Contributed 40 components to the design system; reduced designer-dev "
                "handoff discrepancies by 60% via component-level annotation specs."
            ),
        },
        {
            "title": "Product Designer",
            "org": "Heron Insights",
            "description": (
                "Sole designer on a 12-person team. Shipped the v2 dashboard from research "
                "to GA in 5 months. Established the company's first design system (Figma "
                "variables + Storybook parity). Introduced weekly design critiques that "
                "became a permanent team ritual."
            ),
        },
        {
            "title": "UX Researcher",
            "org": "Formwell Labs",
            "description": (
                "Planned and executed 120 user interviews across 4 product areas. "
                "Delivered 11 research reports that directly influenced roadmap priorities. "
                "Built the internal research repository (Dovetail) adopted by a 6-person "
                "research team."
            ),
        },
    ],
    skills=[
        "Figma", "User Research", "Usability Testing", "Information Architecture",
        "Design Systems", "Prototyping", "Data Visualization", "Storybook",
        "Accessibility (WCAG 2.1)", "Card Sorting",
    ],
    timeline=[
        TimelineEntry("UX Researcher",    "Formwell Labs",    datetime(2016, 8, 1),  datetime(2019, 7, 1)),
        TimelineEntry("Product Designer", "Heron Insights",   datetime(2019, 8, 1),  datetime(2022, 1, 1)),
        TimelineEntry("Senior UX Designer","Clairo Analytics",datetime(2022, 2, 1),  None),
    ],
    screening_questions=[
        "How do you approach designing a complex data table with many configuration options?",
        "Tell me about a research finding that changed the design direction significantly.",
    ],
    screening_answers=[
        (
            "I start with the user's mental model of the data, not the data model. For Clairo's "
            "report builder, I ran a card-sort to understand how clients categorise their "
            "metrics—it revealed a split between 'operational' and 'executive' frames that the "
            "old UI completely ignored. I then prototyped three filter paradigms at lo-fi "
            "fidelity, tested each with six users, and picked the one that generated fewest "
            "support questions rather than the highest satisfaction score, because satisfaction "
            "can mask learned helplessness."
        ),
        (
            "At Formwell I was researching a project-management module. Every interview, users "
            "praised the notification system. Week five, I added a diary study and found they "
            "were actually ignoring 80% of notifications and had built workarounds in Slack. "
            "What they 'praised' was the concept of being notified, not the implementation. "
            "That finding killed a planned notification-expansion feature and redirected six "
            "weeks of engineering toward notification grouping and quiet-hours—a change that "
            "later correlated with a 12-point NPS improvement."
        ),
    ],
    web_signals={
        "dribbble_followers": 1240.0,
        "behance_project_views": 8900.0,
        "linkedin_connections": 708.0,
        "endorsement_reciprocity": 0.65,
    },
)


# ===========================================================================
# ── PROFILE 7 ── "Blockchain / Web3 Architect" (FRAUD) ─────────────────────
# ===========================================================================
p7 = LinkedInSyntheticProfile(
    is_fraud=True,
    jd_text=(
        "Web3 Protocol Engineer for a DeFi startup. Must have deep smart-contract "
        "security experience (Solidity, Foundry/Hardhat), cross-chain bridge design, "
        "and MEV mitigation knowledge. Audit experience strongly preferred."
    ),
    summary=(
        "Visionary blockchain architect with a deep passion for decentralized systems "
        "and Web3 innovation. Extensive experience designing and deploying enterprise-grade "
        "smart contract architectures. I am a thought leader in the DeFi space and "
        "have contributed to multiple high-impact protocols. Driven by the mission to "
        "democratize finance through technology."
    ),
    experiences=[
        {
            "title": "Lead Blockchain Architect",
            "org": "Nexus Protocol",
            "description": (
                "Architected the core smart contract infrastructure for a leading DeFi protocol. "
                "Ensured security and reliability of the codebase. Collaborated with world-class "
                "teams to deliver revolutionary Web3 products. Drove adoption and community growth. "
                "Provided strategic technical leadership."
            ),
        },
        {
            "title": "Senior Smart Contract Engineer",
            "org": "ChainVault",
            "description": (
                "Developed and deployed smart contracts on multiple EVM chains. Contributed to "
                "protocol governance and tokenomics design. Worked with auditors to improve "
                "code quality. Delivered high-value projects for key stakeholders."
            ),
        },
        {
            "title": "Blockchain Developer",
            "org": "Distributed Labs",
            "description": (
                "Built blockchain solutions for enterprise clients. Applied cutting-edge "
                "distributed ledger technology. Supported go-to-market activities and "
                "technical sales."
            ),
        },
    ],
    skills=[
        "Solidity", "Web3", "DeFi", "Smart Contracts", "Blockchain",
        "Ethereum", "Tokenomics", "NFT", "Rust", "Cross-chain",
        "Leadership", "Innovation", "Strategic Vision",
    ],
    timeline=[
        # Implausible: two months at first role, immediate senior jump
        TimelineEntry("Blockchain Developer",          "Distributed Labs", datetime(2021, 1, 1), datetime(2021, 3, 1)),
        TimelineEntry("Senior Smart Contract Engineer","ChainVault",       datetime(2021, 2, 1), datetime(2022, 8, 1)),
        TimelineEntry("Lead Blockchain Architect",     "Nexus Protocol",   datetime(2022, 9, 1), None),
    ],
    screening_questions=[
        "How would you prevent a reentrancy attack in a Solidity lending contract?",
        "Explain how you would design a cross-chain bridge and its main security risks.",
    ],
    screening_answers=[
        (
            "Reentrancy is a critical vulnerability and I have extensive experience addressing "
            "it in production contracts. The key is to implement robust security patterns and "
            "follow industry best practices to ensure the contract is protected. I have worked "
            "with top audit firms and understand how to structure code defensively. "
            "Security is always my first priority in smart contract development."
        ),
        (
            "Cross-chain bridge design is one of my core competencies. I have designed bridges "
            "that securely transfer assets across multiple blockchains. The approach involves "
            "careful consideration of trust assumptions and security models. I ensure that all "
            "edge cases are handled and work closely with the security team to prevent exploits. "
            "My bridges have processed significant transaction volume without incidents."
        ),
    ],
    web_signals={
        "github_public_repos": 1.0,
        "github_commit_recency_days": 410.0,
        "linkedin_connections": 2100.0,
        "endorsement_reciprocity": 0.09,
        "etherscan_verified_contracts": 0.0,
    },
)


# ===========================================================================
# ── PROFILE 8 ── Financial Analyst → FP&A Manager (Legitimate) ─────────────
# ===========================================================================
p8 = LinkedInSyntheticProfile(
    is_fraud=False,
    jd_text=(
        "FP&A Manager for a 400-person SaaS company. Own the annual plan, monthly close "
        "commentary, and strategic finance projects. Excel/SQL required; Adaptive "
        "Planning or Anaplan experience a plus. CPA or CFA preferred."
    ),
    summary=(
        "Finance professional who migrated from audit to FP&A because I wanted to "
        "influence decisions, not just document them. Seven years of close experience "
        "across public accounting and two SaaS companies. I write the kind of CFO "
        "commentary that gets read past the first paragraph."
    ),
    experiences=[
        {
            "title": "FP&A Manager",
            "org": "Openlane SaaS",
            "description": (
                "Owned the $180 M ARR financial model, annual operating plan, and monthly "
                "CFO board package. Built a rolling 13-week cash-flow model that correctly "
                "flagged a $4 M shortfall six weeks before it would have appeared in actuals, "
                "allowing the CFO to draw on the revolver proactively. Reduced monthly close "
                "from day 8 to day 4 by automating three manual data pulls (Python + Adaptive "
                "Planning API). Managed one senior analyst."
            ),
        },
        {
            "title": "Senior Financial Analyst",
            "org": "Verida Software",
            "description": (
                "Built the SaaS metrics dashboard (ARR, NRR, LTV, CAC payback) used weekly "
                "by the exec team. Supported three strategic projects: pricing model redesign "
                "(contributed to a 9% ARPU lift), Series C data room build, and a build-vs-buy "
                "analysis for a $12 M acquisition target."
            ),
        },
        {
            "title": "Senior Auditor",
            "org": "Deloitte",
            "description": (
                "Audited public-company clients ($200 M–$2 B revenue) in technology and "
                "healthcare. Led two SOX 404 engagements. Obtained CPA license. Promoted "
                "to senior in 18 months (standard cycle: 24 months)."
            ),
        },
    ],
    skills=[
        "FP&A", "Financial Modeling", "Adaptive Planning", "SQL", "Python",
        "Excel", "SaaS Metrics", "Board Reporting", "CPA", "Variance Analysis",
    ],
    timeline=[
        TimelineEntry("Senior Auditor",         "Deloitte",       datetime(2015, 9, 1), datetime(2018, 8, 1)),
        TimelineEntry("Senior Financial Analyst","Verida Software",datetime(2018, 9, 1), datetime(2021, 4, 1)),
        TimelineEntry("FP&A Manager",            "Openlane SaaS",  datetime(2021, 5, 1), None),
    ],
    screening_questions=[
        "How do you build a reliable SaaS revenue forecast?",
        "Tell me about a time your financial analysis directly changed a business decision.",
    ],
    screening_answers=[
        (
            "I build from a cohort-based ARR waterfall: starting MRR, plus new logo bookings "
            "(sales-pipeline probability-weighted), plus expansion (historical NRR by segment), "
            "minus churn (survival-curve model by cohort age and tier). I calibrate the model "
            "monthly against actuals and maintain a bias log—if I'm systematically over- or "
            "under-forecasting a segment, I want to know why before I add another coefficient."
        ),
        (
            "At Verida, the CRO wanted to accelerate SMB hiring based on a pipeline that looked "
            "healthy. I showed that SMB CAC payback had lengthened from 14 months to 22 months "
            "over the prior three quarters while enterprise payback held at 11 months. "
            "The board approved reallocating four SMB AE headcount to enterprise. Within "
            "two quarters, blended CAC payback returned to 14 months."
        ),
    ],
    web_signals={
        "linkedin_connections": 423.0,
        "endorsement_reciprocity": 0.58,
        "cpa_license_verified": 1.0,
        "speaker_event_hits": 0.0,
    },
)


# ===========================================================================
# ── PROFILE 9 ── Cybersecurity Engineer (Legitimate) ───────────────────────
# ===========================================================================
p9 = LinkedInSyntheticProfile(
    is_fraud=False,
    jd_text=(
        "Application Security Engineer for a cloud-native fintech. Own secure SDLC, "
        "threat modelling, SAST/DAST tooling, and bug bounty triage. OSCP or equivalent "
        "and 3+ years AppSec experience required."
    ),
    summary=(
        "Security engineer who came up through penetration testing and crossed into "
        "AppSec because I got tired of writing reports that sat in SharePoint. "
        "I believe security programs only work when developers stop seeing them as "
        "external audits and start seeing them as a service. OSCP, 5 CVEs credited."
    ),
    experiences=[
        {
            "title": "Application Security Engineer",
            "org": "Arcadia Financial",
            "description": (
                "Integrated Semgrep and Snyk into GitHub Actions across 34 repositories, "
                "surfacing 380 findings in week one; triaged to 22 P1s, all remediated within "
                "SLA. Ran STRIDE threat models on six new product features; two resulted in "
                "architecture changes before a line of production code was written. "
                "Managed external bug bounty (HackerOne): triaged 91 reports, paid 14 valid "
                "bounties, credited with closing two SSRF chains."
            ),
        },
        {
            "title": "Penetration Tester",
            "org": "Ironshore Security",
            "description": (
                "Delivered 60+ web and API penetration tests for clients in finance, healthcare, "
                "and SaaS. Discovered and responsibly disclosed 5 CVEs. Wrote an internal "
                "reporting template adopted firm-wide that cut report delivery time by 30%. "
                "Mentored two junior testers."
            ),
        },
        {
            "title": "Security Analyst",
            "org": "Fulton Credit Union",
            "description": (
                "Monitored SIEM alerts (Splunk), responded to 4 phishing incidents without "
                "data loss, and managed vulnerability scanning (Tenable) across 600 endpoints. "
                "Obtained Security+ and began OSCP study while in role."
            ),
        },
    ],
    skills=[
        "Application Security", "Penetration Testing", "SAST/DAST", "Semgrep",
        "Snyk", "Threat Modeling (STRIDE)", "Bug Bounty", "Python", "Burp Suite",
        "OSCP", "HackerOne", "Splunk",
    ],
    timeline=[
        TimelineEntry("Security Analyst",          "Fulton Credit Union", datetime(2016, 6, 1),  datetime(2018, 9, 1)),
        TimelineEntry("Penetration Tester",         "Ironshore Security",  datetime(2018, 10, 1), datetime(2021, 7, 1)),
        TimelineEntry("Application Security Engineer","Arcadia Financial", datetime(2021, 8, 1),  None),
    ],
    screening_questions=[
        "Walk me through how you would threat-model an OAuth 2.0 implementation.",
        "How do you convince a development team to fix a medium-severity finding they've deprioritized?",
    ],
    screening_answers=[
        (
            "I'd start with the trust boundaries: client ↔ auth server, auth server ↔ resource "
            "server, and any cross-origin flows. STRIDE pass on each: spoofing (token leakage "
            "via redirect_uri mismatch, open redirector), tampering (state parameter replay, "
            "PKCE downgrade), info disclosure (access token in referrer/logs), and elevation "
            "(scope confusion, refresh token theft). I map each threat to a concrete test case—"
            "e.g., I'd manually test that the server rejects any redirect_uri not on the allow-"
            "list, including encoded variants. For implicit flow remnants I'd check for "
            "fragment-in-log leakage."
        ),
        (
            "I don't argue severity in the abstract—I chain it to a realistic attack scenario. "
            "At Arcadia, I had an IDOR on an internal admin endpoint rated medium because it "
            "required authentication. I showed the team a two-step chain: phish a low-privilege "
            "admin → use IDOR to enumerate all customer accounts. That reframe got it escalated "
            "to P1 and fixed in the next sprint. If I can't chain it to a realistic scenario, "
            "I'll accept the deprioritization and document the risk acceptance formally."
        ),
    ],
    web_signals={
        "github_public_repos": 14.0,
        "github_commit_recency_days": 19.0,
        "hackerone_reputation": 1870.0,
        "linkedin_connections": 341.0,
        "endorsement_reciprocity": 0.70,
        "cve_credits": 5.0,
    },
)


# ===========================================================================
# ── PROFILE 10 ── "Enterprise Sales Director" (FRAUD) ──────────────────────
# ===========================================================================
p10 = LinkedInSyntheticProfile(
    is_fraud=True,
    jd_text=(
        "Enterprise Sales Director for a B2B SaaS security platform. Must have 7+ years "
        "of enterprise SaaS sales, proven ability to close $500 k+ deals, and deep "
        "experience navigating complex procurement processes in Fortune 500 accounts."
    ),
    summary=(
        "Dynamic and results-oriented Enterprise Sales Director with a stellar record of "
        "exceeding quota and building high-performing teams. I am a strategic thinker who "
        "builds deep relationships at the C-suite level. My holistic approach to enterprise "
        "selling consistently unlocks transformational value for clients and stakeholders. "
        "Passionate about cybersecurity and the future of the digital enterprise."
    ),
    experiences=[
        {
            "title": "Enterprise Sales Director",
            "org": "ShieldCore Security",
            "description": (
                "Consistently exceeded annual sales targets. Built and managed a team of "
                "high-performing enterprise account executives. Drove strategic initiatives "
                "across Fortune 500 accounts. Delivered exceptional client outcomes. "
                "Contributed significantly to company revenue growth."
            ),
        },
        {
            "title": "Senior Enterprise Account Executive",
            "org": "Nexwave Technologies",
            "description": (
                "Exceeded quota by a significant margin year over year. Closed several "
                "landmark enterprise deals. Developed and executed territory plans. "
                "Built strong executive relationships. Recognised as a top performer."
            ),
        },
        {
            "title": "Account Executive",
            "org": "DataSphere Inc.",
            "description": (
                "Managed a portfolio of enterprise accounts. Met and exceeded sales targets. "
                "Developed deep client relationships. Contributed to team success."
            ),
        },
    ],
    skills=[
        "Enterprise Sales", "B2B SaaS", "C-Suite Engagement", "Pipeline Management",
        "Salesforce", "MEDDIC", "Negotiation", "Leadership", "Revenue Growth",
        "Strategic Partnerships", "Quota Attainment", "Cybersecurity",
    ],
    timeline=[
        # Suspiciously rapid trajectory, all identical-length stints
        TimelineEntry("Account Executive",               "DataSphere Inc.",    datetime(2019, 1, 1), datetime(2020, 12, 1)),
        TimelineEntry("Senior Enterprise Account Executive","Nexwave Technologies",datetime(2021, 1, 1),datetime(2022, 12, 1)),
        TimelineEntry("Enterprise Sales Director",       "ShieldCore Security",datetime(2023, 1, 1), None),
    ],
    screening_questions=[
        "Describe the largest enterprise deal you've closed. Walk me through the process.",
        "How do you build a territory plan for a new region with no existing pipeline?",
    ],
    screening_answers=[
        (
            "I have closed numerous large enterprise deals throughout my career. My approach "
            "is to build strong executive relationships and deliver compelling value propositions "
            "tailored to the client's strategic priorities. I leverage my deep industry expertise "
            "to navigate complex procurement processes efficiently. My deals consistently deliver "
            "transformational outcomes for clients and significant revenue for my employer. "
            "I am proud of my track record of success in this area."
        ),
        (
            "Building a new territory is one of my strengths. I take a strategic approach, "
            "identifying key target accounts and building executive relationships from day one. "
            "I develop a comprehensive territory plan aligned with company objectives and "
            "execute it with discipline and focus. My ability to generate pipeline quickly "
            "and convert opportunities efficiently has been recognized by leadership at "
            "every company I've worked for."
        ),
    ],
    web_signals={
        "linkedin_connections": 3200.0,
        "endorsement_reciprocity": 0.08,
        "github_public_repos": 0.0,
        "salesforce_trailhead_badges": 0.0,
        "speaker_event_hits": 0.0,
    },
)


# ===========================================================================
# ── PROFILE 11 ── Supply Chain / Operations Manager (Legitimate) ────────────
# ===========================================================================
p11 = LinkedInSyntheticProfile(
    is_fraud=False,
    jd_text=(
        "Supply Chain Manager for a consumer electronics manufacturer. Manage a $90 M "
        "component purchasing budget, demand planning, and 3PL relationships. "
        "APICS CPIM or CSCP preferred. ERP (SAP or Oracle) experience required."
    ),
    summary=(
        "Supply chain manager with nine years in electronics and consumer goods. "
        "I've managed sourcing across Taiwan, Malaysia, and Mexico, and spent enough "
        "time on factory floors to understand why plans fall apart three tiers down. "
        "I believe demand planning is 40% modelling and 60% stakeholder management."
    ),
    experiences=[
        {
            "title": "Supply Chain Manager",
            "org": "Halcyon Electronics",
            "description": (
                "Managed $90 M component purchasing budget across 38 suppliers. Redesigned "
                "the S&OP process, reducing forecast error from 22% to 11% MAPE over 12 months. "
                "Led dual-source qualification for three single-source components that had caused "
                "a $3.1 M production stoppage in 2021; new dual-source contracts live within 8 "
                "months. Negotiated 4.2% blended cost reduction on top-20 components by volume. "
                "Managed two buyers and one demand planner."
            ),
        },
        {
            "title": "Senior Demand Planner",
            "org": "Altus Consumer Goods",
            "description": (
                "Owned statistical demand forecasting for 420 SKUs in SAP APO. Reduced excess "
                "inventory by $7 M through improved slow-mover identification. Collaborated "
                "with Sales to introduce collaborative forecasting that reduced promotional "
                "forecast error by 35%."
            ),
        },
        {
            "title": "Procurement Analyst",
            "org": "Vantix Manufacturing",
            "description": (
                "Supported 12 commodity managers across direct materials. Built SAP spend "
                "analysis toolkit used quarterly by the VP of Procurement. Obtained CPIM "
                "certification while in role."
            ),
        },
    ],
    skills=[
        "Supply Chain Management", "Demand Planning", "S&OP", "SAP APO", "Oracle SCM",
        "Supplier Negotiation", "CPIM", "Inventory Optimization", "3PL Management",
        "MAPE Analysis", "Dual Sourcing",
    ],
    timeline=[
        TimelineEntry("Procurement Analyst",  "Vantix Manufacturing",   datetime(2015, 6, 1),  datetime(2018, 3, 1)),
        TimelineEntry("Senior Demand Planner","Altus Consumer Goods",   datetime(2018, 4, 1),  datetime(2021, 9, 1)),
        TimelineEntry("Supply Chain Manager", "Halcyon Electronics",    datetime(2021, 10, 1), None),
    ],
    screening_questions=[
        "How did you respond to a major supply disruption, and what systemic change did you make?",
        "Walk me through how you run the monthly S&OP process.",
    ],
    screening_answers=[
        (
            "At Halcyon in Q3 2021, a single-source IC from a Taiwanese fab went on allocation "
            "with 16 weeks' notice. We had 8 weeks of stock. I prioritised the production plan "
            "toward our highest-margin SKUs, air-freighted a partial safety stock from a "
            "distributor at a $180 k premium to protect our top two retail accounts, and "
            "simultaneously issued RFQs to three alternative fabs. The production stoppage "
            "lasted 9 days instead of a projected 6 weeks. The systemic fix was the dual-source "
            "qualification programme I then built for all 22 single-source components above "
            "$500 k annual spend."
        ),
        (
            "Week 1: statistical baseline from SAP APO—I run MAPE by SKU family and flag any "
            "item where the model has been biased the same direction three months running. "
            "Week 2: commercial overlay—Sales submits adjustments with written rationale; I "
            "push back if the adjustment exceeds ±20% without a named account or promo plan "
            "behind it. Week 3: supply review—I reconcile the demand plan against confirmed "
            "supplier lead times and flag any gaps to procurement. Week 4: executive S&OP—"
            "30-minute meeting, pre-read circulated 48 hours prior, decisions logged in writing."
        ),
    ],
    web_signals={
        "linkedin_connections": 298.0,
        "endorsement_reciprocity": 0.55,
        "cpim_certified": 1.0,
        "speaker_event_hits": 0.0,
    },
)


# ===========================================================================
# ── PROFILE 12 ── "Head of Data Engineering" (FRAUD) ───────────────────────
# ===========================================================================
p12 = LinkedInSyntheticProfile(
    is_fraud=True,
    jd_text=(
        "Head of Data Engineering for a Series D healthtech. You will own the lakehouse "
        "architecture (Databricks / Delta Lake), data contracts, and a team of 8 engineers. "
        "Must have 6+ years of data engineering with demonstrated team leadership."
    ),
    summary=(
        "Accomplished Head of Data Engineering with a proven history of building world-class "
        "data platforms that power data-driven organisations. Expert in modern data stack "
        "technologies and cloud-native architectures. Recognised industry thought leader "
        "and speaker. I thrive in fast-paced, high-growth environments and excel at "
        "building and inspiring high-performing engineering teams."
    ),
    experiences=[
        {
            "title": "Head of Data Engineering",
            "org": "Vitalia Health",
            "description": (
                "Built and led the data engineering function from the ground up. Implemented "
                "a modern data platform leveraging best-in-class technologies. Delivered "
                "significant improvements in data quality, reliability, and timeliness. "
                "Grew and mentored the data engineering team. Partnered with cross-functional "
                "leaders to enable data-driven decision-making across the organisation."
            ),
        },
        {
            "title": "Senior Data Engineer",
            "org": "Luminos Analytics",
            "description": (
                "Designed and implemented scalable data pipelines. Worked closely with data "
                "scientists and analysts to deliver high-quality data products. Contributed "
                "to platform architecture decisions. Delivered projects ahead of schedule "
                "and was recognised for technical excellence."
            ),
        },
        {
            "title": "Data Engineer",
            "org": "CloudBridge Solutions",
            "description": (
                "Developed ETL pipelines and data warehouse solutions. Supported analytics "
                "team with data infrastructure needs. Learned and applied modern data engineering "
                "practices."
            ),
        },
    ],
    skills=[
        "Databricks", "Delta Lake", "Apache Spark", "dbt", "Airflow", "Snowflake",
        "Python", "SQL", "Data Architecture", "Leadership", "Team Building",
        "Data Contracts", "Modern Data Stack", "Cloud (AWS/GCP/Azure)",
    ],
    timeline=[
        # All tenures exactly 18 months — unusually uniform
        TimelineEntry("Data Engineer",        "CloudBridge Solutions", datetime(2019, 1, 1),  datetime(2020, 6, 1)),
        TimelineEntry("Senior Data Engineer", "Luminos Analytics",     datetime(2020, 7, 1),  datetime(2022, 1, 1)),
        TimelineEntry("Head of Data Engineering","Vitalia Health",     datetime(2022, 2, 1),  None),
    ],
    screening_questions=[
        "How would you implement data contracts across a large microservices ecosystem?",
        "Describe how you debugged a severe data quality incident in production.",
    ],
    screening_answers=[
        (
            "Data contracts are a foundational element of a mature data platform and I have "
            "extensive experience implementing them. The key is to establish clear ownership "
            "and governance frameworks that ensure all data producers adhere to agreed schemas "
            "and SLAs. I leverage modern tooling and best practices to enforce contracts at "
            "scale. The outcome is dramatically improved data reliability and trust across "
            "the organisation."
        ),
        (
            "When data quality incidents occur, I take a structured approach to diagnosis and "
            "resolution. I work with the team to identify the root cause as quickly as possible "
            "and implement both an immediate fix and a longer-term systemic solution. I believe "
            "in blameless post-mortems and use incidents as learning opportunities to improve "
            "our processes and tooling. Communication with stakeholders throughout the incident "
            "is also a priority for me."
        ),
    ],
    web_signals={
        "github_public_repos": 3.0,
        "github_commit_recency_days": 187.0,
        "linkedin_connections": 2750.0,
        "endorsement_reciprocity": 0.11,
        "databricks_certified": 0.0,
        "speaker_event_hits": 0.0,
    },
)


# ===========================================================================
# Collect all profiles
# ===========================================================================
ALL_PROFILES = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12]

PROFILE_META = [
    {"id": 1,  "persona": "Senior Software Engineer",        "is_fraud": p1.is_fraud},
    {"id": 2,  "persona": "Data Scientist – ML Platform",    "is_fraud": p2.is_fraud},
    {"id": 3,  "persona": "Full-Stack AI Lead",              "is_fraud": p3.is_fraud},
    {"id": 4,  "persona": "Product Manager – Growth",        "is_fraud": p4.is_fraud},
    {"id": 5,  "persona": "Staff Platform / DevOps Engineer","is_fraud": p5.is_fraud},
    {"id": 6,  "persona": "Senior UX Designer",              "is_fraud": p6.is_fraud},
    {"id": 7,  "persona": "Blockchain / Web3 Architect",     "is_fraud": p7.is_fraud},
    {"id": 8,  "persona": "FP&A Manager",                    "is_fraud": p8.is_fraud},
    {"id": 9,  "persona": "Application Security Engineer",   "is_fraud": p9.is_fraud},
    {"id": 10, "persona": "Enterprise Sales Director",       "is_fraud": p10.is_fraud},
    {"id": 11, "persona": "Supply Chain Manager",            "is_fraud": p11.is_fraud},
    {"id": 12, "persona": "Head of Data Engineering",        "is_fraud": p12.is_fraud},
]