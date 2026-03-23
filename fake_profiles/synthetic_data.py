from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import hf_token

hf_token.ensure_hf_environment()

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np

from features.timeline_coherence import TimelineEntry, TimelineCoherenceScorer
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
class LinkedInSyntheticProfile:
    is_fraud: bool
    jd_text: str
    summary: str
    experiences: List[Dict[str, str]]
    skills: List[str]
    timeline: List
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


@dataclass
class SignalProcessor:
    fraud_rate: float = 0.30
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))

    def __post_init__(self) -> None:
        
        self.skill_taxonomy_scorer = SkillTaxonomyScorer()

        self.timeline_scorer = TimelineCoherenceScorer()
        self.voice_scorer = ProfileVoiceConsistencyScorer()
        self.career_scorer = CareerProgressionSmoothnessScorer()
        self.structure_scorer = StructuralOrganizationScorer()
        self.operational_scorer = OperationalSpecificityScorer()
        self.cross_scorer = CrossAnswerConsistencyScorer()
        self.depth_scorer = DepthCollapseDeltaScorer()
        self.narrative_scorer = NarrativeCausalityScorer()
        self.answer_perplexity_scorer = AnswerPerplexityScorer()

    def random_profile(self) -> LinkedInSyntheticProfile:
        is_fraud = self.rng.random() < self.fraud_rate
        jd_text = "Software Engineer role at a tech company."
        summary = "Experienced software engineer with a passion for building scalable applications."
        experiences = [
            {"title": "Software Engineer", "org": "TechCorp", "description": "Worked on backend services."},
            {"title": "Junior Developer", "org": "WebStart", "description": "Assisted in developing web applications."}
        ] 
        skills = ["Python", "Machine Learning", "Data Analysis"]
        timeline = [
            TimelineEntry(stage=0, timestamp=datetime.now() - timedelta(days=365)),
            TimelineEntry(stage=1, timestamp=datetime.now() - timedelta(days=180)),
            TimelineEntry(stage=2, timestamp=datetime.now() - timedelta(days=90)),

        ]  
        screening_questions = [
            "Tell us about a challenging project you led.",
            "How do you stay current with technology trends?"
        ]
        screening_answers = [
            "I led a team to redesign our data pipeline, reducing query time by 40%.",
            "I regularly read technical blogs and contribute to open-source projects."
        ]
        web_signals = {
            "github_repos": self.rng.integers(5, 50),
            "stackoverflow_reputation": self.rng.integers(100, 5000),
            "linkedin_connections": self.rng.integers(100, 500)
        }
        
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
        skill_taxonomy_result = self.skill_taxonomy_scorer.score()
        skill_taxonomy = {
            "coverage_score": skill_taxonomy_result.coverage_score,
            "idiosyncrasy_score": skill_taxonomy_result.idiosyncrasy_score,
            "semantic_mirror_score": skill_taxonomy_result.semantic_mirror_score,
        }

        timeline_coherence_result = self.timeline_scorer.score(profile.timeline)
        timeline_coherence = {
            "timeline_coherence_fraud_score": timeline_coherence_result.timeline_coherence_fraud_score,
        }

        career_smoothness_result = self.career_scorer.score(profile.timeline)
        career_smoothness = {
            "career_smoothness_fraud_score": career_smoothness_result.career_smoothness_fraud_score,
        }

        voice_score_result = self.voice_scorer.score(profile.sections)
        voice_score = {
            "profile_voice_fraud_score": voice_score_result.profile_voice_fraud_score,
            "soft_skill_mean_rate": voice_score_result.soft_skill_mean_rate,
            "soft_skill_cv": voice_score_result.soft_skill_cv,
            "sentence_uniformity": voice_score_result.sentence_uniformity,
            "tech_density_cv": voice_score_result.tech_density_cv,
            "voice_cluster_penalty": voice_score_result.voice_cluster_penalty
        }
        return {**skill_taxonomy, **timeline_coherence, **career_smoothness, **voice_score}

    def stage_one(self, profile: LinkedInSyntheticProfile) -> dict:
        # Stage 1
        a1 = profile.screening_answers[0]

        # q1_op_fraud = self._operational_specificity_fraud(a1, profile.is_fraud)
        
        narrative_result = self.narrative_scorer.score(a1) 
        narrative = {
            "causal_span_score": narrative_result.causal_span_score,
            "result_entity_score": narrative_result.result_entity_score,
            "coherence_score": narrative_result.coherence_score,
        }

        answer_perplexity = self.answer_perplexity_scorer.score(a1)
        asnwer_perplexity_score = {
            "answer_perplexity_score": answer_perplexity.answer_perplexity_score,
            "raw_perplexity": answer_perplexity.raw_perplexity,
            "conditioned_perplexity": answer_perplexity.conditioned_perplexity,
            "mean_log_prob": answer_perplexity.mean_log_prob,
            "log_prob_cv": answer_perplexity.log_prob_cv,
            "high_prob_token_fraction": answer_perplexity.high_prob_token_fraction,
            "low_prob_token_fraction": answer_perplexity.low_prob_token_fraction,
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
        
        depth_delta = self.depth_scorer.compute_delta(a1, q1, a2, q2).depth_collapse_delta
        return {"depth_collapse_delta": depth_delta}

        # Stage 3
    def stage_three(self, profile: LinkedInSyntheticProfile) -> dict:
        cross = self.cross_scorer.score(profile.screening_answers)
        cross_answer_score = {
            "cross_answer_consistency_fraud_score": cross.cross_answer_consistency_fraud_score,
            "combined_consitency_score": cross.combined_consistency_score,
        }
        # q2_op_fraud = self._operational_specificity_fraud(a2, profile.is_fraud)
        # operational_agg = clip(0.5 * q1_op_fraud + 0.5 * q2_op_fraud)

        return cross_answer_score

       

