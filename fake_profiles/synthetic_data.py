import numpy as np
from dataclasses import dataclass, field
from typing import Dict
from features.tailored_aware import TailoredAwareFeatureExtractor 

@dataclass
class CandidateGenerator:
    fraud_rate: float = 0.3    # 30% of candidates are fraudulent
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(42)
    )

    def sample(self) -> Dict:
        is_fraud = self.rng.random() < self.fraud_rate

        if is_fraud:
            return self._fraud_candidate()
        return self._genuine_candidate()

    def _genuine_candidate(self) -> Dict:
        r = self.rng
        return {
            "is_fraud": False,
            # Profile: low burst, patchy skills, few retroactive metrics
            "vocab_burst_score":            r.normal(0.15, 0.08),
            "skill_completeness":           r.normal(0.55, 0.15),
            "retroactive_metric_density":   r.normal(0.8,  0.5),
            "buzzword_jd_overlap":          r.normal(0.35, 0.10),
            "connection_temporal_alignment":r.normal(0.65, 0.15),
            # Screening: medium perplexity, specific, hedges appropriately
            "answer_perplexity_q1":         r.normal(0.55, 0.12),
            "operational_specificity_q1":   r.normal(0.65, 0.15),
            "depth_collapse_delta":         r.normal(0.15, 0.08),
            "uncertainty_expression_rate":  r.normal(0.30, 0.10),
            "structural_org_score":         r.normal(0.35, 0.12),
            "recency_accuracy":             r.normal(0.70, 0.15),
            # Web: commits exist, traces exist
            "github_commit_density":        r.normal(0.60, 0.20),
            "tenure_web_exhaust_ratio":     r.normal(0.25, 0.10),
            "forum_temporal_trace":         r.normal(0.55, 0.20),
            "publication_verify_score":     r.normal(0.80, 0.15),
        }

    def _fraud_candidate(self) -> Dict:
        r = self.rng
        return {
            "is_fraud": True,
            # Profile: high burst, suspiciously clean, lots of retroactive numbers
            "vocab_burst_score":            r.normal(0.75, 0.10),
            "skill_completeness":           r.normal(0.92, 0.05),
            "retroactive_metric_density":   r.normal(3.5,  0.8),
            "buzzword_jd_overlap":          r.normal(0.82, 0.08),
            "connection_temporal_alignment":r.normal(0.15, 0.10),
            # Screening: low perplexity (LLM-smooth), vague, overconfident
            "answer_perplexity_q1":         r.normal(0.18, 0.08),
            "operational_specificity_q1":   r.normal(0.20, 0.10),
            "depth_collapse_delta":         r.normal(0.68, 0.12),
            "uncertainty_expression_rate":  r.normal(0.03, 0.03),
            "structural_org_score":         r.normal(0.88, 0.08),
            "recency_accuracy":             r.normal(0.25, 0.12),
            # Web: no traces
            "github_commit_density":        r.normal(0.05, 0.05),
            "tenure_web_exhaust_ratio":     r.normal(0.92, 0.08),
            "forum_temporal_trace":         r.normal(0.08, 0.06),
            "publication_verify_score":     r.normal(0.10, 0.10),
        }
    

    def extract_features(self, candidate: Dict) -> np.ndarray:
        # Order features as they are revealed in the environment



        return np.array([
            candidate["skill_completeness"],
            candidate["retroactive_metric_density"],
            candidate["buzzword_jd_overlap"],
            candidate["connection_temporal_alignment"],
            candidate["answer_perplexity_q1"],
            candidate["operational_specificity_q1"],
            candidate["depth_collapse_delta"],
            candidate["uncertainty_expression_rate"],
            candidate["structural_org_score"],
            candidate["recency_accuracy"],
            candidate["github_commit_density"],
            candidate["tenure_web_exhaust_ratio"],
            candidate["forum_temporal_trace"],
            candidate["publication_verify_score"],
        ], dtype=np.float32)


# For more realistic synthetic data, sdv (Synthetic Data Vault) lets you fit a 
# multivariate model to whatever ground-truth distribution you have and sample 
# from it; Faker is good for generating the raw text that features get computed 
# from; Great Expectations helps validate that generated data stays in realistic 
# ranges.

class ProfileGenerator:
    """Generate realistic LinkedIn profiles and resumes for candidates."""
    
    LINKEDIN_PROFILE_PROMPT = """Generate a LinkedIn profile summary for a {job_title} candidate.
Requirements:
- ~2-3 sentences, conversational tone
- Mention 2-3 relevant skills but NOT all required skills from the job posting
- Include a minor gap or pivot in background (e.g., "transitioned from X to Y")
- Use varied sentence structure: some short, some longer
- Avoid clichés like "passionate" or "results-driven"
- Sound like a real person, not polished or templated

Job posting requirements: {job_requirements}
Candidate background: {background}"""

    RESUME_SUMMARY_PROMPT = """Write a resume professional summary for a {job_title} role.
Requirements:
- 3-4 sentences maximum
- Highlight experience, but leave room for growth
- Missing 1-2 skills from the job posting (realistic gaps)
- Use varied phrasing: "led X", "worked on Y", "responsible for Z"
- Include one slightly awkward phrasing or grammar quirk (naturalistic)
- Avoid generic buzzwords; be specific about 1-2 achievements

Job posting: {job_posting}
Years of experience: {years_exp}"""

    WORK_EXPERIENCE_PROMPT = """Generate 3 work experience entries for a {job_title} candidate.
Requirements per entry:
- Job title, company, dates (realistic timeline gaps allowed)
- 2-3 bullet points per role
- Mix of relevant and semi-relevant responsibilities
- Vary bullet point length and specificity
- Include metrics in only 1-2 bullets (not all)
- Sound authentic: "Helped improve X" not "Led transformational Y"

Target role: {target_role}
Career trajectory: {career_path}"""

    SKILLS_PROMPT = """List relevant skills for a {job_title} candidate.
Requirements:
- 10-15 skills total
- Include 60% matching job requirements, 40% adjacent/transferable skills
- Vary proficiency levels: expert in 3-4, intermediate in rest
- Include 1-2 outdated or declining skills (realistic)
- Format as simple list, not keyword-stuffed

Job posting: {job_posting}"""