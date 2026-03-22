"""
─────────────────────────────────────────────────────────
1. Coverage uniformity (the "LLM normal curve")
   A fraudster's profile covers ALL JD requirements at suspiciously
   similar similarity scores (uniform distribution).
   A genuine profile covers some requirements strongly, misses others
   entirely, and has a long tail of unmatched skills.

2. Semantic mirroring vs. genuine overlap
   A fabricated profile paraphrases JD requirements back at the model.
   The embedding similarity is high but the profile adds no independent
   semantic content. A genuine profile has high similarity on a subset
   of requirements PLUS a cluster of skills that point away from the JD.

3. Idiosyncrasy (the "off-JD experience signal")
    A fabricated profile has no skills that fall outside the JD's scope —
    it was generated to match the JD, so it has no independent texture.
    A genuine profile includes some skills that don't closely match any
    JD requirement, reflecting real experience beyond the JD's narrow ask.
_________________________________________________________
    Distinguishes between two legitimate reasons a profile might
    closely match a job description:

      Type A — Fabrication: profile was generated/invented to mirror JD.
               High alignment + shallow substance + no independent texture.

      Type B — Genuine tailoring: real experience reframed in JD vocabulary.
               High alignment + operational depth + idiosyncratic evidence
               outside the JD's scope.

    The fraud signal is only escalated when HIGH ALIGNMENT co-occurs with
    LOW SUBSTANCE. High alignment with high substance is a well-qualified
    candidate who did their homework — not a fraud signal.
"""

import re
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer


@dataclass
class SkillTaxonomyResult:
    # fraud_signal:             float
    coverage_score:           float      # fraction of JD requirements matched
    idiosyncrasy_score:       float      # profile skills outside JD scope
    semantic_mirror_score:    float      # profile just paraphrases JD (high = bad)
    matched_requirements:     List[dict] # each JD req with best profile match
    unmatched_requirements:   List[str]  # JD reqs with no strong profile match
    independent_skills:       List[str]  # profile skills outside JD scope
    similarity_distribution:  dict       # stats on the similarity score distribution
    classification:           str        # "fabricated" | "tailored" | "genuine" | "weak"


# ── Main scorer ───────────────────────────────────────────────────────────────

class SkillTaxonomyScorer:
    """
    Embedding-based skill coverage scorer.

    Parameters
    ----------
    match_threshold : float
        Minimum cosine similarity to count a profile skill as matching
        a JD requirement. 0.55 is a reasonable default for this model —
        high enough to require genuine semantic overlap, low enough to
        handle natural paraphrasing.

    strong_match_threshold : float
        Above this threshold, a match is considered "strong" — the
        candidate demonstrably has this specific skill. Used for
        distinguishing deep coverage from surface coverage.
    """
    _MIN_TOKENS = 3
    _MAX_TOKENS = 30

    # Sections in a profile that contain skills
    _PROFILE_SECTIONS = re.compile(
        r'(?:skills?|technologies|tools|experience|expertise|'
        r'proficiencies|competencies|stack)[:\s]*',
        re.IGNORECASE
    )

    # Noise to strip before splitting
    _NOISE = re.compile(r'[\u2022\u2023\u25e6\u2043\u2219•◦▪▸►]')


    def __init__(
        self,
        match_threshold: float = 0.55,
        strong_match_threshold: float = 0.72,
        profile_text: str = "",
        jd_text: str = "",
    ):
        self.match_threshold        = match_threshold
        self.strong_match_threshold = strong_match_threshold
        self.model = SentenceTransformer("Nashhz/SBERT_KFOLD_JobDescriptions_Skills_UserPortfolios")

        self.profile_text = profile_text
        self.jd_text = jd_text

        self.profile_skills   = self.extract_profile_skills(profile_text)
        self.jd_requirements  = self.extract_jd_requirements(jd_text)

    def score(self) -> SkillTaxonomyResult:
        """
        Parameters
        ----------
        profile_skills   : list of skill strings extracted from the profile
                           e.g. ["5 years PyTorch", "built Kafka pipelines",
                                 "MLflow experiment tracking", ...]
        jd_requirements  : list of requirement strings extracted from the JD
                           e.g. ["experience with feature stores",
                                 "MLOps and model serving", ...]
        profile_text     : optional full profile text for supplementary
                           semantic mirror detection

        Returns
        -------
        SkillTaxonomyResult with fraud_signal in [0, 1]
        """

        profile_skills   = self.profile_skills
        jd_requirements = self.jd_requirements
        profile_text = self.profile_text

        model = self.model

        # ── Encode everything ─────────────────────────────────────
        profile_embs = model.encode(profile_skills,    normalize_embeddings=True)
        jd_embs      = model.encode(jd_requirements,   normalize_embeddings=True)

        # similarity[i, j] = similarity between profile_skills[i] and jd_requirements[j]
        # Shape: (n_profile_skills, n_jd_requirements)
        sim_matrix = model.similarity(profile_embs, jd_embs).numpy()

        # ── Signal 1: Coverage and uniformity ─────────────────────
        coverage, matched, unmatched = self._coverage_signals(
            sim_matrix, profile_skills, jd_requirements
        )

        # ── Signal 2: Idiosyncrasy — skills outside JD scope ─────
        idiosyncrasy, independent_skills = self._idiosyncrasy_signal(
            sim_matrix, profile_skills
        )

        # ── Signal 3: Semantic mirroring ──────────────────────────
        mirror_score = self._semantic_mirror_signal(
            sim_matrix, profile_embs, jd_embs,
            profile_text, jd_requirements, model
        )

        # ── Similarity distribution stats ─────────────────────────
        # The max similarity each JD requirement gets from any profile skill
        best_match_per_req = sim_matrix.max(axis=0)
        sim_dist = {
            "mean":   round(float(np.mean(best_match_per_req)), 4),
            "std":    round(float(np.std(best_match_per_req)), 4),
            "min":    round(float(np.min(best_match_per_req)), 4),
            "max":    round(float(np.max(best_match_per_req)), 4),
            "cv":     round(
                float(np.std(best_match_per_req) / max(np.mean(best_match_per_req), 1e-6)),
                4
            ),
        }

        # ── Composite fraud signal ────────────────────────────────
        #
        # Fraud indicators (increase signal):
        #   high coverage_uniformity  — suspiciously even coverage of all JD reqs
        #   high mirror_score         — profile just paraphrases JD
        #   low idiosyncrasy          — no skills outside JD scope
        #
        # Genuine indicators (decrease signal):
        #   high idiosyncrasy         — real skills beyond what JD asked for
        #   low sim_dist["cv"]        — all areas matched equally is suspicious
        #                               but caught by uniformity already
        #
        # fraud_signal = float(np.clip(
        #   + 0.40 * mirror_score
        #   + 0.40 * (1.0 - idiosyncrasy)    # low idiosyncrasy → fraud
        #   + 0.20 * coverage,               # very high coverage is mildly suspicious
        #     0.0, 1.0
        # ))

        classification = self._classify(
            coverage, idiosyncrasy, mirror_score
        )

        return SkillTaxonomyResult(
            # fraud_signal             = round(fraud_signal, 4),
            coverage_score           = round(coverage, 4),
            idiosyncrasy_score       = round(idiosyncrasy, 4),
            semantic_mirror_score    = round(mirror_score, 4),
            matched_requirements     = matched,
            unmatched_requirements   = unmatched,
            independent_skills       = independent_skills,
            similarity_distribution  = sim_dist,
            classification           = classification,
        )

    # ── Signal implementations ────────────────────────────────────────────────

    def _coverage_signals(
        self,
        sim_matrix:      np.ndarray,    # (n_profile, n_jd)
        profile_skills:  List[str],
        jd_requirements: List[str],
    ):
        """
        Coverage: what fraction of JD requirements does the profile match?
        Uniformity: are those matches suspiciously even?

        Genuine pattern: strong matches on some requirements (0.8+), weak
        or absent on others, producing HIGH variance in best-match scores.

        LLM pattern: moderate matches across ALL requirements — the
        profile paraphrases every JD bullet at ~0.65 similarity.
        This produces LOW variance = high uniformity = fraud signal.
        """
        # Best profile skill match for each JD requirement
        best_per_req = sim_matrix.max(axis=0)   # shape: (n_jd,)

        # Coverage: fraction of JD reqs with at least one match above threshold
        matched_mask = best_per_req >= self.match_threshold
        coverage     = float(matched_mask.mean())

        # Uniformity: CV of best-match scores (low CV = uniform = suspicious)
        cv        = np.std(best_per_req) / max(np.mean(best_per_req), 1e-6)
        # Invert and normalize: low CV → high uniformity score → high fraud signal

        # Build matched/unmatched lists with details
        matched   = []
        unmatched = []
        for j, req in enumerate(jd_requirements):
            best_sim = float(best_per_req[j])
            best_idx = int(sim_matrix[:, j].argmax())
            if best_sim >= self.match_threshold:
                matched.append({
                    "requirement":     req,
                    "matched_skill":   profile_skills[best_idx],
                    "similarity":      round(best_sim, 4),
                    "strength":        "strong" if best_sim >= self.strong_match_threshold
                                       else "moderate",
                })
            else:
                unmatched.append(req)

        return coverage, matched, unmatched

    def _idiosyncrasy_signal(
        self,
        sim_matrix:     np.ndarray,   # (n_profile, n_jd)
        profile_skills: List[str],
    ):
        """
        Idiosyncrasy: what fraction of profile skills have NO strong
        match to any JD requirement?

        These are the "off-JD tools" — evidence of real experience
        beyond what the JD asked for. Fabricators have none of these
        because they generated the profile against the JD.
        """
        # Best JD match for each profile skill
        best_per_skill = sim_matrix.max(axis=1)   # shape: (n_profile,)

        # Skills that don't match any JD requirement well
        independent_mask   = best_per_skill < self.match_threshold
        idiosyncrasy_score = float(independent_mask.mean())
        independent_skills = [
            profile_skills[i]
            for i in range(len(profile_skills))
            if independent_mask[i]
        ]

        return idiosyncrasy_score, independent_skills

    def _semantic_mirror_signal(
        self,
        sim_matrix:      np.ndarray,
        profile_embs:    np.ndarray,
        jd_embs:         np.ndarray,
        profile_text:    Optional[str],
        jd_requirements: List[str],
        model,
    ) -> float:
        """
        Semantic mirror: does the profile embedding sit suspiciously
        close to the JD embedding in the shared semantic space?

        A fabricated profile is essentially a restatement of the JD —
        its centroid in embedding space will be very close to the JD
        centroid. A genuine profile has its own semantic center of gravity
        that partially overlaps with the JD but is not identical.

        We measure this as the cosine similarity between the mean
        profile-skill embedding and the mean JD-requirement embedding.
        """
        profile_centroid = profile_embs.mean(axis=0, keepdims=True)
        jd_centroid      = jd_embs.mean(axis=0, keepdims=True)

        centroid_sim = float(
            model.similarity(profile_centroid, jd_centroid)[0][0]
        )

        # Also check: if full profile text provided, embed it and compare
        # directly against JD requirements centroid
        text_sim = 0.0
        if profile_text:
            text_emb = model.encode([profile_text], normalize_embeddings=True)
            text_sim  = float(model.similarity(text_emb, jd_centroid)[0][0])

        mirror_score = (
            0.6 * centroid_sim + 0.4 * text_sim
            if profile_text else centroid_sim
        )

        # Scale: centroid similarity of 0.9+ is strongly suspicious
        # 0.5 is expected for a well-qualified genuine candidate
        # Normalize: map [0.5, 0.95] → [0, 1]
        normalized = np.clip((mirror_score - 0.5) / 0.45, 0.0, 1.0)
        return float(normalized)

    # ── Classification ────────────────────────────────────────────────────────

    @staticmethod
    def _classify(
        coverage:     float,
        idiosyncrasy: float,
        mirror:       float,
    ) -> str:
        """
        Four-quadrant classification using the most discriminating signals.

               High uniformity        Low uniformity
               ┌───────────────────┬──────────────────┐
        Low    │   fabricated       │    weak_fit      │
        idiosy │  (JD mirror, no    │  (partial match, │
               │   own experience)  │   no depth)      │
               ├───────────────────┼──────────────────┤
        High   │   tailored         │    genuine       │
        idiosy │  (JD-optimized +   │  (real exp, not  │
               │   own experience)  │   JD-optimized)  │
               └───────────────────┴──────────────────┘
        """

        return idiosyncrasy, mirror, coverage


# ── Skill / requirement extractor ────────────────────────────────────────────

    def extract_profile_skills(self, profile_text: str) -> List[str]:
        return self._extract(profile_text)

    def extract_jd_requirements(self, jd_text: str) -> List[str]:
        return self._extract(jd_text)

    def _extract(self, text: str) -> List[str]:
        # Normalise bullets and split on natural boundaries
        text   = self._NOISE.sub('\n', text)
        chunks = re.split(r'[\n;,]|(?<=[a-z])\.\s', text)

        results = []
        for chunk in chunks:
            chunk = chunk.strip().strip('.-')
            tokens = chunk.split()
            if self._MIN_TOKENS <= len(tokens) <= self._MAX_TOKENS:
                results.append(chunk)

        return results

class SkillExtractor:
    """
    Converts raw profile text and JD text into lists of skill/requirement
    strings that the scorer can embed.

    Strategy: split on bullet points, newlines, and semicolons, then
    filter to segments that contain actionable skill content.
    This gives the model short, focused strings to embed rather than
    long paragraphs — the model was trained on skill-length phrases.
    """





# ── Integration helper ────────────────────────────────────────────────────────

def score_from_text(
    profile_text: str,
    jd_text:      str,
    match_threshold:        float = 0.55,
    strong_match_threshold: float = 0.72,
) -> SkillTaxonomyResult:
    """
    Convenience wrapper: takes raw text strings, extracts skills/requirements,
    runs the scorer, and returns the result.

    This is the function to call from CandidateEnv or a pipeline.
    """
    extractor = SkillExtractor()
 
    profile_skills   = extractor.extract_profile_skills(profile_text)
    jd_requirements  = extractor.extract_jd_requirements(jd_text)

    return scorer.score(profile_skills, jd_requirements, profile_text)


# ── Usage ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    jd = """
    We are looking for a Senior ML Platform Engineer.
    Requirements:
    - Experience with feature stores such as Feast or Tecton
    - Model serving with Triton Inference Server or TorchServe
    - MLflow or Weights & Biases for experiment tracking
    - Kubeflow or Airflow for pipeline orchestration
    - Fine-tuning large language models using LoRA or RLHF techniques
    - Strong Kubernetes and distributed systems background
    - Real-time streaming experience with Kafka or Flink
    """

    # Case 1: Fabricated — mirrors JD exactly, no independent skills
    fabricated_profile = """
    Senior ML Platform Engineer with expertise in:
    - Feature store implementation using Feast and Tecton
    - Real-time model serving via Triton Inference Server
    - Experiment tracking with MLflow and Weights & Biases
    - Pipeline orchestration using Kubeflow and Airflow
    - LLM fine-tuning with LoRA and RLHF techniques
    - Kubernetes cluster management and distributed systems
    - Real-time streaming with Kafka and Flink
    """

    # Case 2: Genuine — strong in some areas, uses different tools in others
    genuine_profile = """
    ML infrastructure engineer, 4 years building model serving systems.
    Deep experience with Kafka consumer pipelines and Spark for batch ETL.
    Built our internal feature platform on Redis and Postgres before Feast
    was production-ready — evaluated Feast 0.28, ran into silent partition
    drops on schema changes, decided to wait.
    Run inference through TorchServe; Triton was overkill for our model sizes.
    Airflow for all orchestration — we use a mono-DAG pattern.
    Some LoRA fine-tuning on an internal NER model last quarter.
    Strong Python, SQL. Haven't used Kubeflow or RLHF in production.
    """

    # Case 3: Tailored — genuine experience rewritten in JD language
    tailored_profile = """
    Senior ML platform engineer with hands-on experience in feature store
    design (built a custom store; evaluating Feast for migration), model
    serving infrastructure (TorchServe in production, familiar with Triton),
    pipeline orchestration (Airflow, exploring Kubeflow), and LLM fine-tuning
    (LoRA adapters on internal classification models).
    Additional: Redis caching layer, Spark batch processing, custom Kafka
    consumers for real-time feature computation, Prometheus + Grafana.
    """

    


    for label, profile in [
        ("Fabricated",         fabricated_profile),
        ("Genuine",            genuine_profile),
        ("Tailored (genuine)", tailored_profile),
    ]:
        result = scorer    = SkillTaxonomyScorer(0.55, 0.72, profile, jd).score()
        print(f"\n{'='*58}")
        print(f"  {label}")
        print(f"{'='*58}")
        # print(f"  fraud_signal:          {result.fraud_signal}")
        print(f"  classification:        {result.classification}")
        print(f"  coverage_score:        {result.coverage_score}")
        print(f"  idiosyncrasy_score:    {result.idiosyncrasy_score}  ← high = genuine off-JD experience")
        print(f"  semantic_mirror_score: {result.semantic_mirror_score}  ← high = profile paraphrases JD")
        print(f"  sim distribution:      {result.similarity_distribution}")
        print(f"  unmatched reqs:        {result.unmatched_requirements[:2]}")
        print(f"  independent skills:    {result.independent_skills[:3]}")