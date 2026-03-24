import re
import numpy as np
from typing import Dict, List, Tuple, Optional


# ── Lazy model registry (process-wide — hf_pipeline_cache) ───────────────────

def _load(key: str):
    try:
        from .hf_pipeline_cache import get_sentence_transformer, get_transformers_pipeline
    except ImportError:
        from hf_pipeline_cache import get_sentence_transformer, get_transformers_pipeline

    loaders = {
        "zeroshot": lambda: get_transformers_pipeline(
            "zero-shot-classification",
            "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
        ),
        "embedder": lambda: get_sentence_transformer("all-MiniLM-L6-v2"),
    }
    return loaders[key]()


# ── Sentence splitter (unchanged) ─────────────────────────────────────────────

def _split_sentences(text: str, min_len: int = 8) -> List[str]:
    sentences = re.split(r'[.!?\n•]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > min_len]


# ── Zero-shot label sets ──────────────────────────────────────────────────────

# Replace SOFT_SKILL_MARKERS — describes interpersonal/people function
_SOFT_LABELS = [
    "describes interpersonal, collaborative, or people-leadership activity",
    "describes a technical, clinical, analytical, or domain-specific task",
]

# Replace TECHNICAL_MARKERS — describes domain-specific functional work
_TECH_LABELS = [
    "describes a technical, clinical, analytical, or domain-specific task",
    "describes interpersonal, collaborative, or management activity",
]

# Fallback regex (retained for when model unavailable) — broadened cross-domain
_SOFT_FALLBACK = re.compile(
    r'\b(?:'
    # Universal leadership / people
    r'mentored?|coached?|onboarded?|supervised?|managed\s+(?:a\s+team|staff|people)|'
    r'hired|recruited|retained|developed\s+(?:team|staff|talent)|'
    # Universal collaboration
    r'collaborated?|partnered\s+with|liaised?|coordinated\s+with|'
    r'facilitated?|mediated?|negotiated?\s+(?:with|between|across)|'
    r'presented?\s+(?:to|findings|results)\s+(?:the|senior|leadership)|'
    # Universal communication / influence
    r'communicated?\s+(?:with|to|across)|influenced?\s+without|'
    r'built\s+(?:consensus|trust|relationships?|rapport)|'
    r'conflict\s+resolution|de-escalated?|aligned\s+(?:stakeholders?|teams?)|'
    # Universal navigation of ambiguity
    r'navigated?\s+(?:ambiguity|competing|uncertainty|complexity)|'
    r'drove?\s+(?:alignment|adoption|change)|'
    # Domain-specific soft skills (finance, legal, healthcare, marketing)
    r'advised?\s+(?:clients?|patients?|counsel|the\s+board)|'
    r'counselled?|patient\s+advocacy|family\s+communication|'
    r'client\s+relationship|account\s+management|stakeholder\s+engagement'
    r')\b',
    re.IGNORECASE
)

_TECH_FALLBACK = re.compile(
    r'\b(?:'
    # Software
    r'implemented|deployed|architected|built|designed|optimized|'
    r'configured|migrated|developed|integrated|automated|scaled|'
    # Finance / analysis
    r'modelled?|valued?|analysed?|analyzed?|constructed?|structured?|'
    r'underwrote?|hedged?|executed?\s+(?:trades?|transactions?)|priced?|'
    # Healthcare / clinical
    r'administered|diagnosed?|treated?|assessed?|monitored?|intubated?|'
    r'prescribed?|performed?\s+(?:surgery|procedure|assessment)|'
    # Marketing / creative
    r'launched?|campaigned?|tested?|optimised?|A/B\s+tested?|'
    r'drafted?|authored?|produced?|created?\s+(?:content|assets?|copy)|'
    # Legal
    r'drafted?|filed?|argued?|litigated?|negotiated?\s+(?:contracts?|terms?)|'
    r'reviewed?\s+(?:contracts?|agreements?|documents?)'
    r')\b',
    re.IGNORECASE
)

_FIRST_PERSON = re.compile(r'\bI\b')


# ── Main scorer ───────────────────────────────────────────────────────────────

class ProfileVoiceConsistencyScorer:
    """
    Measures tonal and structural uniformity across profile sections.

    High uniformity  = every section sounds like the same robot → fraud signal
    Natural variation = sections have different registers, soft-skill ratios,
                        sentence length variance → genuine profile

    Works for any professional domain: software, finance, healthcare,
    marketing, legal, consulting, research.

    Five signals (4 original + 1 new):
      1. Soft-skill ratio uniformity across sections  (zero-shot, was regex)
      2. Sentence length variance within sections     (unchanged)
      3. Domain-task density uniformity across sections (zero-shot, was regex)
      4. First-person pronoun distribution            (unchanged)
      5. Cross-section semantic voice clustering      (NEW — SBERT embeddings)
    """

    def __init__(self, use_models: bool = True):
        """
        Parameters
        ----------
        use_models : bool
            If True, uses zero-shot + SBERT. If False, falls back to
            the broadened regex patterns for speed/offline use.
        """
        self.use_models = use_models

    # ── Public API ────────────────────────────────────────────────────────────

    def score(self, sections: Dict[str, str]) -> dict:
        """
        Parameters
        ----------
        sections : dict mapping section name → text
            e.g. {
              "summary":    "Results-driven engineer...",
              "role_1":     "Led the migration of...",
              "role_2":     "Joined as the founding...",
              "skills":     "Python, Kubernetes, dbt...",
            }
        """
        if len(sections) < 2:
            return {
                "profile_voice_fraud_score": 0.4,
                "verdict": "too few sections to compare",
            }

        section_profiles = {
            name: self._profile_section(text)
            for name, text in sections.items()
            if text.strip()
        }

        # ── Signal 1: Soft-skill ratio uniformity ─────────────────
        # Low CV across sections = every section has same soft/technical mix
        # = suspicious (LLM generates all sections with same template)
        soft_ratios        = [p["soft_skill_ratio"] for p in section_profiles.values()]
        soft_cv            = _cv(soft_ratios)
        uniformity_penalty = max(0.0, 0.6 - soft_cv) * 0.5

        # Near-zero soft skills across ALL sections = strong flag
        # (profile reads as pure keyword dump with no human texture)
        zero_soft_penalty = 0.35 if float(np.mean(soft_ratios)) < 0.05 else 0.0

        # ── Signal 2: Sentence length variance (unchanged) ────────
        sent_len_vars    = [p["sentence_length_cv"] for p in section_profiles.values()]
        mean_sent_var    = float(np.mean(sent_len_vars))
        sentence_penalty = max(0.0, 0.4 - mean_sent_var) * 0.4

        # ── Signal 3: Domain-task density uniformity ──────────────
        # (Replaces "ATS keyword density" — same concept, model-backed)
        tech_densities = [p["technical_density"] for p in section_profiles.values()]
        tech_cv        = _cv(tech_densities)
        ats_penalty    = max(0.0, 0.5 - tech_cv) * 0.25

        # ── Signal 4: First-person pronoun distribution (unchanged)
        fp_rates        = [p["first_person_rate"] for p in section_profiles.values()]
        fp_cv           = _cv(fp_rates)
        pronoun_penalty = max(0.0, 0.4 - fp_cv) * 0.20

        # ── Signal 5: Cross-section semantic voice clustering (new)
        # Tightly clustered sections = one homogeneous voice = LLM
        # Dispersed sections = different registers over time = genuine
        voice_cluster_penalty = self._voice_cluster_penalty(
            list(sections.values())
        )

        final = float(np.clip(
            uniformity_penalty
          + zero_soft_penalty
          + sentence_penalty
          + ats_penalty
          + pronoun_penalty
          + voice_cluster_penalty,
            0.0, 1.0
        ))

        return {
            "profile_voice_fraud_score": round(final, 4),
            "soft_skill_mean_rate":      round(float(np.mean(soft_ratios)), 4),
            "soft_skill_cv":             round(float(soft_cv), 4),
            "sentence_uniformity":       round(mean_sent_var, 4),
            "tech_density_cv":           round(float(tech_cv), 4),
            "voice_cluster_penalty":     round(voice_cluster_penalty, 4),
            "section_profiles":          section_profiles,
            "verdict":                   self._interpret(final),
        }

    # ── Section profiler ──────────────────────────────────────────────────────

    def _profile_section(self, text: str) -> dict:
        """
        Computes per-section metrics. Soft/technical ratio now uses
        zero-shot classification; everything else is unchanged.
        """
        sentences = _split_sentences(text)
        tokens    = text.split()
        n_tok     = max(len(tokens), 1)

        # Soft vs. technical ratio via zero-shot (or regex fallback)
        soft_ratio, tech_density = self._soft_tech_ratio(text, sentences)

        # First-person rate (unchanged)
        fp_hits = len(_FIRST_PERSON.findall(text))

        # Sentence length CV (unchanged)
        sent_lens = [len(s.split()) for s in sentences]
        sent_cv   = (
            float(np.std(sent_lens) / max(np.mean(sent_lens), 1))
            if sent_lens else 0.0
        )

        return {
            "soft_skill_ratio":   round(soft_ratio, 4),
            "technical_density":  round(tech_density, 4),
            "first_person_rate":  round(fp_hits / (n_tok / 100), 4),
            "sentence_length_cv": round(sent_cv, 3),
            "sentence_count":     len(sentences),
        }

    def _soft_tech_ratio(
        self, text: str, sentences: List[str]
    ) -> Tuple[float, float]:
        """
        Replaces SOFT_SKILL_MARKERS and TECHNICAL_MARKERS regex.

        Returns (soft_skill_ratio, technical_density).

        Zero-shot strategy: classify each sentence on the soft/technical
        axis. Average the "soft skill" probability across sentences for
        the ratio; average "technical" probability for density.

        Both labels describe narrative function across all domains —
        the model has no knowledge of what "soft skill" or "technical"
        means for any specific profession.
        """
        if not sentences:
            return 0.0, 0.0

        if not self.use_models:
            return self._soft_tech_ratio_fallback(text)

        try:
            zs = _load("zeroshot")

            soft_scores = []
            tech_scores = []

            for sent in sentences[:12]:   # cap per section for speed
                if len(sent.split()) < 4:
                    continue

                # Soft skill score for this sentence
                s_result = zs(sent, _SOFT_LABELS, multi_label=False)
                soft_scores.append(
                    s_result["scores"][s_result["labels"].index(_SOFT_LABELS[0])]
                )

                # Technical/domain score for this sentence
                t_result = zs(sent, _TECH_LABELS, multi_label=False)
                tech_scores.append(
                    t_result["scores"][t_result["labels"].index(_TECH_LABELS[0])]
                )

            if not soft_scores:
                return self._soft_tech_ratio_fallback(text)

            # soft_ratio: mean probability of "soft skill" interpretation
            soft_ratio   = float(np.mean(soft_scores))

            # technical_density: scale to approximate the original per-100-token metric
            # (original was count/n_tok*100; here we use probability × sentence_density)
            n_tok        = max(len(text.split()), 1)
            sent_density = len(sentences) / (n_tok / 100)   # sentences per 100 tokens
            tech_density = float(np.mean(tech_scores)) * sent_density

            return soft_ratio, tech_density

        except Exception:
            return self._soft_tech_ratio_fallback(text)

    @staticmethod
    def _soft_tech_ratio_fallback(text: str) -> Tuple[float, float]:
        """
        Regex fallback when models are unavailable or use_models=False.
        Uses the broadened cross-domain marker patterns.
        """
        tokens     = text.split()
        n_tok      = max(len(tokens), 1)
        soft_hits  = len(_SOFT_FALLBACK.findall(text))
        tech_hits  = len(_TECH_FALLBACK.findall(text))
        total      = max(soft_hits + tech_hits, 1)

        soft_ratio   = soft_hits / total
        tech_density = tech_hits / (n_tok / 100)
        return soft_ratio, tech_density

    # ── Signal 5: Cross-section semantic voice clustering (new) ──────────────

    def _voice_cluster_penalty(self, section_texts: List[str]) -> float:
        """
        Measures how semantically similar all sections are to each other.

        Genuine profiles: sections written at different career stages and
        in different contexts → naturally varied semantic content → low
        mean pairwise similarity, high variance in pairwise scores.

        LLM-generated profiles: all sections generated in one pass with
        a consistent template → tight semantic cluster → high mean
        pairwise similarity, low variance.

        Penalty:
          high mean similarity + low variance → robot voice → high penalty
          lower mean or high variance → natural voice → low penalty
        """
        if len(section_texts) < 2 or not self.use_models:
            return 0.0

        try:
            from sklearn.metrics.pairwise import cosine_similarity
            from itertools import combinations

            embedder   = _load("embedder")
            embeddings = embedder.encode(section_texts, normalize_embeddings=True)
            pairs      = list(combinations(range(len(section_texts)), 2))

            sims = [
                float(cosine_similarity([embeddings[i]], [embeddings[j]])[0][0])
                for i, j in pairs
            ]

            mean_sim = float(np.mean(sims))
            var_sim  = float(np.var(sims))

            # High mean + low variance = all sections sound the same = suspicious
            # Map: mean_sim ~0.90 → penalty up to 0.20
            #      var_sim  ~0.00 → no variance reduction
            raw_penalty = max(0.0, mean_sim - 0.65) * 0.5    # kicks in above 0.65 sim
            variance_reduction = np.clip(var_sim * 10, 0.0, 0.15)  # high variance forgives

            return float(np.clip(raw_penalty - variance_reduction, 0.0, 0.20))

        except Exception:
            return 0.0

    # ── Interpretation ────────────────────────────────────────────────────────

    @staticmethod
    def _interpret(score: float) -> str:
        if score > 0.70:
            return "unnaturally uniform voice — ATS-optimised or single-pass LLM generation"
        if score > 0.45:
            return "low variation across sections — missing human texture, probe further"
        return "natural voice variation — sections have different registers and emphasis"


# ── Coefficient of variation helper ──────────────────────────────────────────

def _cv(values: List[float]) -> float:
    arr  = np.array(values)
    mean = float(np.mean(arr))
    return float(np.std(arr) / max(mean, 0.01))


# ── Usage: three domains ──────────────────────────────────────────────────────

if __name__ == "__main__":
    scorer = ProfileVoiceConsistencyScorer(use_models=True)

    # ── Software engineer ──────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Software engineer")
    print("═" * 60)

    sw_fabricated = {
        "summary": (
            "Senior ML Platform Engineer with deep expertise in distributed "
            "feature stores, model serving infrastructure, and MLOps best "
            "practices. Passionate about cross-functional collaboration and "
            "delivering scalable, reliable systems."
        ),
        "role_1": (
            "Implemented distributed feature store using Feast. Deployed "
            "real-time model serving via Triton. Integrated MLflow for "
            "experiment tracking. Optimized pipeline throughput by 40%. "
            "Collaborated with cross-functional teams to deliver results."
        ),
        "role_2": (
            "Architected scalable data pipelines using Kafka and Spark. "
            "Configured Kubernetes clusters for model serving. Migrated "
            "legacy systems to cloud infrastructure. Aligned stakeholders "
            "across engineering and product teams to drive adoption."
        ),
        "skills": (
            "PyTorch, TensorFlow, Feast, Triton, MLflow, Kubeflow, Airflow, "
            "Kafka, Spark, Kubernetes, Docker, Terraform, Python, SQL, Scala."
        ),
    }
    sw_genuine = {
        "summary": (
            "ML infra engineer. Spent most of the last four years making "
            "model serving fast and not terrible to operate. Strong opinions "
            "about replay buffers and weak opinions about Kubernetes operators."
        ),
        "role_1": (
            "Built our online feature system on Redis before Feast was stable "
            "— evaluated 0.28, hit silent partition drops on schema changes, "
            "waited. TorchServe for inference; Triton was overkill at our "
            "scale. Sarah on the research team found the memory fragmentation "
            "bug that was killing p99. Airflow for orchestration, mono-DAG "
            "pattern. Haven't used Kubeflow in production."
        ),
        "role_2": (
            "Joined as the third engineer. Wore a lot of hats — half my time "
            "was Kafka consumers, half was arguing with the CEO about whether "
            "we needed Spark yet (we didn't). Ended up building a pretty "
            "scrappy but fast feature pipeline on Postgres + Redis. "
            "Left when we ran out of runway."
        ),
        "skills": (
            "Python, SQL, some Rust. Kafka, Redis, Postgres, Airflow, "
            "TorchServe. Comfortable with Kubernetes — not an expert. "
            "Weak on Spark, never touched Kubeflow."
        ),
    }
    for label, sections in [("Fabricated", sw_fabricated), ("Genuine", sw_genuine)]:
        r = scorer.score(sections)
        print(f"\n  [{label}]")
        print(f"    fraud_score:          {r['profile_voice_fraud_score']}")
        print(f"    soft_skill_cv:        {r['soft_skill_cv']}")
        print(f"    tech_density_cv:      {r['tech_density_cv']}")
        print(f"    sentence_uniformity:  {r['sentence_uniformity']}")
        print(f"    voice_cluster_penalty:{r['voice_cluster_penalty']}")
        print(f"    verdict: {r['verdict']}")

    # ── Financial analyst ──────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Financial analyst")
    print("═" * 60)

    fin_fabricated = {
        "summary": (
            "Senior Financial Analyst with expertise in DCF valuation, M&A "
            "transaction advisory, and revenue forecasting. Passionate about "
            "driving alignment across stakeholders and delivering impactful "
            "financial insights in fast-paced environments."
        ),
        "role_1": (
            "Constructed DCF and LBO models for M&A transactions. Analysed "
            "comparable company data using Bloomberg and FactSet. Presented "
            "findings to senior leadership. Collaborated with cross-functional "
            "teams to deliver strategic recommendations on time."
        ),
        "role_2": (
            "Developed revenue forecasting models and managed budget variance "
            "analysis. Coordinated with business units to align financial "
            "targets. Communicated insights to C-suite stakeholders. "
            "Optimised reporting processes to improve efficiency by 30%."
        ),
        "skills": "Excel, Bloomberg, FactSet, SQL, Python, Tableau, PowerPoint.",
    }
    fin_genuine = {
        "summary": (
            "FP&A at Series B SaaS. Rebuilt the three-statement model from "
            "scratch after the previous one had hardcoded assumptions from 2021. "
            "Caught a $400K budget overrun in Q3 because of it."
        ),
        "role_1": (
            "Did the financial diligence package for our acquisition of a "
            "smaller competitor. Deal closed at 4.2x revenue — below our "
            "initial 5x offer after we found deferred revenue recognition "
            "issues that the target's auditors had glossed over. The credit "
            "committee pushed back hard; I spent two weeks in the data room "
            "rebuilding their revenue model from the contract-level data."
        ),
        "role_2": (
            "Standard analyst programme. Lots of pitchbooks. The interesting "
            "part was the restructuring work — we had a retail client in "
            "Chapter 11 and I built the liquidation analysis that ended up "
            "being used in the 363 sale process. Learned more about asset "
            "recovery waterfall mechanics in three months than I had in "
            "two years of undergrad finance."
        ),
        "skills": (
            "Excel (strong). Bloomberg, FactSet for data. SQL — can write "
            "it, not an expert. Python for automation. Haven't used Tableau "
            "seriously; we lived in PowerPoint and Google Slides."
        ),
    }
    for label, sections in [("Fabricated", fin_fabricated), ("Genuine", fin_genuine)]:
        r = scorer.score(sections)
        print(f"\n  [{label}]")
        print(f"    fraud_score:          {r['profile_voice_fraud_score']}")
        print(f"    soft_skill_cv:        {r['soft_skill_cv']}")
        print(f"    tech_density_cv:      {r['tech_density_cv']}")
        print(f"    voice_cluster_penalty:{r['voice_cluster_penalty']}")
        print(f"    verdict: {r['verdict']}")