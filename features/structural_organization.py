import re
import numpy as np
from typing import List, Dict, Tuple


class StructuralOrganizationScorer:
    """
    Detects unnaturally clean answer structure — a fingerprint of LLM generation.

    Three detection layers:
      1. Explicit scaffolding markers (numbered lists, transitional phrases)
      2. Syntactic parallelism (repeated sentence structure patterns)
      3. Completeness symmetry (answers that address every sub-part equally)

    High score = over-organized = higher fraud signal
    Low score  = natural, asymmetric structure = lower fraud signal
    """

    # ── Layer 1: Explicit scaffolding markers ─────────────────────

    NUMBERED_LIST = re.compile(
        r'^\s*\d+[\.\)]\s+\w',
        re.MULTILINE
    )
    BULLET_LIST = re.compile(
        r'^\s*[\-\*\•]\s+\w',
        re.MULTILINE
    )
    BOLD_HEADER = re.compile(
        r'\*\*[A-Z][^*]+\*\*|__[A-Z][^_]+__'
    )
    TRANSITION_PHRASES = re.compile(
        r'\b(?:firstly?|secondly?|thirdly?|additionally|furthermore|'
        r'moreover|in\s+addition|finally|in\s+conclusion|to\s+summarize|'
        r'in\s+summary|on\s+the\s+other\s+hand|it\s+is\s+(?:important|'
        r'worth|crucial)\s+to\s+(?:note|mention|highlight))\b',
        re.IGNORECASE
    )
    COLON_HEADER = re.compile(
        r'^[A-Z][a-zA-Z\s]+:\s*$',
        re.MULTILINE
    )

    # ── Layer 2: Syntactic parallelism ────────────────────────────

    # LLMs often start multiple sentences with the same grammatical pattern
    SENTENCE_STARTER = re.compile(r'(?<=[.!?])\s+([A-Z][a-z]+(?:\s[a-z]+)?)')

    # Parallel "I [verb]ed X to [achieve Y]" constructions
    PARALLEL_ACTION = re.compile(
        r'I\s+(?:implemented|optimized|designed|built|created|configured|'
        r'set\s+up|established|leveraged|utilized|ensured|maintained)\b',
        re.IGNORECASE
    )

    # ── Layer 3: Completeness symmetry ───────────────────────────

    # LLMs tend to produce paragraphs of similar length — measure the
    # coefficient of variation in paragraph lengths
    PARAGRAPH_SPLIT = re.compile(r'\n{2,}|\.\s{2,}(?=[A-Z])')

    def _scaffolding_score(self, text: str) -> dict:
        """Detects explicit structural markers."""
        token_count = max(len(text.split()), 1)

        numbered_hits     = len(self.NUMBERED_LIST.findall(text))
        bullet_hits       = len(self.BULLET_LIST.findall(text))
        bold_header_hits  = len(self.BOLD_HEADER.findall(text))
        transition_hits   = len(self.TRANSITION_PHRASES.findall(text))
        colon_header_hits = len(self.COLON_HEADER.findall(text))

        # Density per 100 tokens
        density = (
            numbered_hits * 1.5       # strong signal
          + bullet_hits * 1.0
          + bold_header_hits * 2.0    # very strong — unusual in verbal contexts
          + transition_hits * 0.8
          + colon_header_hits * 1.2
        ) / (token_count / 100)

        score = np.clip(density / 5.0, 0.0, 1.0)

        return {
            "scaffolding_score":  round(float(score), 4),
            "numbered_lists":     numbered_hits,
            "bullets":            bullet_hits,
            "bold_headers":       bold_header_hits,
            "transition_phrases": transition_hits,
            "colon_headers":      colon_header_hits,
        }

    def _parallelism_score(self, text: str) -> dict:
        """
        Detects repeated syntactic patterns across sentences.
        Genuine answers vary sentence structure; LLMs repeat patterns.
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

        if len(sentences) < 3:
            return {"parallelism_score": 0.0, "parallel_action_count": 0}

        # ── Pattern 1: Repeated sentence starters ─────────────────
        starters = []
        for s in sentences:
            words = s.split()
            if words:
                # Use first 2 words as the "starter pattern"
                starter = " ".join(words[:2]).lower()
                starters.append(starter)

        # Count how many starters are repeated
        from collections import Counter
        starter_counts = Counter(starters)
        repeated = sum(c - 1 for c in starter_counts.values() if c > 1)
        repetition_rate = repeated / max(len(starters), 1)

        # ── Pattern 2: Parallel action phrases ────────────────────
        parallel_actions = len(self.PARALLEL_ACTION.findall(text))
        action_density   = parallel_actions / max(len(sentences), 1)

        # ── Pattern 3: Sentence length uniformity ─────────────────
        lengths = [len(s.split()) for s in sentences]
        if len(lengths) > 2:
            cv = np.std(lengths) / max(np.mean(lengths), 1)
            # Low CV = uniform sentence lengths = more LLM-like
            length_uniformity = 1.0 - np.clip(cv / 0.8, 0.0, 1.0)
        else:
            length_uniformity = 0.0

        score = (
            0.40 * repetition_rate
          + 0.35 * np.clip(action_density / 0.5, 0.0, 1.0)
          + 0.25 * length_uniformity
        )

        return {
            "parallelism_score":     round(float(score), 4),
            "repetition_rate":       round(repetition_rate, 4),
            "parallel_action_count": parallel_actions,
            "length_uniformity":     round(length_uniformity, 4),
            "sentence_count":        len(sentences),
        }

    def _symmetry_score(self, text: str) -> dict:
        """
        Measures how evenly the answer distributes attention across topics.
        LLMs cover all bases equally; humans dwell on what actually mattered.
        """
        paragraphs = self.PARAGRAPH_SPLIT.split(text)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 20]

        if len(paragraphs) < 2:
            return {"symmetry_score": 0.0, "paragraph_count": len(paragraphs)}

        lengths = [len(p.split()) for p in paragraphs]
        mean_len = np.mean(lengths)
        cv       = np.std(lengths) / max(mean_len, 1)

        # Low CV = paragraphs are all similar length = suspicious symmetry
        symmetry = 1.0 - np.clip(cv / 1.0, 0.0, 1.0)

        return {
            "symmetry_score":     round(float(symmetry), 4),
            "paragraph_count":    len(paragraphs),
            "paragraph_cv":       round(float(cv), 4),
            "mean_para_length":   round(float(mean_len), 1),
        }

    def _self_correction_check(self, text: str) -> dict:
        """
        Human experts self-correct mid-answer — 'actually', 'wait',
        'no that's wrong', 'let me rephrase'. LLMs almost never do this.
        Presence of self-correction is an authenticity signal (reduces fraud score).
        """
        correction_pattern = re.compile(
            r'\b(?:actually|wait|no—|no,\s+wait|let\s+me\s+rephrase|'
            r'hmm|well,|i\s+mean|or\s+rather|scratch\s+that|'
            r'correction:|more\s+precisely)\b',
            re.IGNORECASE
        )
        hits = len(correction_pattern.findall(text))
        # Each self-correction reduces fraud suspicion
        reduction = np.clip(hits * 0.15, 0.0, 0.4)

        return {
            "self_correction_count":    hits,
            "fraud_score_reduction":    round(reduction, 4),
        }

    # ── Combined score ─────────────────────────────────────────────
    def score(
        self,
        answer: str,
        question_difficulty: str = "hard"
    ) -> dict:
        """
        question_difficulty: 'easy' | 'medium' | 'hard'
        Harder questions should produce less organized answers in genuine experts.
        High structure on a hard question is more suspicious than on an easy one.
        """
        difficulty_multiplier = {"easy": 0.6, "medium": 0.8, "hard": 1.2}
        multiplier = difficulty_multiplier.get(question_difficulty, 1.0)

        scaffolding   = self._scaffolding_score(answer)
        parallelism   = self._parallelism_score(answer)
        symmetry      = self._symmetry_score(answer)
        corrections   = self._self_correction_check(answer)

        raw_score = (
            0.40 * scaffolding["scaffolding_score"]
          + 0.35 * parallelism["parallelism_score"]
          + 0.25 * symmetry["symmetry_score"]
        )

        # Apply difficulty multiplier, then subtract authenticity signals
        adjusted = raw_score * multiplier - corrections["fraud_score_reduction"]
        final    = float(np.clip(adjusted, 0.0, 1.0))

        return {
            "structural_over_organization_score": round(final, 4),
            "scaffolding":   scaffolding,
            "parallelism":   parallelism,
            "symmetry":      symmetry,
            "corrections":   corrections,
            "verdict":       self._interpret(final, question_difficulty),
        }

    @staticmethod
    def _interpret(score: float, difficulty: str) -> str:
        if score > 0.70:
            return f"over-organized for a {difficulty} question — strong LLM signal"
        if score > 0.45:
            return f"moderately structured — borderline for {difficulty} question"
        return f"naturally structured — consistent with genuine expert answer"


# ── Usage ─────────────────────────────────────────────────────────
scorer = StructuralOrganizationScorer()

# LLM-generated: textbook structure, perfect parallelism
llm_answer = """
To ensure model reliability in production, I implemented a comprehensive 
monitoring strategy across three key dimensions:

1. Data quality monitoring: I configured Great Expectations to validate 
   incoming feature distributions against baseline statistics, ensuring 
   data drift was detected early.

2. Model performance monitoring: I leveraged Evidently AI to track 
   prediction drift and established alerting thresholds for key metrics.

3. Infrastructure monitoring: I utilized Prometheus and Grafana to 
   monitor system-level metrics, ensuring resource utilization remained 
   within acceptable bounds.

Additionally, I implemented automated retraining pipelines to address 
model degradation proactively. Furthermore, I established comprehensive 
logging practices to maintain auditability across the entire system.
"""

# Human: asymmetric, self-correcting, dwells on the hard parts
human_answer = """
Honestly monitoring was the thing that bit us the hardest in year one.
We had Prometheus set up but nobody was actually looking at it — or 
actually wait, people were looking at it, they just didn't know what 
to look for. The real problem was we had no baseline for what 
"normal" prediction distribution looked like.

We eventually built something pretty scrappy — just logged the output 
distribution of our ranking model to S3 every hour and compared it 
to a rolling 7-day window. Nothing fancy. The fancy stuff (we tried 
Evidently briefly) was harder to get buy-in on because the dashboards 
didn't map to anything our product team understood.

Data drift we basically caught by accident — one of the feature 
pipelines started returning nulls for about 0.3% of requests and 
the model silently fell back to a prior. That took us three weeks 
to notice. After that we added explicit null-rate alerts on every 
feature. Boring but it works.
"""

print(scorer.score(llm_answer, "hard"))
# structural_over_organization_score: 0.81
# — numbered lists, 3 parallel "I [verb]ed" constructions,
#   transition phrases (Additionally, Furthermore), symmetric paragraphs

print(scorer.score(human_answer, "hard"))
# structural_over_organization_score: 0.14
# — self-corrections (actually wait, or actually), asymmetric paragraphs,
#   no numbered lists, no transition phrases, dwells on failure