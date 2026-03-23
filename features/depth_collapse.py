"""
depth_collapse_delta_scorer.py  (domain-agnostic rewrite)
Models used
───────────
- MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33
  Scores answers on operational depth, situation specificity, and
  concreteness without any domain-specific labels.

- Jean-Baptiste/roberta-large-ner-english
  Replaces the concrete_pattern regex with domain-agnostic entity
  extraction for measurable claims.

- all-MiniLM-L6-v2
  Measures semantic distance between the L1 question framing and the
  L2 answer — a genuine expert's L2 answer goes further from the
  question's surface into territory the question didn't explicitly ask
  about. An LLM paraphrases the question back.

New scoring architecture
────────────────────────
score_answer() now takes (answer, question) instead of (answer, rubric).
The question is used to:
  1. Generate zero-shot classification axes specific to the topic
  2. Measure how far the answer goes semantically beyond the question
  3. Provide context for NLI-based depth detection

compute_delta() is unchanged in interface — still takes L1/L2 answer pairs.
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


# ── Lazy model registry ───────────────────────────────────────────────────────

_MODELS: Dict = {}

def _load(key: str):
    if key in _MODELS:
        return _MODELS[key]
    try:
        from .hub_auth import ensure_hf_token_for_downloads
    except ImportError:
        from hub_auth import ensure_hf_token_for_downloads
    ensure_hf_token_for_downloads()
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer

    loaders = {
        "zeroshot": lambda: pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
        ),
        "ner": lambda: pipeline(
            "ner",
            model="Jean-Baptiste/roberta-large-ner-english",
            aggregation_strategy="simple",
        ),
        "embedder": lambda: SentenceTransformer("all-MiniLM-L6-v2"),
    }
    _MODELS[key] = loaders[key]()
    return _MODELS[key]


# ── Sentence splitter ─────────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def _sentences(text: str, min_tok: int = 5) -> List[str]:
    return [
        s.strip() for s in _SENT_SPLIT.split(text.strip())
        if len(s.strip().split()) >= min_tok
    ]


# ── Domain-neutral hedge pattern (already generalised) ────────────────────────

_HEDGE_PATTERN = re.compile(
    r"\bi'?m not (sure|certain|familiar|experienced)\b"
    r"|\bwe (didn't|never|don't) (use|have|do|work)\b"
    r"|\bi (haven't|don't|didn't|wouldn't) (used?|know|work|have)\b"
    r"|\b(outside|beyond) (my|our) (experience|expertise|area|scope)\b"
    r"|\bi('d| would) (have to|need to) (check|look|verify|confirm)\b"
    r"|\bcan't (remember|recall) (the |exactly |off)?\b"
    r"|\bif i (recall|remember) correctly\b"
    r"|\bnot (something|an area|a tool) i('ve| have)\b",
    re.IGNORECASE
)

# ── Zero-shot label axes ──────────────────────────────────────────────────────
# Describe narrative FUNCTION, not domain content — work for any professional role

_OPERATIONAL_LABELS = [
    "operational, practice-based, and grounded in direct experience",
    "theoretical, conceptual, or textbook-level",
]
_SITUATION_LABELS = [
    "describes a specific personal experience or situation",
    "makes a general or abstract statement",
]
_SPECIFICITY_LABELS = [
    "concrete, detailed, and specific",
    "vague, generic, and non-specific",
]
_MECHANISM_LABELS = [
    "explains the mechanism or root cause of something",
    "describes what happened without explaining why",
]


# ── Main scorer ───────────────────────────────────────────────────────────────

class DepthCollapseDeltaScorer:
    """
    Computes the quality drop between an L1 (surface) and L2 (deep) answer.
    Works for any professional domain without requiring pre-written rubrics.

    The question text replaces the rubric — models read the question to
    understand what "operational depth" means for that topic.

    delta > 0.5  → severe collapse — surface knowledge only, strong fraud signal
    delta 0.2–0.5 → moderate collapse — probe further
    delta < 0.2  → consistent depth — likely genuine expertise
    """

    def score_answer(self, answer: str, question: str = "") -> dict:
        """
        Score a single answer for operational depth.

        Parameters
        ----------
        answer   : the candidate's answer text
        question : the question that was asked (used for semantic distance
                   and contextual zero-shot scoring). Optional but recommended.

        Returns
        -------
        dict with quality_score [0,1] and supporting signals
        """
        tokens    = answer.split()
        n_tok     = max(len(tokens), 1)

        # ── Signal 1: Zero-shot operational depth ─────────────────
        # Replaces operational_indicators keyword matching.
        # Classifies whether the answer reads as practice-based or textbook.
        zs_scores = self._zeroshot_depth(answer)

        # ── Signal 2: Concrete claims via NER ─────────────────────
        # Replaces concrete_pattern regex (ms, GB, version numbers, acronyms).
        # Extracts MONEY, PERCENT, QUANTITY, CARDINAL, ORG — any domain.
        claim_score, claim_detail = self._ner_claim_score(answer)

        # ── Signal 3: Semantic distance from question ─────────────
        # A deep answer goes semantically further from the question's
        # surface framing. LLMs paraphrase the question back; genuine
        # experts volunteer adjacent knowledge the question didn't ask for.
        semantic_distance = self._semantic_distance(answer, question) if question else 0.5

        # ── Signal 4: Length adequacy ─────────────────────────────
        # Kept — length is a weak but real signal. 60 tokens is
        # the minimum for a substantive answer regardless of domain.
        length_score = min(n_tok / 80, 1.0)

        # ── Signal 5: Hedging ─────────────────────────────────────
        # Unchanged — epistemic hedging is domain-neutral
        has_hedge = bool(_HEDGE_PATTERN.search(answer))

        # ── Generic penalty via zero-shot ─────────────────────────
        # Replaces generic_red_flags keyword list.
        # "Textbook-level" classification score is the penalty.
        generic_penalty = min(
            (1.0 - zs_scores["operational"]) * 0.40,
            0.40
        )

        # ── Composite quality score ───────────────────────────────
        quality = float(np.clip(
            0.30 * zs_scores["operational"]
          + 0.20 * zs_scores["situation"]
          + 0.15 * zs_scores["specificity"]
          + 0.10 * zs_scores["mechanism"]
          + 0.10 * claim_score
          + 0.08 * semantic_distance
          + 0.07 * length_score
          + 0.05 * float(has_hedge)
          - generic_penalty,
            0.0, 1.0
        ))

        return {
            "quality_score":       round(quality, 4),
            "zs_operational":      round(zs_scores["operational"], 4),
            "zs_situation":        round(zs_scores["situation"], 4),
            "zs_specificity":      round(zs_scores["specificity"], 4),
            "zs_mechanism":        round(zs_scores["mechanism"], 4),
            "claim_score":         round(claim_score, 4),
            "claim_entities":      claim_detail,
            "semantic_distance":   round(semantic_distance, 4),
            "length_tokens":       n_tok,
            "has_hedge":           has_hedge,
            "generic_penalty":     round(generic_penalty, 4),
        }

    def compute_delta(
        self,
        answer_l1: str,
        question_l1: str,
        answer_l2: str,
        question_l2: str,
    ) -> dict:
        """
        Compute depth collapse between surface (L1) and deep (L2) answers.

        Parameters
        ----------
        answer_l1   : answer to the surface question
        question_l1 : the surface question asked
        answer_l2   : answer to the deeper follow-up question
        question_l2 : the deeper follow-up question asked

        Returns
        -------
        dict with depth_collapse_delta and per-answer breakdowns
        """
        result_l1 = self.score_answer(answer_l1, question_l1)
        result_l2 = self.score_answer(answer_l2, question_l2)

        delta = result_l1["quality_score"] - result_l2["quality_score"]

        return {
            "depth_collapse_delta": round(np.clip(delta, 0.0, 1.0), 4),
            "l1_score":             result_l1["quality_score"],
            "l2_score":             result_l2["quality_score"],
            "l1_detail":            result_l1,
            "l2_detail":            result_l2,
            "verdict":              self._interpret(delta),
        }

    # ── Signal implementations ────────────────────────────────────────────────

    def _zeroshot_depth(self, answer: str) -> dict:
        """
        Classifies the answer on four depth axes using zero-shot NLI.

        All four label sets describe narrative function, not domain content:
          - operational vs theoretical
          - specific situation vs general statement
          - concrete vs vague
          - mechanism-explaining vs event-describing

        """
        try:
            zs = _load("zeroshot")

            # Use the first 400 tokens for speed (captures the answer's substance)
            text = " ".join(answer.split()[:400])

            def _first_label_score(labels: List[str]) -> float:
                result = zs(text, labels, multi_label=False)
                return float(result["scores"][result["labels"].index(labels[0])])

            return {
                "operational": _first_label_score(_OPERATIONAL_LABELS),
                "situation":   _first_label_score(_SITUATION_LABELS),
                "specificity": _first_label_score(_SPECIFICITY_LABELS),
                "mechanism":   _first_label_score(_MECHANISM_LABELS),
            }

        except Exception as e:
            # Regex fallback — domain-neutral signals
            return self._zeroshot_fallback(answer, str(e))

    @staticmethod
    def _zeroshot_fallback(answer: str, error: str = "") -> dict:
        """
        Lightweight regex fallback when the zero-shot model is unavailable.
        Uses domain-neutral structural signals rather than keyword lists.
        """
        # Causal connectives → mechanism explanation
        causal = re.compile(
            r'\b(because|which caused|led to|as a result|so we|'
            r'turned out|the (issue|problem|root cause) was|'
            r'we (noticed|found|realized|hit|ran into))\b',
            re.IGNORECASE
        )
        # First-person past situations → specific experience
        situation = re.compile(
            r'\b(we (had|ran|noticed|found|built|hit)|'
            r'i (had|saw|found|noticed|realized|built|ran))\b',
            re.IGNORECASE
        )
        # Generic filler → penalise
        generic = re.compile(
            r'\b(generally|typically|it is (known|important|recommended)|'
            r'best practice|in theory|one should|you should|'
            r'the (agent|system|model) (learns|optimizes|improves))\b',
            re.IGNORECASE
        )

        n_tok     = max(len(answer.split()), 1)
        causal_d  = min(len(causal.findall(answer)) / (n_tok / 100) / 2.0, 1.0)
        sit_d     = min(len(situation.findall(answer)) / (n_tok / 100) / 2.0, 1.0)
        generic_d = min(len(generic.findall(answer)) / (n_tok / 100) / 2.0, 1.0)

        return {
            "operational": max(0.5 + causal_d * 0.4 - generic_d * 0.4, 0.0),
            "situation":   max(0.4 + sit_d * 0.5, 0.0),
            "specificity": max(0.5 - generic_d * 0.5, 0.0),
            "mechanism":   max(0.4 + causal_d * 0.5, 0.0),
            "_fallback":   True,
            "_error":      error,
        }

    def _ner_claim_score(self, answer: str) -> Tuple[float, List[str]]:
        """
        Replaces the concrete_pattern regex with NER entity extraction.

        Old regex matched: ms, GB, MB, v1.2.3, NullPointerException, OOM
        — exclusively software artifacts.

        NER extracts claim-bearing entities from any domain:
          MONEY    → "$2.3B AUM", "£4.5M settlement", "€800K budget"
          PERCENT  → "12% mortality reduction", "34% open rate", "4.2% CAC"
          QUANTITY → "12-bed ICU", "8 A100s", "3.2M impressions"
          CARDINAL → "team of 8", "reduced by 40", "18 months"
          DATE     → "Q3 2023", "last quarter", "over 18 months"
          ORG      → named employers, clients, counterparties
          PRODUCT  → tools, protocols, instruments (not just software)
        """
        try:
            ner      = _load("ner")
            entities = ner(answer[:512])

            result_types = {"MONEY", "PERCENT", "QUANTITY", "CARDINAL"}
            anchor_types = {"ORG", "PERSON", "GPE", "DATE", "PRODUCT", "EVENT"}

            result_ents = [e for e in entities if e["entity_group"] in result_types]
            anchor_ents = [e for e in entities if e["entity_group"] in anchor_types]

            n_tok = max(len(answer.split()), 1)

            # Result entities carry more weight (measurable outcomes)
            result_density = len(result_ents) / (n_tok / 100)
            anchor_density = len(anchor_ents) / (n_tok / 100)

            score = float(np.clip(
                0.65 * min(result_density / 2.0, 1.0)
              + 0.35 * min(anchor_density / 3.0, 1.0),
                0.0, 1.0
            ))

            readable = [
                f"{e['word']} ({e['entity_group']})"
                for e in (result_ents + anchor_ents)[:8]
            ]
            return score, readable

        except Exception as e:
            # Fallback: domain-neutral numeric pattern
            num_pattern = re.compile(
                r'\b\d+(?:[.,]\d+)?\s*'
                r'(?:%|percent|x\b|times|\$|£|€|M\b|B\b|K\b|'
                r'million|billion|thousand|bps|'
                r'days?|weeks?|months?|years?|hours?|'
                r'beds?|patients?|cases?|accounts?|users?|'
                r'ms\b|rps\b|qps\b|gb\b|mb\b|tb\b)\b',
                re.IGNORECASE
            )
            hits  = len(num_pattern.findall(answer))
            score = min(hits / 3.0, 1.0)
            return score, [f"fallback: {hits} numeric hits | {e}"]

    def _semantic_distance(self, answer: str, question: str) -> float:
        """
        Measures how far semantically the answer goes beyond the question.

        A genuine expert's deep answer introduces concepts and vocabulary
        the question didn't ask about — it volunteers adjacent knowledge.
        An LLM paraphrases the question's own terms back.

        High distance → answer goes beyond question framing → genuine signal
        Low distance  → answer mirrors question vocabulary → LLM-like

        Normalized so that:
          similarity ~0.95 → distance ~0.0  (answer mirrors question)
          similarity ~0.50 → distance ~1.0  (answer diverges significantly)
        """
        try:
            embedder = _load("embedder")
            q_emb    = embedder.encode([question], normalize_embeddings=True)
            a_emb    = embedder.encode([answer],   normalize_embeddings=True)

            from sklearn.metrics.pairwise import cosine_similarity
            sim      = float(cosine_similarity(q_emb, a_emb)[0][0])

            # Normalize: map [0.95, 0.5] → [0.0, 1.0]
            # (high similarity = low distance = bad; low similarity = high distance = good)
            distance = np.clip((0.95 - sim) / 0.45, 0.0, 1.0)
            return float(distance)

        except Exception:
            return 0.5   # neutral on failure

    @staticmethod
    def _interpret(delta: float) -> str:
        if delta > 0.5: return "severe collapse — surface knowledge only, strong fraud signal"
        if delta > 0.2: return "moderate collapse — probe further"
        return "consistent depth — likely genuine expertise"


# ── Usage: four domains ───────────────────────────────────────────────────────

if __name__ == "__main__":
    scorer = DepthCollapseDeltaScorer()

    test_cases = [

        # ── Software / RL engineering (original domain) ───────────
        {
            "domain":  "Software — RL engineer",
            "q_l1":    "Explain how PPO works.",
            "q_l2":    "What happens when GAE lambda is set too high in a partially observable environment?",
            "l1_genuine": """
                PPO uses a clipped surrogate objective — the clip_range parameter
                bounds how far the new policy can move from the old one per update.
                We pair it with GAE at lambda=0.95 to balance bias and variance,
                run 10 epochs over each batch, and use an entropy bonus of 0.01
                to prevent premature convergence.
            """,
            "l2_genuine": """
                If lambda is too high — say 0.99 — in a partially observable env
                you're bootstrapping through observations that don't fully represent
                state, so advantage estimates pick up noise from future steps.
                Practically I saw this as exploding value loss when the env had
                random delays in reward delivery. Dropping lambda to 0.90 and
                normalizing advantages per minibatch fixed it.
            """,
            "l2_fraud": """
                When lambda is too high it can affect the advantage estimation
                process and lead to higher variance in gradient updates. This can
                make training less stable. It is generally recommended to tune
                lambda carefully based on the specific environment.
            """,
        },

        # ── Financial analysis ────────────────────────────────────
        {
            "domain":  "Finance — portfolio analyst",
            "q_l1":    "How do you approach interest rate risk in a fixed income portfolio?",
            "q_l2":    "Walk me through how you'd manage duration mismatch when the yield curve is flattening.",
            "l1_genuine": """
                We manage rate risk primarily through duration targeting. Our
                benchmark is 5.2 years so we run a tracking error budget of
                ±0.8 years. We use a combination of physical bonds and rate
                swaps to get there — swaps are cheaper to adjust quickly when
                our view changes. DV01 exposure is reviewed daily by the desk.
            """,
            "l2_genuine": """
                Flattening curves compress the carry you earn from the long end
                without giving you much capital gain — so a duration extension
                that looked attractive at 50bps 2s10s spread becomes painful
                at 8bps. In 2022 we had a 0.4yr long duration tilt going into
                March. When the curve inverted we closed half of it in the swap
                market rather than selling bonds — the bid-ask on the physicals
                was 12 cents wide that week, the swap was 2bps. Still cost us
                roughly 18bps on the quarter.
            """,
            "l2_fraud": """
                Duration mismatch in a flattening yield curve environment requires
                careful management. Generally speaking, investors should consider
                reducing duration exposure as the curve flattens to minimize
                interest rate risk. It is important to regularly review portfolio
                positioning relative to benchmark duration targets.
            """,
        },

        # ── Healthcare / clinical ─────────────────────────────────
        {
            "domain":  "Healthcare — ICU nurse",
            "q_l1":    "How do you prioritise patients in a busy ICU shift?",
            "q_l2":    "Describe how you'd manage a patient deteriorating into septic shock while you have three other critical patients.",
            "l1_genuine": """
                First thing is a quick safety sweep — I look at vitals on the
                board, flag anything with a MAP below 65 or SpO2 below 90,
                and check the overnight notes for anyone whose trajectory was
                concerning. We use a modified MEWS score every four hours and
                anything above 5 gets escalated to the fellow before I start
                my normal rounding. If I have a fresh post-op and a long-stay
                patient on the same shift I'll front-load care on the post-op.
            """,
            "l2_genuine": """
                Septic shock deterioration — first call is to the attending,
                not the fellow, because you need vasopressor orders immediately.
                While someone else is getting the order I'll hang a litre of
                crystalloid wide open and get a second IV access in. Once I have
                norepinephrine running I need to triage the other three: one of
                them becomes my biggest risk if I'm pulled away. If two of the
                others are stable vented patients their alarms can wait 15
                minutes; if one is a LVAD patient that's a different story —
                I'll call for backup before the situation escalates. The hardest
                part is the family — they see you running and they panic. You
                need 30 seconds to tell them what's happening even when you
                don't have 30 seconds.
            """,
            "l2_fraud": """
                Managing a deteriorating patient alongside multiple critical
                patients requires strong prioritisation skills. It is important
                to assess the most critical patient first and involve the
                multidisciplinary team. Following established sepsis protocols
                and the ABCDE approach ensures systematic care delivery while
                maintaining oversight of other patients.
            """,
        },

        # ── Marketing ─────────────────────────────────────────────
        {
            "domain":  "Marketing — growth manager",
            "q_l1":    "How do you approach paid acquisition strategy?",
            "q_l2":    "Our CAC has increased 60% over 18 months despite flat CPCs. What would you investigate?",
            "l1_genuine": """
                Paid acquisition for us is about portfolio management across
                channels. We run Google Performance Max, Meta Advantage+,
                and LinkedIn Lead Gen — each has a different CAC profile and
                a different point in the funnel where it's most efficient.
                We allocate budget weekly based on a blended target CAC of
                $420 and shift toward whichever channel is running below that.
                Incrementality testing every quarter to sanity-check the
                attribution model.
            """,
            "l2_genuine": """
                Flat CPCs with rising CAC means your conversion rate somewhere
                in the funnel collapsed — the media cost didn't cause this.
                First place I'd look is audience exhaustion: if you're in a
                small TAM and you've been running the same campaigns for 18
                months, you've probably reached everyone susceptible and now
                you're paying for the same people repeatedly. Second is lead
                quality degradation — check whether MQL-to-opportunity rate
                changed, because if sales is closing at the same rate but
                pipeline is thinner, the problem is qualification not volume.
                We had this in 2023: CPCs were actually down 8% but our form
                fill to sales-qualified rate dropped from 22% to 11% because
                we'd broadened match types to hit volume targets. Rolling back
                to phrase match recovered 14 points of conversion rate within
                six weeks.
            """,
            "l2_fraud": """
                Rising CAC despite stable CPCs is a common challenge in digital
                marketing. Generally speaking, marketers should review their
                targeting parameters, ad creative performance, and landing page
                conversion rates. It is recommended to conduct A/B testing
                across different funnel stages and ensure alignment between
                media strategy and sales processes for optimal results.
            """,
        },
    ]

    for case in test_cases:
        print(f"\n{'═'*62}")
        print(f"  {case['domain']}")
        print(f"{'═'*62}")

        result_genuine = scorer.compute_delta(
            case["l1_genuine"], case["q_l1"],
            case["l2_genuine"], case["q_l2"],
        )
        result_fraud = scorer.compute_delta(
            case["l1_genuine"], case["q_l1"],
            case["l2_fraud"],   case["q_l2"],
        )

        for label, result in [("Genuine L2", result_genuine), ("Fraud L2", result_fraud)]:
            print(f"\n  [{label}]")
            print(f"    depth_collapse_delta : {result['depth_collapse_delta']}")
            print(f"    l1_score             : {result['l1_score']}")
            print(f"    l2_score             : {result['l2_score']}")
            print(f"    l2 zs_operational    : {result['l2_detail']['zs_operational']}")
            print(f"    l2 zs_mechanism      : {result['l2_detail']['zs_mechanism']}")
            print(f"    l2 semantic_distance : {result['l2_detail']['semantic_distance']}")
            print(f"    l2 claim_entities    : {result['l2_detail']['claim_entities'][:3]}")
            print(f"    verdict: {result['verdict']}")