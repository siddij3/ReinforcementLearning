"""
Model choices and why
─────────────────────
1. tanfiona/unicausal-tok-cls-baseline
   Fine-tuned on 7 causal NLP datasets (CausalTimeBank, EventStoryLine,
   SemEval 2010 Task 8, BECauSE 2.0, etc.). Classifies tokens as
   CAUSE or EFFECT spans. Domain-agnostic — trained on news, finance,
   biomedical, and general text. Replaces CAUSAL_CONNECTIVES regex.

2. MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33
   Fine-tuned on 33 NLI datasets. Enables zero-shot classification with
   arbitrary labels — "describes a specific situation", "vague and generic",
   "contains a concrete outcome". No domain assumptions baked in.
   Replaces SITUATION_MARKERS and GENERIC_PATTERNS regex.

3. Jean-Baptiste/roberta-large-ner-english
   NER model covering PERSON, ORG, GPE, DATE, MONEY, PERCENT, QUANTITY,
   ORDINAL, CARDINAL. Domain-agnostic quantity and entity detection.
   Replaces RESULT_ANCHORS (which was hard-coded to tech metrics like
   p99, rps, ms). A financial analyst mentioning "$2.3M cost reduction"
   and a doctor mentioning "mortality rate dropped 12%" both score well.

4. cross-encoder/nli-deberta-v3-small
   Fast cross-encoder NLI model. Used to check whether a claimed
   outcome is entailed by the described action — catches metric/output
   mismatches without any domain-specific rules.
   Replaces MISMATCH_PATTERNS regex.
"""

import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from functools import lru_cache


# ── Lazy model registry ───────────────────────────────────────────────────────

_MODELS: Dict = {}

def _load(key: str):
    """Loads a model once and caches it. Call only when needed."""
    if key in _MODELS:
        return _MODELS[key]
    try:
        from .hub_auth import ensure_hf_token_for_downloads
    except ImportError:
        from hub_auth import ensure_hf_token_for_downloads
    ensure_hf_token_for_downloads()

    from transformers import pipeline

    loaders = {
        "causal": lambda: pipeline(
            "token-classification",
            model="tanfiona/unicausal-tok-cls-baseline",
            aggregation_strategy="simple",
        ),
        "zeroshot": lambda: pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
        ),
        "ner": lambda: pipeline(
            "ner",
            model="Jean-Baptiste/roberta-large-ner-english",
            aggregation_strategy="simple",
        ),
        "nli": lambda: pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small",
        )
    }

    _MODELS[key] = loaders[key]()
    return _MODELS[key]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class CausalityResult:
    causal_span_score:      float   # density of detected cause/effect spans
    # situation_score:        float   # zero-shot: describes a specific situation
    # specificity_score:      float   # zero-shot: concrete vs. vague language
    result_entity_score:    float   # NER: quantities and outcomes present
    coherence_score:        float   # NLI: actions entail claimed outcomes


# ── Main scorer ───────────────────────────────────────────────────────────────

class NarrativeCausalityScorer:
    """
    Domain-agnostic narrative causality scorer.

    Works for any professional domain — software, finance, healthcare,
    marketing, legal, operations — by replacing regex patterns with
    models that reason about causality and specificity semantically.

    Parameters
    ----------
    causal_threshold : float
        Minimum score for a causal span to be counted (0–1)
    """

    # Zero-shot label sets — these work across all domains
    _SITUATION_LABELS = [
        "describes a specific situation or context",
        "makes a generic or abstract statement",
    ]
    _SPECIFICITY_LABELS = [
        "concrete, specific, and detailed",
        "vague, generic, and non-specific",
    ]
    _OUTCOME_LABELS = [
        "describes a measurable outcome or result",
        "describes an activity without a clear outcome",
    ]

    # Light regex still useful for structural splitting — not domain-specific
    _SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
    _MIN_SENT_LEN   = 8    # tokens

    def __init__(
        self,
        causal_threshold: float = 0.70,
    ):
        self.causal_threshold = causal_threshold

    def score(self, answer: str, question: str = "") -> CausalityResult:
        """
        Score a single answer for narrative causality richness.

        Parameters
        ----------
        answer   : the candidate's answer text
        question : the question asked (used for coherence checking)
        """
        sentences = self._split_sentences(answer)

        # ── Signal 1: Causal span detection (UniCausal) ───────────
        causal_score, causal_detail = self._causal_span_score(answer)
        # ── Signal 2: Result entity detection (NER) ───────────────
        result_score, ner_detail = self._result_entity_score(answer)

        # ── Signal 3: Coherence / mismatch (NLI) ─────────────────
        coherence_score, nli_detail = self._coherence_score(sentences)

        # ── Composite fraud signal ────────────────────────────────
        #
        # High causal spans      → low fraud (genuine narrative)
        # High situation score   → low fraud (specific context described)
        # High specificity       → low fraud (concrete language)
        # High result entities   → low fraud (measurable outcomes cited)
        # High coherence         → low fraud (actions → outcomes make sense)
        #
        # # Fraud signal = inverse of narrative richness
        # narrative_richness = np.clip(
        #     0.30 * causal_score
        #   + 0.20 * situation_score
        #   + 0.20 * specificity_score
        #   + 0.15 * result_score
        #   + 0.15 * coherence_score,
        #     0.0, 1.0
        # )


        # fraud_signal  = float(np.clip(1.0 - narrative_richness, 0.0, 1.0))

        return CausalityResult(
            causal_span_score     = round(causal_score, 4),
            result_entity_score   = round(result_score, 4),
            coherence_score       = round(coherence_score, 4),
            # detail = {
            #     "causal":     causal_detail,
            #     "ner":        ner_detail,
            #     "nli":        nli_detail,
            # },
        )

    # ── Signal implementations ────────────────────────────────────────────────

    def _causal_span_score(self, text: str) -> Tuple[float, dict]:
        """
        Uses tanfiona/unicausal-tok-cls-baseline to find CAUSE and EFFECT
        token spans. Returns a density score (spans per 100 tokens).

        A rich answer has multiple cause→effect pairs. A keyword list has none.
        Works for any domain — "the campaign underperformed because..." scores
        the same way as "the service degraded because...".
        """
        try:
            causal_pipe = _load("causal")
            spans       = causal_pipe(text[:512])   # model max length

            # Filter to confident spans only
            valid = [
                s for s in spans
                if s["score"] >= self.causal_threshold
                and s["entity_group"] in ("CAUSE", "EFFECT", "B-C", "B-E")
            ]

            tokens        = text.split()
            n_tok         = max(len(tokens), 1)
            cause_count   = sum(1 for s in valid if "C" in s["entity_group"])
            effect_count  = sum(1 for s in valid if "E" in s["entity_group"])

            # Need both cause AND effect spans for a complete chain
            complete_chains = min(cause_count, effect_count)
            density         = complete_chains / (n_tok / 100)
            score           = min(density / 2.0, 1.0)   # 2 chains/100 tokens → score 1.0

            return score, {
                "cause_spans":    cause_count,
                "effect_spans":   effect_count,
                "complete_chains":complete_chains,
                "examples":       [s["word"] for s in valid[:4]],
            }

        except Exception as e:
            # Fallback: light regex for causal connectives (domain-neutral set)
            connective_pattern = re.compile(
                r'\b(because|since|therefore|consequently|as\s+a\s+result|'
                r'which\s+led|this\s+caused|resulting\s+in|so\s+that|'
                r'due\s+to|owing\s+to|hence|thus)\b',
                re.IGNORECASE
            )
            hits  = len(connective_pattern.findall(text))
            score = min(hits / 3.0, 1.0)
            return score, {"fallback": True, "hits": hits, "error": str(e)}

    def _situation_score(self, sentences: List[str]) -> Tuple[float, dict]:
        """
        Zero-shot classifies each sentence as "describes a specific situation"
        vs "makes a generic statement". Domain-agnostic — works because the
        labels describe narrative function, not content.
        """
        try:
            zs_pipe   = _load("zeroshot")
            scores    = []
            examples  = []

            for sent in sentences[:8]:   # cap for speed
                if len(sent.split()) < self._MIN_SENT_LEN:
                    continue
                result   = zs_pipe(sent, self._SITUATION_LABELS, multi_label=False)
                # Score for the "specific situation" label
                specific_score = result["scores"][
                    result["labels"].index("describes a specific situation or context")
                ]
                scores.append(specific_score)
                if specific_score > 0.65:
                    examples.append(sent[:60])

            mean_score = float(np.mean(scores)) if scores else 0.0
            return mean_score, {
                "sentences_scored": len(scores),
                "high_specificity_examples": examples[:2],
            }

        except Exception as e:
            # Fallback: count first-person past-tense situation markers
            pattern = re.compile(
                r'\b(we\s+(found|noticed|saw|hit|encountered|discovered|realized)|'
                r'i\s+(found|noticed|saw|had|encountered|realized)|'
                r'when\s+(we|i|the|our))\b',
                re.IGNORECASE
            )
            hits  = len(pattern.findall(" ".join(sentences)))
            score = min(hits / 2.0, 1.0)
            return score, {"fallback": True, "hits": hits, "error": str(e)}

    def _specificity_score(self, sentences: List[str]) -> Tuple[float, dict]:
        """
        Zero-shot classifies language as "concrete and specific" vs "vague
        and generic". Catches corporate filler language across all domains:
        """
        try:
            zs_pipe  = _load("zeroshot")
            scores   = []
            generic  = []

            for sent in sentences[:8]:
                if len(sent.split()) < self._MIN_SENT_LEN:
                    continue
                result = zs_pipe(sent, self._SPECIFICITY_LABELS, multi_label=False)
                concrete_score = result["scores"][
                    result["labels"].index("concrete, specific, and detailed")
                ]
                scores.append(concrete_score)
                if concrete_score < 0.40:
                    generic.append(sent[:60])

            mean_score = float(np.mean(scores)) if scores else 0.0
            return mean_score, {
                "sentences_scored": len(scores),
                "generic_examples": generic[:2],
            }

        except Exception as e:
            # Fallback: penalize known cross-domain filler verbs
            filler = re.compile(
                r'\b(leveraged|utilized|spearheaded|championed|drove|'
                r'ensured|delivered|facilitated|managed|oversaw|executed|'
                r'collaborated\s+on|worked\s+on|helped\s+with)\b',
                re.IGNORECASE
            )
            n_tok   = max(len(" ".join(sentences).split()), 1)
            hits    = len(filler.findall(" ".join(sentences)))
            penalty = min(hits / (n_tok / 100) / 3.0, 1.0)
            score   = max(1.0 - penalty, 0.0)
            return score, {"fallback": True, "filler_hits": hits, "error": str(e)}

    def _result_entity_score(self, text: str) -> Tuple[float, dict]:
        """
        NER-based result detection. Looks for MONEY, PERCENT, QUANTITY,
        CARDINAL, DATE entities that indicate measurable outcomes.

        Domain examples this handles:
          - Software: "reduced p99 by 40%" → PERCENT entity
          - Finance:  "grew revenue $2.3M" → MONEY entity
          - Healthcare:"mortality rate fell 12 points" → PERCENT + CARDINAL
          - Marketing: "increased CTR from 1.2% to 3.4%" → PERCENT ×2
          - Legal:     "settled for $4.5M in 8 months" → MONEY + DATE

        All of these fail the old tech-specific regex but pass NER detection.
        """
        try:
            ner_pipe = _load("ner")
            entities = ner_pipe(text[:512])

            # Entity types that signal measurable, specific outcomes
            result_types = {"MONEY", "PERCENT", "QUANTITY", "CARDINAL", "ORDINAL"}
            anchor_types = {"ORG", "PERSON", "GPE", "DATE", "TIME"}

            result_ents = [e for e in entities if e["entity_group"] in result_types]
            anchor_ents = [e for e in entities if e["entity_group"] in anchor_types]

            n_tok = max(len(text.split()), 1)

            # Density: result entities per 100 tokens
            result_density = len(result_ents) / (n_tok / 100)
            anchor_density = len(anchor_ents) / (n_tok / 100)

            # Score: result entities are the primary signal;
            # anchor entities (names, dates, places) add authenticity
            score = min(
                0.70 * min(result_density / 2.0, 1.0)
              + 0.30 * min(anchor_density / 3.0, 1.0),
                1.0
            )

            return float(score), {
                "result_entities":  [(e["word"], e["entity_group"]) for e in result_ents[:5]],
                "anchor_entities":  [(e["word"], e["entity_group"]) for e in anchor_ents[:5]],
                "result_count":     len(result_ents),
                "anchor_count":     len(anchor_ents),
            }

        except Exception as e:
            # Fallback: generic numeric pattern (domain-neutral)
            numeric = re.compile(
                r'\b\d+(?:[.,]\d+)?\s*(?:%|percent|x\b|times|'
                r'\$|£|€|million|billion|thousand|k\b|'
                r'days?|weeks?|months?|years?|hours?)\b',
                re.IGNORECASE
            )
            hits  = len(numeric.findall(text))
            score = min(hits / 3.0, 1.0)
            return score, {"fallback": True, "numeric_hits": hits, "error": str(e)}

    def _coherence_score(self, sentences: List[str]) -> Tuple[float, dict]:
        """
        NLI-based coherence checking. For each pair of adjacent sentences
        where one describes an action and the next describes an outcome,
        checks whether the outcome is entailed by (consistent with) the action.
        """
        try:
            nli_pipe = _load("nli")

            # Check consecutive sentence pairs
            coherence_scores = []
            incoherent_pairs = []

            for i in range(len(sentences) - 1):
                premise    = sentences[i]
                hypothesis = sentences[i + 1]

                if len(premise.split()) < 5 or len(hypothesis.split()) < 5:
                    continue

                # NLI: does the premise entail the hypothesis?
                result = nli_pipe(f"{premise} [SEP] {hypothesis}")
                label  = result[0]["label"].upper()
                conf   = result[0]["score"]

                # "CONTRADICTION" between adjacent sentences is the mismatch signal
                if label == "CONTRADICTION" and conf > 0.70:
                    incoherent_pairs.append({
                        "premise":    premise[:60],
                        "hypothesis": hypothesis[:60],
                        "confidence": round(conf, 3),
                    })

            # Score: fewer contradictions = higher coherence = lower fraud
            total_pairs = max(len(sentences) - 1, 1)
            contradiction_rate = len(incoherent_pairs) / total_pairs
            coherence          = 1.0 - contradiction_rate

            return float(coherence), {
                "pairs_checked":     total_pairs,
                "contradictions":    len(incoherent_pairs),
                "incoherent_pairs":  incoherent_pairs[:2],
            }

        except Exception as e:
            return 0.5, {"fallback": True, "error": str(e)}

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> List[str]:
        sentences = self._SENTENCE_SPLIT.split(text.strip())
        return [
            s.strip() for s in sentences
            if len(s.strip().split()) >= self._MIN_SENT_LEN
        ]

# ── Usage ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    scorer = NarrativeCausalityScorer()

    # ── Software engineer bullet (original domain) ─────────────────
    sw_genuine = """
    Our consumer lag spiked to 40 minutes on the payments topic — turned out
    max.poll.records was set too high and our Protobuf deserialization was
    hitting a decode error on 0.1% of messages, causing the whole poll loop
    to stall. Dropped max.poll.records from 500 to 50, routed bad messages
    to a dead-letter topic, and p99 lag dropped from 40 minutes to 8 seconds.
    """

    # ── Financial analyst bullet (new domain) ──────────────────────
    finance_genuine = """
    Our APAC portfolio was underperforming by 340bps versus benchmark because
    the FX hedging ratio hadn't been updated since Q1 and the dollar had moved
    12% since then. Rebuilt the hedge ratio model to update monthly on realized
    vol rather than implied vol — tracking error dropped from 380bps to 90bps
    over the next two quarters.
    """

    # ── Marketing bullet (new domain) ─────────────────────────────
    marketing_genuine = """
    Our email open rates fell from 28% to 11% over six weeks — the sending
    domain had been flagged for spam after a batch campaign hit a purchased
    list. Migrated to a subdomain, rebuilt the list against confirmed opt-ins,
    and warmed the new domain over 8 weeks. Open rate recovered to 24% and
    deliverability score went from 61 to 94.
    """

    # ── Generic fabricated bullet (any domain) ────────────────────
    fabricated = """
    Leveraged cross-functional collaboration to drive alignment across
    stakeholders and deliver impactful results. Utilized best practices
    to ensure robust and scalable outcomes that exceeded expectations.
    """

    for label, text in [
        ("SW engineer (genuine)",        sw_genuine),
        ("Financial analyst (genuine)",  finance_genuine),
        ("Marketing manager (genuine)",  marketing_genuine),
        ("Fabricated (any domain)",      fabricated),
    ]:
        result = scorer.score(text)
        print(f"\n{'─'*56}")
        print(f"  {label}")
        print(f"{'─'*56}")
        print(f"  causal_spans:      {result.causal_span_score}")
        print(f"  situation:         {result.situation_score}")
        print(f"  specificity:       {result.specificity_score}")
        print(f"  result_entities:   {result.result_entity_score}")
        print(f"  coherence:         {result.coherence_score}")