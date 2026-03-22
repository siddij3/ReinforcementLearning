"""
cross_answer_consistency_scorer.py  (domain-agnostic rewrite)

What was domain-specific and why
─────────────────────────────────
1. extract_claims — version numbers as the primary claim type
   Patterns like r'Ray 2.6' or r'Python 3.9' only fire for software.
   A finance candidate saying "$2.3B AUM" or a nurse saying "12-bed ICU"
   produce no structured claims under the old extractor.

   Fix: NER (Jean-Baptiste/roberta-large-ner-english) extracts typed
   entities — MONEY, PERCENT, CARDINAL, DATE, ORG, PERSON, QUANTITY —
   that cover measurable claims across any professional domain.

2. factual_consistency — hardcoded infra contradiction list
   ["self-managed" vs "confluent cloud", "on-prem" vs "aws"] only
   catches software infrastructure contradictions. A financial analyst
   claiming both "bear" and "bull" market positions, or a marketer
   claiming both "increased" and "decreased" conversion rates, would
   not be caught.

   Fix: cross-encoder/nli-deberta-v3-small checks whether pairs of
   extracted claim sentences contradict each other directly. No domain
   vocabulary needed — the model reasons about logical consistency.

3. Team size / version regex
   Team size regex is already reasonable but version pattern fires only
   on software versioning notation (X.Y.Z).

   Fix: generalise to "named entity + cardinal number" pairs via NER,
   which captures "Series B round of $40M", "portfolio of 12 companies",
   "panel of 8 physicians", "campaign with 3.2M impressions" equally.

Models used
───────────
- all-MiniLM-L6-v2  (semantic layer — unchanged, already domain-agnostic)
- Jean-Baptiste/roberta-large-ner-english  (claim extraction)
- cross-encoder/nli-deberta-v3-small       (factual contradiction detection)
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from itertools import combinations
from collections import defaultdict


# ── Lazy model loading ────────────────────────────────────────────────────────

_MODELS: Dict = {}

def _load(key: str):
    if key in _MODELS:
        return _MODELS[key]
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer

    loaders = {
        "embedder": lambda: SentenceTransformer(
            "all-MiniLM-L6-v2"
        ),
        "ner": lambda: pipeline(
            "ner",
            model="Jean-Baptiste/roberta-large-ner-english",
            aggregation_strategy="simple",
        ),
        "nli": lambda: pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small",
        ),
    }
    _MODELS[key] = loaders[key]()
    return _MODELS[key]


# ── Sentence splitter ─────────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def _split_sentences(text: str, min_tokens: int = 4) -> List[str]:
    return [
        s.strip() for s in _SENT_SPLIT.split(text.strip())
        if len(s.strip().split()) >= min_tokens
    ]


# ── Main scorer ───────────────────────────────────────────────────────────────

class CrossAnswerConsistencyScorer:
    """
    Detects inconsistency across screening answers for any professional domain.

    Works for: software engineers, financial analysts, marketers, nurses,
    product managers, lawyers, consultants, HR professionals, researchers.

    Two layers (unchanged architecture):
      1. Semantic  — embedding cosine similarity + variance across answer pairs
      2. Factual   — NER claim extraction + NLI contradiction detection

    Low consistency → high fraud signal (answers contradict each other)
    High consistency → low fraud signal (answers tell a coherent story)
    """

    # ── Tolerances for numeric claim comparison ───────────────────────────────
    # How much two claims of the same type can differ before flagging
    _NUMERIC_TOLERANCES = {
        "CARDINAL":  0.30,   # 30% relative difference → contradiction
        "PERCENT":   10.0,   # 10 percentage points → contradiction
        "MONEY":     0.40,   # 40% relative difference → contradiction
        "QUANTITY":  0.35,
    }

    # NLI confidence threshold to call something a contradiction
    _CONTRADICTION_THRESHOLD = 0.72

    # ── Layer 1: Semantic (unchanged — already domain-agnostic) ───────────────

    def semantic_consistency(self, answers: List[str]) -> dict:
        """
        Embeds all answers and measures pairwise cosine similarity.
        LLM-generated answers each describe a slightly different generic world
        → high variance in pairwise similarity even when mean looks acceptable.
        """
        if len(answers) < 2:
            return {"semantic_consistency": 1.0, "similarity_variance": 0.0,
                    "pairwise_scores": []}

        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        embedder    = _load("embedder")
        embeddings  = embedder.encode(answers, normalize_embeddings=True)
        pairs       = list(combinations(range(len(answers)), 2))
        similarities = [
            float(cos_sim([embeddings[i]], [embeddings[j]])[0][0])
            for i, j in pairs
        ]

        mean_sim    = float(np.mean(similarities))
        var_sim     = float(np.var(similarities))
        consistency = mean_sim * (1.0 - np.clip(var_sim * 5, 0.0, 0.5))

        return {
            "semantic_consistency": round(consistency, 4),
            "mean_pairwise_sim":    round(mean_sim, 4),
            "similarity_variance":  round(var_sim, 4),
            "pairwise_scores": [
                (f"A{i+1}-A{j+1}", round(s, 3))
                for (i, j), s in zip(pairs, similarities)
            ],
        }

    # ── Layer 2: Factual — NER-based claim extraction ─────────────────────────

    def extract_claims(self, answer: str) -> dict:
        """
        Extracts structured claims using NER instead of regex patterns.

        Replaces:
          - version_pattern  (software-only)  → PRODUCT + CARDINAL NER pairs
          - team_pattern     (partially general) → CARDINAL near team nouns
          - infra_pattern    (software-only)  → ORG + context entities

        New universal claim types:
          numeric_claims   — MONEY, PERCENT, QUANTITY, CARDINAL with context
          entity_claims    — ORG, PERSON, GPE, DATE, PRODUCT, EVENT mentions
          team_size        — CARDINAL near organisational nouns
          named_quantities — named entity + numeric value pairs (generalises versions)
        """
        claims = {
            "numeric_claims":    [],   # {type, value, context}
            "entity_claims":     [],   # {entity_group, word, context}
            "team_size":         None,
            "named_quantities":  {},   # {name_lower: value}  generalises versions
        }

        try:
            ner      = _load("ner")
            entities = ner(answer[:512])

            result_types = {"MONEY", "PERCENT", "QUANTITY", "CARDINAL"}
            anchor_types = {"ORG", "PERSON", "GPE", "DATE", "TIME",
                            "PRODUCT", "WORK_OF_ART", "EVENT"}

            # ── Numeric claims ────────────────────────────────────
            for ent in entities:
                if ent["entity_group"] in result_types:
                    # Grab surrounding context (±40 chars)
                    start   = max(0, ent["start"] - 40)
                    end     = min(len(answer), ent["end"] + 40)
                    context = answer[start:end].strip()
                    claims["numeric_claims"].append({
                        "type":    ent["entity_group"],
                        "value":   ent["word"],
                        "context": context,
                    })

            # ── Named entity claims ───────────────────────────────
            for ent in entities:
                if ent["entity_group"] in anchor_types:
                    start   = max(0, ent["start"] - 30)
                    end     = min(len(answer), ent["end"] + 30)
                    context = answer[start:end].strip()
                    claims["entity_claims"].append({
                        "entity_group": ent["entity_group"],
                        "word":         ent["word"],
                        "context":      context,
                    })

            # ── Named quantities: entity name + adjacent cardinal ─
            # Generalises "Ray 2.6" → any (PRODUCT/ORG, CARDINAL) pair
            # Covers: "Series B of $40M", "12-bed ICU", "panel of 8", etc.
            prev_anchor = None
            for ent in sorted(entities, key=lambda e: e["start"]):
                if ent["entity_group"] in anchor_types:
                    prev_anchor = ent
                elif ent["entity_group"] in result_types and prev_anchor:
                    gap = ent["start"] - prev_anchor["end"]
                    if gap < 15:   # within ~15 chars of the named entity
                        key = prev_anchor["word"].lower().strip()
                        claims["named_quantities"][key] = ent["word"]
                    prev_anchor = None

        except Exception as e:
            # Fallback: light regex for universal numerics
            claims["_fallback_error"] = str(e)
            claims["numeric_claims"] = self._regex_numeric_fallback(answer)

        # ── Team size (kept as regex — robust cross-domain pattern) ──
        team_pattern = re.compile(
            r'(\d+)[\-\s]?(?:person|people|member|employee|staff|'
            r'engineer|analyst|nurse|lawyer|physician|consultant|'
            r'researcher|designer|reporter)s?\s*(?:team|group|unit|department)?'
            r'|(?:team|group|unit|department|cohort|panel)\s+of\s+(\d+)'
            r'|\bmanaged\s+(\d+)\b|\bled\s+(\d+)\b|\bsupervised\s+(\d+)\b',
            re.IGNORECASE
        )
        team_matches = team_pattern.findall(answer)
        if team_matches:
            sizes = [int(x) for match in team_matches for x in match if x]
            if sizes:
                claims["team_size"] = sizes[0]

        return claims

    @staticmethod
    def _regex_numeric_fallback(text: str) -> List[dict]:
        """Domain-neutral numeric extraction when NER is unavailable."""
        pattern = re.compile(
            r'\b(\d+(?:[.,]\d+)?)\s*'
            r'(%|percent|x\b|times|\$|£|€|M\b|B\b|K\b|'
            r'million|billion|thousand|days?|weeks?|months?|years?|'
            r'hours?|beds?|patients?|cases?|accounts?|clients?|users?)\b',
            re.IGNORECASE
        )
        return [
            {"type": "CARDINAL", "value": f"{m.group(1)}{m.group(2)}", "context": ""}
            for m in pattern.finditer(text)
        ]

    # ── Layer 2: Factual — NLI contradiction detection ────────────────────────

    def factual_consistency(
        self,
        answers:   List[str],
        questions: Optional[List[str]] = None,
    ) -> dict:
        """
        Detects factual contradictions across answers using two passes:

        Pass 1: Structured claim comparison
          Compare extracted named_quantities and team sizes across answers.
          These are hard facts that should not change answer-to-answer.
          Replaces the hardcoded version + infra contradiction lists.

        Pass 2: NLI sentence-pair contradiction detection
          For each pair of answers, extract salient claim sentences and run
          the NLI cross-encoder to check for logical contradiction.
          Catches domain-agnostic contradictions the structured pass misses:
            - "bull market positioning" vs "bear market positioning"  (finance)
            - "mortality rate improved" vs "mortality rate worsened"  (healthcare)
            - "CTR increased" vs "CTR decreased"  (marketing)
            - "we grew the team" vs "we reduced headcount"  (any domain)
        """
        all_claims     = [self.extract_claims(a) for a in answers]
        contradictions = []

        # ── Pass 1: Named quantity contradictions ─────────────────
        # Generalises version_mismatch to any (name, value) pair
        entity_values: Dict[str, Tuple[int, str]] = {}
        for i, claims in enumerate(all_claims):
            for name, value in claims.get("named_quantities", {}).items():
                if name in entity_values:
                    prev_i, prev_v = entity_values[name]
                    if prev_v != value:
                        contradictions.append({
                            "type":     "named_quantity_mismatch",
                            "entity":   name,
                            "answer_a": prev_i + 1,
                            "value_a":  prev_v,
                            "answer_b": i + 1,
                            "value_b":  value,
                            "severity": "hard",
                        })
                entity_values[name] = (i, value)

        # ── Pass 1b: Team size contradictions (tolerance ±3) ──────
        team_sizes = [
            (i, c["team_size"])
            for i, c in enumerate(all_claims)
            if c.get("team_size") is not None
        ]
        for (i, s1), (j, s2) in combinations(team_sizes, 2):
            if abs(s1 - s2) > 3:
                contradictions.append({
                    "type":     "team_size_mismatch",
                    "answer_a": i + 1, "size_a": s1,
                    "answer_b": j + 1, "size_b": s2,
                    "severity": "moderate",
                })

        # ── Pass 2: NLI-based sentence contradiction detection ─────
        nli_contradictions = self._nli_contradiction_pass(answers)
        contradictions.extend(nli_contradictions)

        # ── Score ─────────────────────────────────────────────────
        # Hard contradictions (named quantity mismatches) penalise more
        hard_count     = sum(1 for c in contradictions if c.get("severity") == "hard")
        moderate_count = sum(1 for c in contradictions
                             if c.get("severity") in ("moderate", "nli"))

        penalty        = np.clip(hard_count * 0.35 + moderate_count * 0.20, 0.0, 1.0)
        factual_score  = 1.0 - float(penalty)

        return {
            "factual_consistency":  round(factual_score, 4),
            "n_contradictions":     len(contradictions),
            "hard_contradictions":  hard_count,
            "nli_contradictions":   moderate_count,
            "contradictions":       contradictions,
            "claims_per_answer":    all_claims,
        }

    def _nli_contradiction_pass(self, answers: List[str]) -> List[dict]:
        """
        Runs NLI on salient sentence pairs across all answer combinations.

        For each (answer_i, answer_j) pair, extracts the most claim-dense
        sentences from each (those containing numerics or named entities)
        and checks whether they contradict each other.

        Domain-agnostic: works on any professional domain because NLI
        reasons about logical consistency, not domain vocabulary.
        """
        contradictions = []
        try:
            nli_pipe = _load("nli")

            for i, j in combinations(range(len(answers)), 2):
                sents_i = self._salient_sentences(answers[i])
                sents_j = self._salient_sentences(answers[j])

                if not sents_i or not sents_j:
                    continue

                # Check each salient sentence from answer i against
                # each salient sentence from answer j
                for sent_i in sents_i[:3]:   # cap per answer for speed
                    for sent_j in sents_j[:3]:
                        premise    = sent_i
                        hypothesis = sent_j

                        result = nli_pipe(f"{premise} [SEP] {hypothesis}")
                        label  = result[0]["label"].upper()
                        conf   = result[0]["score"]

                        if label == "CONTRADICTION" and conf >= self._CONTRADICTION_THRESHOLD:
                            contradictions.append({
                                "type":      "nli_contradiction",
                                "answer_a":  i + 1,
                                "answer_b":  j + 1,
                                "sentence_a": sent_i[:80],
                                "sentence_b": sent_j[:80],
                                "confidence": round(conf, 3),
                                "severity":  "nli",
                            })

        except Exception as e:
            contradictions.append({
                "type":    "nli_unavailable",
                "error":   str(e),
                "severity":"nli",
            })

        return contradictions

    @staticmethod
    def _salient_sentences(text: str) -> List[str]:
        """
        Extracts sentences most likely to contain checkable factual claims.
        Prefers sentences with numbers, comparisons, or named entities.
        """
        sentences = _split_sentences(text)
        if not sentences:
            return []

        numeric_pattern = re.compile(r'\d')
        compare_pattern = re.compile(
            r'\b(?:increased|decreased|improved|worsened|grew|fell|rose|'
            r'dropped|higher|lower|more|less|better|worse|from|to)\b',
            re.IGNORECASE
        )

        scored = []
        for s in sentences:
            score = 0
            if numeric_pattern.search(s):   score += 2
            if compare_pattern.search(s):   score += 1
            if len(s.split()) > 8:          score += 1   # prefer complete sentences
            scored.append((score, s))

        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored if _ > 0]

    # ── Combined score ────────────────────────────────────────────────────────

    def score(self, answers: List[str], questions: Optional[List[str]] = None) -> dict:
        semantic = self.semantic_consistency(answers)
        factual  = self.factual_consistency(answers, questions)

        combined     = (
            0.40 * semantic["semantic_consistency"]
          + 0.60 * factual["factual_consistency"]
        )
        fraud_signal = 1.0 - combined

        return {
            "cross_answer_consistency_fraud_score": round(fraud_signal, 4),
            "combined_consistency":                 round(combined, 4),
            "semantic":                             semantic,
            "factual":                              factual,
            "verdict":                              self._interpret(fraud_signal),
        }

    @staticmethod
    def _interpret(fraud_signal: float) -> str:
        if fraud_signal > 0.65:
            return "high inconsistency — likely separate LLM calls per question"
        if fraud_signal > 0.35:
            return "moderate inconsistency — probe for contradictions directly"
        return "internally consistent — answers share a coherent implied world"


# ── Usage: four domains ───────────────────────────────────────────────────────

if __name__ == "__main__":
    scorer = CrossAnswerConsistencyScorer()

    # ── Domain 1: Software engineering (original) ─────────────────
    print("\n" + "═" * 60)
    print("  Software engineering")
    print("═" * 60)

    sw_fraud = [
        "We ran training on Ray 2.6 across 8 A100s. Team had 12 ML engineers.",
        "I set up the Ray 2.1 distributed setup myself. Small team of 4.",
        "Feature pipeline used self-managed Kafka on EC2.",
    ]
    sw_genuine = [
        "Training on Ray 2.6 — 8 A100 workers. Team of 12 across MLOps and research.",
        "The Ray 2.6 cluster was my setup. Core team of 4 owned infra, 12 consumed it.",
        "Self-managed Kafka — looked at Confluent Cloud but cost didn't justify it.",
    ]
    for label, answers in [("Fraud", sw_fraud), ("Genuine", sw_genuine)]:
        r = scorer.score(answers)
        print(f"\n  [{label}] fraud_signal={r['cross_answer_consistency_fraud_score']}  "
              f"contradictions={r['factual']['n_contradictions']}  "
              f"verdict: {r['verdict']}")

    # ── Domain 2: Financial analyst ───────────────────────────────
    print("\n" + "═" * 60)
    print("  Financial analyst")
    print("═" * 60)

    fin_fraud = [
        # Q1: AUM $2.3B, bull market positioning, team of 8 analysts
        "I managed a $2.3B long-only equity portfolio with a bullish tilt. "
        "Led a team of 8 analysts focused on technology sector.",

        # Q2: AUM $800M (mismatch), bear market positioning (NLI contradiction)
        "Our $800M fund maintained defensive positioning through the downturn. "
        "We were net short the market through most of last year.",

        # Q3: team of 3 (mismatch with Q1's 8 analysts)
        "It was a lean operation — just 3 of us covering the whole portfolio.",
    ]
    fin_genuine = [
        "Managed a $2.3B long-only equity portfolio focused on tech and healthcare. "
        "Led 8 analysts — 5 sector specialists and 3 generalists.",

        "The $2.3B was split roughly 60/40 across public and private credit. "
        "We ran a barbell strategy — defensive positions offset growth bets.",

        "My team of 8 was responsible for coverage. I had 4 direct reports, "
        "the others were matrixed from the broader research team.",
    ]
    for label, answers in [("Fraud", fin_fraud), ("Genuine", fin_genuine)]:
        r = scorer.score(answers)
        print(f"\n  [{label}] fraud_signal={r['cross_answer_consistency_fraud_score']}  "
              f"contradictions={r['factual']['n_contradictions']}  "
              f"verdict: {r['verdict']}")
        if r["factual"]["contradictions"]:
            for c in r["factual"]["contradictions"][:2]:
                print(f"    → {c['type']}: {c.get('entity', c.get('type', ''))}")

    # ── Domain 3: Healthcare / nursing ────────────────────────────
    print("\n" + "═" * 60)
    print("  Healthcare")
    print("═" * 60)

    health_fraud = [
        # Q1: 24-bed ICU, mortality rate 4.2%
        "I worked in a 24-bed ICU at a Level I trauma centre. "
        "Our unit mortality rate was around 4.2% that year.",

        # Q2: 12-bed unit (mismatch), mortality rate improved to 8% (NLI: higher = worse)
        "Our 12-bed cardiac ICU had strong outcomes. "
        "We brought mortality up to about 8% through better protocols.",

        # Q3: supervised 6 nurses (team size — need to check vs context)
        "As charge nurse I supervised 6 nurses per shift across the ward.",
    ]
    health_genuine = [
        "Worked in a 24-bed mixed ICU — medical, surgical, neuro. "
        "Unit mortality was around 4% annually, below the national benchmark.",

        "The 24 beds were split: 12 medical, 8 surgical, 4 step-down. "
        "We maintained a 1:2 nurse-patient ratio in the acute bays.",

        "Charge nurse for the 24-bed unit, overseeing 6 RNs per shift. "
        "Accountable for acuity-based staffing decisions.",
    ]
    for label, answers in [("Fraud", health_fraud), ("Genuine", health_genuine)]:
        r = scorer.score(answers)
        print(f"\n  [{label}] fraud_signal={r['cross_answer_consistency_fraud_score']}  "
              f"contradictions={r['factual']['n_contradictions']}  "
              f"verdict: {r['verdict']}")
        if r["factual"]["contradictions"]:
            for c in r["factual"]["contradictions"][:2]:
                print(f"    → {c['type']}: {str(c)[:80]}")

    # ── Domain 4: Marketing ───────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Marketing")
    print("═" * 60)

    mkt_fraud = [
        # Q1: $6M budget, CAC $340, campaign increased conversions
        "I managed a $6M annual paid budget. Our CAC was $340 and we drove "
        "a 28% increase in trial conversions through Q3.",

        # Q2: $2M budget (mismatch), CAC $1,200 (mismatch), conversions decreased (NLI)
        "Working with a $2M budget, we struggled with rising CAC — hit $1,200 "
        "by Q4. Conversion rates declined despite optimisation efforts.",

        # Q3: team of 12 (need a prior mention to contradict)
        "Leading a team of 12 across paid, lifecycle, and brand. "
        "Our ROAS improved to 4.2x last quarter.",
    ]
    mkt_genuine = [
        "Managed $6M across Google, Meta, and LinkedIn. CAC sat around $340 "
        "— we ran it down from $520 over 18 months by tightening audience targeting.",

        "The $6M was split: 60% performance, 30% brand, 10% experimentation. "
        "CAC $340 was the blended figure — performance channels were $280, brand lifted it.",

        "Led a team of 12: 4 paid specialists, 3 lifecycle, 2 brand, 2 analysts, "
        "1 project manager. ROAS improved from 2.8x to 4.2x over the year.",
    ]
    for label, answers in [("Fraud", mkt_fraud), ("Genuine", mkt_genuine)]:
        r = scorer.score(answers)
        print(f"\n  [{label}] fraud_signal={r['cross_answer_consistency_fraud_score']}  "
              f"contradictions={r['factual']['n_contradictions']}  "
              f"verdict: {r['verdict']}")
        if r["factual"]["contradictions"]:
            for c in r["factual"]["contradictions"][:2]:
                print(f"    → {c['type']}: {str(c)[:80]}")