"""
Models used
───────────
- Jean-Baptiste/roberta-large-ner-english
  Detects MONEY, PERCENT, QUANTITY, CARDINAL (measurable outcomes),
  ORG, PRODUCT, EVENT (named professional entities).
  Replaces: metric_with_context, internal_identifier.

- MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33
  Zero-shot classification using labels that describe artifact
  *function*, not domain content.
  Replaces: exception_class, failure_mode, config_key.

Architecture unchanged
───────────────────────
OperationalArtifact dataclass, weighted density with diminishing
returns per type, type diversity bonus, seniority bar. Only the
extraction layer is swapped out.
"""

import re
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# ── Model registry ────────────────────────────────────────────────────────────

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
    loaders = {
        "ner": lambda: pipeline(
            "ner",
            model="Jean-Baptiste/roberta-large-ner-english",
            aggregation_strategy="simple",
        ),
        "zeroshot": lambda: pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
        ),
    }
    _MODELS[key] = loaders[key]()
    return _MODELS[key]


# ── Dataclass (unchanged) ─────────────────────────────────────────────────────

@dataclass
class OperationalArtifact:
    artifact_type: str
    value:         str
    weight:        float
    context:       str


# ── Sentence splitter ─────────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def _sentences(text: str, min_tok: int = 4) -> List[str]:
    return [
        s.strip() for s in _SENT_SPLIT.split(text.strip())
        if len(s.strip().split()) >= min_tok
    ]

def _context(text: str, start: int, end: int, window: int = 45) -> str:
    return text[max(0, start - window): min(len(text), end + window)].strip()


# ── Main scorer ───────────────────────────────────────────────────────────────

class OperationalSpecificityScorer:
    """
    Scores how densely an answer is populated with operational artifacts —
    concrete signals that only accrue through genuine hands-on experience.

    Artifact weights (unchanged from original — weights reflect how hard
    the artifact type is to fabricate, regardless of domain):
      1.0 — named error / failure condition with a specific label
      0.9 — named failure pattern with mechanism explanation
      0.8 — named parameter or setting with a specific value
      0.7 — domain-specific internal entity (system, protocol, instrument)
      0.6 — named edition/version with a behavioral note
      0.5 — before/after metric comparison with units
      0.4 — bare named edition / version reference
      0.3 — personal / team reference
    """

    # ── Zero-shot label sets (describe artifact function, not content) ─────────

    _NAMED_ERROR_LABELS = [
        "names a specific error, exception, or failure condition",
        "describes a general problem without naming it specifically",
    ]
    _FAILURE_MECHANISM_LABELS = [
        "describes a named failure pattern, root cause, or mechanism",
        "describes what happened without explaining a specific failure pattern",
    ]
    _NAMED_PARAM_LABELS = [
        "references a specific named parameter, setting, threshold, or configuration value",
        "describes an action or outcome without referencing specific named settings",
    ]

    # ── Domain-neutral regex (kept / broadened from original) ─────────────────

    _PERSONAL_REF = re.compile(
        r'\b(?:my\s+colleague|my\s+co[- ]?founder|my\s+manager|my\s+lead|'
        r'our\s+team|our\s+[a-z]+\s+team|internally|'
        r'[A-Z][a-z]+\s+(?:told|said|found|flagged|noticed|pointed\s+out)|'
        r'at\s+[A-Z][a-zA-Z]+(?:\s+we)?|'
        r'we\s+(?:built|named|called|referred\s+to))\b',
        re.IGNORECASE
    )

    _NAMED_EDITION = re.compile(
        r'\bv?\d+\.\d+(?:\.\d+)?\b'                        # semantic versions
        r'|\b(?:DSM|ICD|RFC|ISO|IAS|IFRS|GAAP|Basel|'
        r'HIPAA|GDPR|SOC|PCI|NIST|CFA|FRM)\s*[\-–]?\s*\d+\b'  # named standards
        r'|\b(?:version|release|edition|revision|amendment|'
        r'update|patch)\s+\d+[\.\d]*\b',                   # generic versioning
        re.IGNORECASE
    )

    _EDITION_QUIRK = re.compile(
        r'(?:in|on|with|since|before|after|under|per|'
        r'upgrading\s+to|switching\s+to|migrating\s+to)\s+'
        r'(?:v?\d+[\.\d]*|[A-Z]{2,}[\s\-–]?\d+)\b'
        r'.{5,80}'
        r'(?:bug|issue|broke|changed|fixed|regression|behavior|quirk|'
        r'requirement|threshold|rule|restriction|limit|cap|ceiling|floor|'
        r'criterion|criteria|standard|specification)',
        re.IGNORECASE
    )

    _BEFORE_AFTER = re.compile(
        r'(?:from|reduced\s+from|dropped\s+from|fell\s+from|went\s+from|'
        r'improved\s+from|increased\s+from|up\s+from|down\s+from)\s+'
        r'[\$£€]?\d+(?:[.,]\d+)?\s*'
        r'(?:%|percent|bps|ms\b|s\b|x\b|k\b|M\b|B\b|'
        r'mg|ml|mmHg|mmol|ng|mcg|units?|beds?|'
        r'rps|qps|tps|rpm|fps)?\s*'
        r'(?:to|down\s+to|up\s+to)\s+'
        r'[\$£€]?\d+(?:[.,]\d+)?\s*'
        r'(?:%|percent|bps|ms\b|s\b|x\b|k\b|M\b|B\b|'
        r'mg|ml|mmHg|mmol|ng|mcg|units?|beds?|rps|qps|tps|rpm|fps)?',
        re.IGNORECASE
    )

    def __init__(self):
        self._weights = {
            "named_error":         1.0,   # was: exception_class
            "failure_mechanism":   0.9,   # was: failure_mode
            "named_parameter":     0.8,   # was: config_key
            "domain_entity":       0.7,   # was: internal_identifier
            "edition_quirk":       0.6,   # was: version_quirk
            "before_after_metric": 0.5,   # was: metric_with_context
            "named_edition":       0.4,   # was: version_number
            "personal_reference":  0.3,   # unchanged
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_artifacts(self, text: str) -> List[OperationalArtifact]:
        """
        Extracts all artifact types from the answer text.
        Model-based extractors run per-sentence; regex extractors run on full text.
        """
        artifacts: List[OperationalArtifact] = []

        # ── Model-based extraction (per sentence for speed) ───────
        sentences = _sentences(text)
        
        if sentences:
            artifacts += self._extract_named_errors(text, sentences)
            print(artifacts, "named_errors")
            artifacts += self._extract_failure_mechanisms(sentences)
            print(artifacts, "failure_mechanisms")
            artifacts += self._extract_named_parameters(sentences)
            print(artifacts, "named_parameters")

        # ── NER-based extraction (full text) ──────────────────────
        artifacts += self._extract_domain_entities(text)

        # ── Regex-based extraction (full text) ────────────────────
        artifacts += self._extract_regex(
            text, self._EDITION_QUIRK, "edition_quirk", 0.6
        )
        artifacts += self._extract_regex(
            text, self._BEFORE_AFTER, "before_after_metric", 0.5
        )
        artifacts += self._extract_regex(
            text, self._NAMED_EDITION, "named_edition", 0.4
        )
        artifacts += self._extract_regex(
            text, self._PERSONAL_REF, "personal_reference", 0.3
        )

        return artifacts

    def score(self, answer: str, expected_seniority: str = "senior") -> dict:
        """
        Parameters
        ----------
        answer             : candidate's answer text
        expected_seniority : 'junior' | 'mid' | 'senior' | 'staff'
        """
        seniority_bars = {
            "junior": 0.3, "mid": 0.5, "senior": 0.7, "staff": 0.85
        }
        bar       = seniority_bars.get(expected_seniority, 0.7)
        artifacts = self.extract_artifacts(answer)
        if not artifacts:
            return {
                "operational_specificity_score": 0.0,
                "artifact_count":   0,
                "weighted_density": 0.0,
                "artifacts":        [],
                "verdict": f"no operational artifacts for claimed {expected_seniority} level",
            }

        type_counts  = {}
        weighted_sum = 0.0
        for a in artifacts:
            count            = type_counts.get(a.artifact_type, 0)
            effective_weight = a.weight * (0.7 ** count)
            weighted_sum    += effective_weight
            type_counts[a.artifact_type] = count + 1

        # ── Density per 50 tokens, normalised by seniority bar ───
        n_tok             = max(len(answer.split()), 1)
        density           = weighted_sum / (n_tok / 50)
        specificity_score = float(np.clip(density / (bar * 3.0), 0.0, 1.0))

        # ── Type diversity bonus ──────────────────────────────────
        type_diversity = len(type_counts) / len(self._weights)
        final_score    = 0.75 * specificity_score + 0.25 * type_diversity

        return {
            "operational_specificity_score": round(final_score, 4),
            "artifact_count":   len(artifacts),
            "weighted_density": round(density, 4),
            "type_diversity":   round(type_diversity, 4),
            "artifact_breakdown": {
                t: {"count": c, "weight": self._weights.get(t, 0.5)}
                for t, c in type_counts.items()
            },
            "top_artifacts": [
                {"type": a.artifact_type, "value": a.value[:80],
                 "context": a.context[:120]}
                for a in sorted(artifacts, key=lambda x: -x.weight)[:6]
            ],
            "verdict": self._interpret(final_score, expected_seniority),
        }

    # ── Model-based extractors ────────────────────────────────────────────────

    def _extract_named_errors(
        self, full_text: str, sentences: List[str]
    ) -> List[OperationalArtifact]:
        """
        Zero-shot detects ANY sentence that names a specific error or
        failure condition, regardless of how it is expressed:

        """
        artifacts = []

        # Fast-path regex for software-style named exceptions (keep for speed)
        exc_pattern = re.compile(
            r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+(?:Error|Exception|Fault|'
            r'Warning|Panic|Killed|Timeout)\b'
        )
        for m in exc_pattern.finditer(full_text):
            artifacts.append(OperationalArtifact(
                artifact_type="named_error",
                value=m.group(),
                weight=1.0,
                context=_context(full_text, m.start(), m.end()),
            ))
            full_text_covered = True

        zs = _load("zeroshot")
        for sent in sentences[:10]:
            result = zs(sent, self._NAMED_ERROR_LABELS, multi_label=False)
            score  = result["scores"][
                result["labels"].index(self._NAMED_ERROR_LABELS[0])
            ]
            if score > 0.80:
                if not exc_pattern.search(sent):
                    artifacts.append(OperationalArtifact(
                        artifact_type="named_error",
                        value=sent[:60],
                        weight=1.0,
                        context=sent,
                    ))
        print(artifacts, "named_errors")
        return artifacts

    def _extract_failure_mechanisms(
        self, sentences: List[str]
    ) -> List[OperationalArtifact]:
        """
        Replaces: failure_mode regex (race condition, deadlock, OOM kill, etc.).

        Zero-shot detects sentences that describe a named failure pattern
        with a mechanism, across all domains:
          Software:    "thundering herd when our cache expired"
          Finance:     "the carry trade unwound as correlations spiked to 1"
          Healthcare:  "obstructive shock from tension pneumothorax"
          Marketing:   "audience exhaustion — we'd saturated the lookalike"

        Also kept: a broader regex for universal failure-mechanism language
        as a fallback when the model is unavailable.
        """
        artifacts = []

        # Broad domain-neutral failure vocabulary (fallback regex)
        _fallback = re.compile(
            r'\b(?:race\s+condition|deadlock|split\s+brain|cascade|'
            r'cascading\s+failure|thundering\s+herd|backpressure|'
            r'hot\s+(?:partition|spot|path)|poison\s+pill|'
            r'write\s+amplification|read\s+amplification|'
            r'gc\s+pause|stop[- ]the[- ]world|oom|livelock|'
            # Finance
            r'margin\s+call|flash\s+crash|liquidity\s+crunch|'
            r'correlation\s+breakdown|carry\s+unwind|basis\s+risk|'
            r'convexity\s+trap|duration\s+mismatch|'
            # Healthcare
            r'anaphylactic\s+shock|septic\s+shock|obstructive\s+shock|'
            r'tension\s+pneumothorax|cardiac\s+tamponade|'
            r'acute\s+kidney\s+injury|respiratory\s+failure|'
            # Marketing / operations
            r'audience\s+exhaustion|creative\s+fatigue|'
            r'attribution\s+collapse|churn\s+spike|funnel\s+collapse|'
            r'supply\s+chain\s+disruption|capacity\s+crunch)\b',
            re.IGNORECASE
        )

        try:
            zs = _load("zeroshot")
            for sent in sentences[:10]:
                result = zs(sent, self._FAILURE_MECHANISM_LABELS, multi_label=False)
                score  = result["scores"][
                    result["labels"].index(self._FAILURE_MECHANISM_LABELS[0])
                ]
                if score > 0.78:
                    artifacts.append(OperationalArtifact(
                        artifact_type="failure_mechanism",
                        value=sent[:60],
                        weight=0.9,
                        context=sent,
                    ))
        except Exception:
            # Fallback regex
            for sent in sentences:
                for m in _fallback.finditer(sent):
                    artifacts.append(OperationalArtifact(
                        artifact_type="failure_mechanism",
                        value=m.group(),
                        weight=0.9,
                        context=sent,
                    ))

        return artifacts

    def _extract_named_parameters(
        self, sentences: List[str]
    ) -> List[OperationalArtifact]:
        """
        Replaces: config_key regex (dotted keys like max.poll.interval.ms).

        Zero-shot detects sentences that reference a specific named parameter,
        setting, threshold, or configuration value, regardless of domain:
        """
        artifacts = []

        _fallback = re.compile(
            r'\b(?:'
            # Dotted config keys (software)
            r'[a-z][a-z_\-]+\.[a-z_\-]+(?:\.[a-z_\-]+)?'
            r'|--[a-z][a-z\-]+=\S+'
            # Named threshold / parameter patterns (any domain)
            r'|(?:set|configured|tuned|capped|limited|fixed|'
            r'adjusted|calibrated)\s+(?:to|at)\s+[\d\$£€]'
            r'|(?:threshold|limit|cap|floor|ceiling|ratio|rate|'
            r'target|budget|quota|allocation)\s+(?:of|at|is|was|to)\s+[\d\$£€]'
            r'|[\d]+\s*(?:bps|%|ms\b|mg\b|ml\b|mmHg|cmH2O)\s+'
            r'(?:threshold|limit|cap|target|ceiling|floor|cutoff)'
            r')\b',
            re.IGNORECASE
        )

        try:
            zs = _load("zeroshot")
            for sent in sentences[:10]:
                result = zs(sent, self._NAMED_PARAM_LABELS, multi_label=False)
                score  = result["scores"][
                    result["labels"].index(self._NAMED_PARAM_LABELS[0])
                ]
                if score > 0.75:
                    artifacts.append(OperationalArtifact(
                        artifact_type="named_parameter",
                        value=sent[:60],
                        weight=0.8,
                        context=sent,
                    ))
        except Exception:
            for sent in sentences:
                for m in _fallback.finditer(sent):
                    artifacts.append(OperationalArtifact(
                        artifact_type="named_parameter",
                        value=m.group(),
                        weight=0.8,
                        context=sent,
                    ))

        return artifacts

    # ── NER-based extractor ───────────────────────────────────────────────────

    def _extract_domain_entities(self, text: str) -> List[OperationalArtifact]:
        """
        Replaces: internal_identifier regex (file paths, method calls, env vars)
                  metric_with_context regex (p99, rps, tps, qps).

        NER finds domain-specific named entities and measurable outcomes
        across all professional domains:
        """
        artifacts = []
    
        ner      = _load("ner")
        entities = ner(text[:512])

        # Domain entity types (replaces internal_identifier)
        domain_entity_types = {"PRODUCT", "ORG", "EVENT", "WORK_OF_ART"}
        # Measurable outcome types (replaces metric_with_context)
        metric_types        = {"MONEY", "PERCENT", "QUANTITY", "CARDINAL"}

        for ent in entities:
            start   = ent.get("start", 0)
            end_pos = ent.get("end",   0)
            ctx     = _context(text, start, end_pos)

            if ent["entity_group"] in domain_entity_types:
                artifacts.append(OperationalArtifact(
                    artifact_type="domain_entity",
                    value=ent["word"],
                    weight=0.7,
                    context=ctx,
                ))
            elif ent["entity_group"] in metric_types:
                # Only count as "metric with context" if there's a
                # comparison word nearby — bare numbers score lower
                compare_pattern = re.compile(
                    r'\b(?:from|to|dropped|rose|fell|improved|'
                    r'increased|decreased|reduced|grew|went)\b',
                    re.IGNORECASE
                )
                has_comparison = bool(compare_pattern.search(ctx))
                artifacts.append(OperationalArtifact(
                    artifact_type="before_after_metric" if has_comparison
                                    else "named_edition",
                    value=ent["word"],
                    weight=0.5 if has_comparison else 0.2,
                    context=ctx,
                ))

        return artifacts

    # ── Regex extractor helper ────────────────────────────────────────────────

    @staticmethod
    def _extract_regex(
        text: str, pattern: re.Pattern, artifact_type: str, weight: float
    ) -> List[OperationalArtifact]:
        return [
            OperationalArtifact(
                artifact_type=artifact_type,
                value=m.group()[:80],
                weight=weight,
                context=_context(text, m.start(), m.end()),
            )
            for m in pattern.finditer(text)
        ]

    @staticmethod
    def _interpret(score: float, seniority: str) -> str:
        if score > 0.70:
            return f"high specificity for {seniority} — strong authenticity signal"
        if score > 0.40:
            return f"moderate specificity — acceptable for {seniority}, probe further"
        return f"low specificity for claimed {seniority} level — likely generic answer"


# ── Usage: four domains ───────────────────────────────────────────────────────

if __name__ == "__main__":
    scorer = OperationalSpecificityScorer()

    test_cases = [
        # ── Software (original domain) ─────────────────────────────
        {
            "domain": "Software — Kafka",
            "llm": """
                To optimize Kafka consumer throughput, I focused on tuning
                fetch.min.bytes and fetch.max.wait.ms to batch messages
                efficiently. I also increased consumer threads to match
                partition count and monitored consumer lag using standard
                metrics. I implemented a dead letter queue for poison pill
                messages and ensured proper offset management.
            """,
            "genuine": """
                We had consumer lag spiking to 40 minutes on our payments
                topic — turned out max.poll.records was at 500 and our
                deserialization was hitting a ProtobufDecodeError on 0.1%
                of messages, causing the whole poll loop to retry. The OOM
                kills were a red herring. Fixed it by dropping max.poll.records
                to 50, routing errors to dead-letter-payments-v2, and bumping
                fetch.max.wait.ms from 500 to 1000ms. p99 lag went from 40min
                to 8s.
            """,
        },
        # ── Finance ────────────────────────────────────────────────
        {
            "domain": "Finance — fixed income",
            "llm": """
                Managing interest rate risk requires careful duration analysis
                and regular portfolio rebalancing. Generally speaking, investors
                should monitor DV01 exposure and use rate swaps to hedge against
                adverse movements. It is important to align duration with the
                benchmark and maintain consistent risk management practices.
            """,
            "genuine": """
                We had a 0.4yr long duration tilt going into March 2022. When
                the curve inverted we closed half of it in the swap market
                rather than selling bonds — the bid-ask on the physicals was
                12 cents wide that week, the swap was 2bps. Still cost us
                roughly 18bps on the quarter. DV01 on the long end went from
                $420K to $180K per basis point. Under Basel III our capital
                charge for the residual position was flagged by risk at 11am.
            """,
        },
        # ── Healthcare ─────────────────────────────────────────────
        {
            "domain": "Healthcare — ICU",
            "llm": """
                Managing septic shock requires following established protocols
                and the ABCDE approach. It is important to administer
                intravenous fluids promptly and escalate to vasopressors when
                indicated. Multidisciplinary team involvement ensures optimal
                patient outcomes in critical care settings.
            """,
            "genuine": """
                Patient came in with MAP of 52 despite 2L crystalloid — we
                started norepinephrine at 0.05 mcg/kg/min and got a second
                large-bore IV in the left AC. Lactate was 4.8, which put us
                firmly in septic shock per Sepsis-3 criteria. PEEP was set
                to 8 cmH2O on the ventilator after we intubated. The attending
                wanted a central line for the vasopressors — right IJ because
                the left subclavian had a previous line infection. O2 sat went
                from 88% to 96% within 15 minutes of intubation.
            """,
        },
        # ── Marketing ──────────────────────────────────────────────
        {
            "domain": "Marketing — paid acquisition",
            "llm": """
                Addressing rising CAC requires a comprehensive review of
                targeting parameters and creative performance. Best practices
                suggest conducting A/B testing and ensuring alignment between
                media strategy and sales processes. It is recommended to
                optimise landing page conversion rates and regularly audit
                your attribution model for accuracy.
            """,
            "genuine": """
                CAC crept from $420 to $890 over 18 months because we'd
                exhausted our best-fit audiences and kept bidding into worse
                segments. Our form fill to sales-qualified rate dropped from
                22% to 11% after we broadened match types to hit volume.
                Rolling back to phrase match recovered 14 points of conversion
                within six weeks. We also rebuilt our lookalike seed from
                2,000 to 8,000 verified converters — that brought CAC back to
                $540 and ROAS from 1.8x to 3.4x on the Meta campaigns.
            """,
        },
    ]

    for case in test_cases:

        print(f"\n{'═'*60}")
        print(f"  {case['domain']}")
        print(f"{'═'*60}")
        for label, text in [("LLM", case["llm"]), ("Genuine", case["genuine"])]:
            result = scorer.score(text.strip(), "senior")
            # print(result?)
            print(f"\n  [{label}]")
            print(f"    score:          {result['operational_specificity_score']}")
            print(f"    artifact_count: {result['artifact_count']}")
            print(f"    type_diversity: {result['type_diversity']}")
            if result.get("top_artifacts"):
                print(f"    top artifacts:")
                for a in result["top_artifacts"][:3]:
                    print(f"      [{a['type']}] {a['value']}")
            print(f"    verdict: {result['verdict']}")