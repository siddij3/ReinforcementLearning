import re
import numpy as np
from typing import List, Dict


class ProfileVoiceConsistencyScorer:
    """
    Measures tonal and structural uniformity across profile sections.

    High uniformity  = every section sounds like the same robot → fraud signal
    Natural variation = sections have different registers, soft-skill ratios,
                        sentence length variance → genuine profile
    """

    SOFT_SKILL_MARKERS = re.compile(
        r'\b(?:collaborated?|mentored?|led\s+(?:discussions?|meetings?|workshops?)|'
        r'aligned\s+(?:stakeholders?|teams?|cross.functional)|'
        r'navigated?\s+(?:ambiguity|competing\s+priorities|organizational)|'
        r'communicated?\s+(?:with|to|across)|influenced?\s+(?:without|stakeholders?)|'
        r'grew?\s+(?:the\s+team|our\s+team|junior)|onboarded?|'
        r'conflict\s+resolution|built\s+consensus|drove?\s+alignment)\b',
        re.IGNORECASE
    )

    TECHNICAL_MARKERS = re.compile(
        r'\b(?:implemented|deployed|architected|built|designed|optimized|'
        r'configured|migrated|developed|integrated|automated|scaled)\b',
        re.IGNORECASE
    )

    FIRST_PERSON = re.compile(r'\bI\b')

    def score(self, sections: Dict[str, str]) -> dict:
        """
        Parameters
        ----------
        sections : dict mapping section name → text
            e.g. {"summary": "...", "role_1": "...", "role_2": "...", "skills": "..."}
        """
        if len(sections) < 2:
            return {"profile_voice_fraud_score": 0.4,
                    "verdict": "too few sections to compare"}

        section_profiles = {
            name: self._profile_section(text)
            for name, text in sections.items()
            if text.strip()
        }

        # ── 1. Soft-skill ratio per section ───────────────────────
        soft_ratios = [p["soft_skill_ratio"] for p in section_profiles.values()]
        soft_cv     = np.std(soft_ratios) / max(np.mean(soft_ratios), 0.01)

        # Very low CV = every section has same soft/technical ratio = suspicious
        uniformity_penalty = max(0.0, 0.6 - soft_cv) * 0.5

        # Near-zero soft skills across ALL sections = strong flag
        if np.mean(soft_ratios) < 0.05:
            zero_soft_penalty = 0.35
        else:
            zero_soft_penalty = 0.0

        # ── 2. Sentence length variance per section ───────────────
        sent_len_vars = [p["sentence_length_cv"] for p in section_profiles.values()]
        mean_sent_var = np.mean(sent_len_vars)
        # Low within-section variance = robot sentences
        sentence_penalty = max(0.0, 0.4 - mean_sent_var) * 0.4

        # ── 3. ATS keyword density ────────────────────────────────
        ats_densities = [p["technical_density"] for p in section_profiles.values()]
        ats_cv        = np.std(ats_densities) / max(np.mean(ats_densities), 0.01)
        # All sections equally technical = ATS-optimized throughout
        ats_penalty = max(0.0, 0.5 - ats_cv) * 0.25

        # ── 4. First-person pronoun distribution ──────────────────
        fp_rates = [p["first_person_rate"] for p in section_profiles.values()]
        fp_cv    = np.std(fp_rates) / max(np.mean(fp_rates), 0.01)
        # Real profiles: summary often third-person, bullets first-person
        pronoun_penalty = max(0.0, 0.4 - fp_cv) * 0.20

        final = float(np.clip(
            uniformity_penalty + zero_soft_penalty +
            sentence_penalty + ats_penalty + pronoun_penalty,
            0.0, 1.0
        ))

        return {
            "profile_voice_fraud_score":  round(final, 4),
            "soft_skill_mean_rate":       round(float(np.mean(soft_ratios)), 4),
            "soft_skill_cv":              round(float(soft_cv), 4),
            "sentence_uniformity":        round(float(mean_sent_var), 4),
            "section_profiles":           section_profiles,
            "verdict":                    self._interpret(final),
        }

    def _profile_section(self, text: str) -> dict:
        sentences  = re.split(r'[.!?•\n]+', text)
        sentences  = [s.strip() for s in sentences if len(s.strip()) > 8]
        tokens     = text.split()
        n_tok      = max(len(tokens), 1)

        soft_hits  = len(self.SOFT_SKILL_MARKERS.findall(text))
        tech_hits  = len(self.TECHNICAL_MARKERS.findall(text))
        fp_hits    = len(self.FIRST_PERSON.findall(text))

        sent_lens  = [len(s.split()) for s in sentences]
        sent_cv    = (np.std(sent_lens) / max(np.mean(sent_lens), 1)) if sent_lens else 0

        total_markers = max(soft_hits + tech_hits, 1)
        return {
            "soft_skill_ratio":    soft_hits / total_markers,
            "technical_density":   tech_hits / (n_tok / 100),
            "first_person_rate":   fp_hits   / (n_tok / 100),
            "sentence_length_cv":  round(float(sent_cv), 3),
            "sentence_count":      len(sentences),
        }

