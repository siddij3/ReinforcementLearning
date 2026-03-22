import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

class AnswerPerplexityScorer:
    """
    Measures how LLM-like a screening answer is via perplexity. Signals are fed to RL environment
    Uses a small causal LM (GPT-2 or distilGPT-2 for the PoC).
    In production: fine-tune on a corpus of verified-genuine expert answers.
    """

    def __init__(self, model_name: str = "distilgpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        # Baseline: median perplexity of known-genuine answers (calibrated offline)
        self.genuine_baseline_ppl = 120.0   # tune from your calibration corpus
        self.llm_baseline_ppl     = 35.0    # typical GPT-4 answer perplexity

    @torch.no_grad()
    def _raw_perplexity(self, text: str) -> float:
        """Compute raw perplexity of text under the model."""
        encodings = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = encodings.input_ids
        if input_ids.shape[1] < 5:
            return 100.0   # too short to score reliably

        # Stride to handle long answers — average NLL across chunks
        stride       = 256
        max_length   = self.model.config.n_positions
        nlls         = []
        prev_end_loc = 0

        for begin_loc in range(0, input_ids.size(1), stride):
            end_loc   = min(begin_loc + max_length, input_ids.size(1))
            trg_len   = end_loc - prev_end_loc
            chunk_ids = input_ids[:, begin_loc:end_loc]
            target    = chunk_ids.clone()
            target[:, :-trg_len] = -100   # mask the overlap

            outputs = self.model(chunk_ids, labels=target)
            nlls.append(outputs.loss.item() * trg_len)
            prev_end_loc = end_loc
            if end_loc == input_ids.size(1):
                break

        ppl = torch.exp(torch.tensor(sum(nlls) / input_ids.size(1))).item()
        return ppl

    @torch.no_grad()
    def _token_log_probs(self, text: str) -> List[float]:
        """Returns per-token log probability for every token in text."""
        enc = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=512,
        )
        ids = enc.input_ids
        if ids.shape[1] < 3:
            return []

        out    = self.model(ids, labels=ids)
        # shift: logits[i] predicts token[i+1]
        logits = out.logits[0, :-1, :]
        targets = ids[0, 1:]

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_lps = log_probs[torch.arange(len(targets)), targets]
        return token_lps.tolist()

    def score(self, answer: str, question_context: str = "") -> dict:
        """
        Returns a normalized fraud signal in [0, 1].
        1.0 = maximally LLM-like (low perplexity).
        0.0 = clearly human-authored (high perplexity).
        """
        # Score the answer in isolation
        raw_ppl = self._raw_perplexity(answer)

        # Score the answer conditioned on the question (if provided)
        # This catches cases where someone pastes a generic answer that
        # doesn't even respond to the specific question asked
        conditioned_ppl = raw_ppl
        if question_context:
            combined     = f"Q: {question_context}\nA: {answer}"
            conditioned_ppl = self._raw_perplexity(combined)

        # Normalize against calibrated baselines
        # Maps: llm_baseline → 1.0,  genuine_baseline → 0.0
        normalized = (raw_ppl - self.llm_baseline_ppl) / (
            self.genuine_baseline_ppl - self.llm_baseline_ppl
        )

        ## -- TokenBurstinessScorer (beyond raw perplexity) --
        # These capture more nuanced patterns in token-level probabilities,
        lps = self._token_log_probs(answer)
        if len(lps) < 10:
            return {"token_burstiness_fraud_score": 0.5,
                    "note": "answer too short to score reliably"}

        lps_arr = np.array(lps)
        mean_lp = np.mean(lps_arr)
        std_lp  = np.std(lps_arr)

        # Coefficient of variation of log-probs
        # High CV = bursty = human-like
        cv = std_lp / max(abs(mean_lp), 1e-6)

        # Fraction of very-high-probability tokens (> -0.5 log prob)
        # LLMs have many of these; humans have fewer
        high_prob_frac = np.mean(lps_arr > -0.5)

        # Fraction of surprisingly-low-probability tokens (< -5.0)
        # Humans produce more of these — idiosyncratic word choices,
        # domain-specific terms, names, errors
        low_prob_frac = np.mean(lps_arr < -5.0)



        perplexity_signal = 1.0 - np.clip(normalized, 0.0, 1.0)  # invert: low ppl → high fraud signal

        return {
            "answer_perplexity_score": round(perplexity_signal, 4),
            "raw_perplexity":          round(raw_ppl, 2),
            "conditioned_perplexity":  round(conditioned_ppl, 2),
            "mean_log_prob":                round(float(mean_lp), 4),
            "log_prob_cv":                  round(float(cv), 4),
            "high_prob_token_fraction":     round(float(high_prob_frac), 4),
            "low_prob_token_fraction":      round(float(low_prob_frac), 4),
        }

