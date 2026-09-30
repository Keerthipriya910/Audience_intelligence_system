from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

@dataclass
class EWTCResult:
    original_label: str
    final_label: str
    original_confidence: float
    final_confidence: float
    weighted_agreement: float
    correction_strength: float
    sigma: float
    evidence_label: str | None
    support_ratio: float


def _kernel_weights(distances: list[float]) -> tuple[np.ndarray, float]:
    if not distances:
        return np.array([], dtype=float), 1.0
    d = np.asarray(distances, dtype=float)
    positive = d[d > 1e-9]
    sigma = float(np.median(positive)) if positive.size else 1.0
    sigma = max(sigma, 1e-6)
    w = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
    if float(w.sum()) <= 1e-12:
        w = np.ones_like(d)
    return w / w.sum(), sigma


def apply_ewtc(predicted_label: str, confidence: float, evidence_labels: list[str], distances: list[float]) -> EWTCResult:
    """Evidence-Weighted Trust Correction.

    The implementation follows the proposal's stated design:
    - kernel width is derived from retrieved distances using the median heuristic;
    - closer comments carry more evidence weight;
    - correction strength is lambda = 1-C, derived from model uncertainty;
    - no manually tuned fusion weight is introduced.

    Agreement is represented on [-1, 1]: +1 for supporting evidence and -1 for
    contradicting evidence. The final confidence uses C + (1-C)*A*(1-C), matching
    the proposal's worked example when A is positive. If weighted contradictory
    evidence is dominant and strong enough to cross the neutral midpoint, the
    dominant evidence label is surfaced as the trust-adjusted label.
    """
    c = float(np.clip(confidence, 0.0, 1.0))
    if not evidence_labels:
        return EWTCResult(predicted_label, predicted_label, c, c, 0.0, 1-c, 1.0, None, 0.0)

    weights, sigma = _kernel_weights(distances)
    agreements = np.array([1.0 if lab == predicted_label else -1.0 for lab in evidence_labels], dtype=float)
    weighted_agreement = float(np.dot(weights, agreements))
    lam = 1.0 - c
    corrected = float(np.clip(c + lam * weighted_agreement * (1.0 - c), 0.0, 1.0))

    label_weights: dict[str, float] = {}
    for lab, w in zip(evidence_labels, weights.tolist()):
        label_weights[lab] = label_weights.get(lab, 0.0) + float(w)
    evidence_label = max(label_weights, key=label_weights.get)
    support_ratio = float(label_weights[evidence_label])

    final_label = predicted_label
    # This relabel rule has no extra learned/tuned weight: it uses the natural 0.5 majority boundary.
    if evidence_label != predicted_label and support_ratio > 0.5 and corrected < 0.5:
        final_label = evidence_label
        corrected = max(0.5, 1.0 - corrected)

    return EWTCResult(
        original_label=predicted_label,
        final_label=final_label,
        original_confidence=c,
        final_confidence=corrected,
        weighted_agreement=weighted_agreement,
        correction_strength=lam,
        sigma=sigma,
        evidence_label=evidence_label,
        support_ratio=support_ratio,
    )
