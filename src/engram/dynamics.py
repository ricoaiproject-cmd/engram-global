"""Core memory dynamics - ACT-R activation model + RRF + spreading activation.

Design principle: the embedding (semantic position) stays fixed; activation is
a separate axis that modulates search ranking. Activation is computed fresh at
query time from the event history, so no batch decay update is ever needed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence


def decay_rate(
    importance: int,
    *,
    base: float = 0.5,
    spread: float = 0.2,
    d_min: float = 0.3,
    d_max: float = 0.6,
) -> float:
    """Per-memory decay exponent d_i (implements flashbulb memory).

    Memories acquired in a significant context (high importance) decay more
    slowly and stay recallable for a long time even without rehearsal.
    Trivial memories sink quickly.
        d_i = clamp(base - spread * (importance - 5) / 5, d_min, d_max)
    """
    d = base - spread * (importance - 5) / 5
    return max(d_min, min(d_max, d))


def create_event_weight(importance: int, *, alpha: float = 2.0) -> float:
    """Initial encoding boost. A memory with importance 10 starts life already
    at the strength (w = 1 + alpha) of a memory that's been accessed multiple times."""
    return 1.0 + alpha * (importance / 10.0)


def base_strength(
    events: Iterable[tuple[float, float]],
    now: float,
    decay: float,
    *,
    min_elapsed: float = 60.0,
) -> float:
    """Inner sum of the ACT-R base-level activation, S = sum_j w_j * (now - t_j)^(-d).

    events: a sequence of (unix-seconds timestamp, weight) pairs.
    Elapsed time is clamped to min_elapsed seconds to prevent divergence for
    an access that just happened. A future timestamp (clock skew) is also
    treated as min_elapsed.
    """
    s = 0.0
    for ts, weight in events:
        elapsed = max(now - ts, min_elapsed)
        s += weight * elapsed ** (-decay)
    return s


def activation(events: Iterable[tuple[float, float]], now: float, decay: float,
               *, min_elapsed: float = 60.0) -> float:
    """ACT-R base-level activation, B = ln(S). Returns -inf when there are no events."""
    s = base_strength(events, now, decay, min_elapsed=min_elapsed)
    return math.log(s) if s > 0 else float("-inf")


def activation_norm(events: Iterable[tuple[float, float]], now: float, decay: float,
                    *, min_elapsed: float = 60.0,
                    center: float = -6.0, scale: float = 1.5) -> float:
    """Activation normalized to 0..1: sigmoid((ln S - center) / scale).

    S is a power-law decay measured in seconds, so on a day-scale it lands
    around 1e-4 to 1e-2, and a naive S/(S+1) flattens the differences. As in
    the original ACT-R approach, we compare in log space (B = ln S), with the
    sigmoid's center and scale calibrated so that day-to-month-scale
    differences land in the discriminative range (roughly 0.1-0.9):
      - just created:                    ~0.95 (strong recency)
      - accessed once, 1 day ago:        ~0.55
      - untouched for 90 days (importance 5):  ~0.25
      - untouched for 90 days (importance 10): ~0.8 (flashbulb memory)
    S=0 (no events) -> 0. Usable directly in the re-rank weighted sum.
    """
    s = base_strength(events, now, decay, min_elapsed=min_elapsed)
    if s <= 0.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-(math.log(s) - center) / scale))


def final_score(
    relevance: float,
    act_norm: float,
    importance: int,
    *,
    w_relevance: float = 0.6,
    w_activation: float = 0.25,
    w_importance: float = 0.15,
) -> float:
    """Final search score. Relevance dominates; activation and importance only
    modulate it (preventing a rich-get-richer loop where frequently used
    memories intrude into unrelated contexts)."""
    return (
        w_relevance * relevance
        + w_activation * act_norm
        + w_importance * (importance / 10.0)
    )


def normalize_relevances(
    relevances: Mapping[str, float],
    *,
    floor: float = 0.10,
) -> dict[str, float]:
    """Min-max normalize relevance within the candidate set (Generative Agents style).

    Embedding cosine similarity tends to compress into a narrow band (with
    prefix-style Japanese models such as Ruri, 0.8-0.87, confirmed on real
    data). Feeding the raw value into final_score's weighted sum makes
    relevance's discriminative power far smaller than its nominal weight
    (w_relevance=0.6), letting the activation/importance floor effectively
    dominate the ranking (i.e. a generically high-activation memory intrudes
    into an unrelated query). Generative Agents (Park et al. 2023), the
    source for this weighted-sum design, min-max normalizes each component
    within the retrieval candidate set before summing; we follow that.

    A floor is applied to the denominator: spread = max(max - min, floor).
    When the whole candidate set has similar relevance (i.e. the query isn't
    discriminative), tiny differences are not amplified to 0..1; instead
    relevance's contribution is scaled down in proportion to the actual
    difference, and ranking falls back to activation/importance (analogous to
    ACT-R's fallback to base-level activation when the retrieval cue is
    weak).

    The normalized value is ranking-only. Reported relevance and threshold
    checks (deep/exhaustive auto-trigger, the surface relevance gate) must
    keep using the raw value.
    """
    if not relevances:
        return {}
    lo = min(relevances.values())
    spread = max(max(relevances.values()) - lo, floor)
    return {id_: (v - lo) / spread for id_, v in relevances.items()}


def rrf_merge(rankings: Sequence[Sequence[str]], *, k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion. Merges multiple rankings (vector neighbors, BM25, etc.).

    score(id) = sum_r 1/(k + rank_r(id))  (rank is 1-based; contributes 0 if absent from a list)
    The returned dict is not guaranteed to be sorted by score descending; sort it at the call site.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, mem_id in enumerate(ranking, start=1):
            scores[mem_id] = scores.get(mem_id, 0.0) + 1.0 / (k + rank)
    return scores


def spread(
    seeds: Mapping[str, float],
    neighbors: Callable[[str], Iterable[tuple[str, float]]],
    *,
    max_hops: int = 2,
    hop_decay: float = 0.7,
) -> dict[str, float]:
    """Spreading activation (implements deep recall's "follow the links and you
    can always pull it in" guarantee).

    From each memory in seeds, follows associative links out to max_hops, propagating
        propagated = seed_score * product_over_hops(link_weight * hop_decay)
    When a node is reached via multiple paths, the maximum is kept.
    neighbors(id) is a function returning (neighbor_id, link_weight 0..1) pairs.
    Returns scores for every reached node including the seeds themselves (seeds keep their original value).
    """
    best: dict[str, float] = dict(seeds)
    frontier: dict[str, float] = dict(seeds)
    for _ in range(max_hops):
        next_frontier: dict[str, float] = {}
        for node, score in frontier.items():
            for nbr, link_w in neighbors(node):
                w = max(0.0, min(1.0, link_w))
                propagated = score * w * hop_decay
                if propagated > best.get(nbr, 0.0):
                    best[nbr] = propagated
                    next_frontier[nbr] = max(next_frontier.get(nbr, 0.0), propagated)
        if not next_frontier:
            break
        frontier = next_frontier
    return best
