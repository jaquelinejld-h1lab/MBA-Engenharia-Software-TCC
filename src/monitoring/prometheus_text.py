"""Minimal parser of the Prometheus text exposition format.

Used by the Streamlit panel and by the drift scenario scripts to read the
``/metrics`` endpoint of the API without importing ``prometheus_client``
on the client side. Only the subset the project emits is supported:
gauges, counters and histograms with ``_bucket``/``_sum``/``_count``
series, labels with quoted string values.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)"
)
_LABEL = re.compile(r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<value>(?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class Sample:
    """One series value."""

    name: str
    labels: dict[str, str] = field(default_factory=dict)
    value: float = 0.0


def parse_exposition(text: str) -> list[Sample]:
    """Parse the text format into samples; comments and blank lines are ignored."""
    samples: list[Sample] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if match is None:
            continue
        labels = {
            m.group("key"): m.group("value").replace('\\"', '"')
            for m in _LABEL.finditer(match.group("labels") or "")
        }
        samples.append(Sample(match.group("name"), labels, _to_float(match.group("value"))))
    return samples


def _to_float(token: str) -> float:
    lowered = token.lower()
    if lowered in {"+inf", "inf"}:
        return math.inf
    if lowered == "-inf":
        return -math.inf
    if lowered == "nan":
        return math.nan
    return float(token)


def _matches(sample: Sample, name: str, where: Mapping[str, str] | None) -> bool:
    if sample.name != name:
        return False
    return all(sample.labels.get(k) == v for k, v in (where or {}).items())


def select(samples: Iterable[Sample], name: str, **where: str) -> list[Sample]:
    """Return the samples of ``name`` whose labels contain every ``where`` pair."""
    return [s for s in samples if _matches(s, name, where)]


def total(samples: Iterable[Sample], name: str, **where: str) -> float:
    """Sum of the matching samples (zero when none)."""
    return float(sum(s.value for s in select(samples, name, **where)))


def histogram_quantile(
    samples: Iterable[Sample], name: str, quantile: float, **where: str
) -> float | None:
    """Quantile of a cumulative histogram, aggregated over the matching label sets.

    Uses the same linear interpolation inside the bucket as Prometheus'
    ``histogram_quantile``; returns ``None`` when the histogram has no
    observations. The value is over the whole process lifetime (the
    exposition carries no time window); Prometheus with ``rate()`` gives
    the windowed version.
    """
    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between 0 and 1")
    buckets: dict[float, float] = {}
    for sample in select(samples, f"{name}_bucket", **where):
        upper = _to_float(sample.labels.get("le", "inf"))
        buckets[upper] = buckets.get(upper, 0.0) + sample.value
    if not buckets:
        return None
    edges = sorted(buckets)
    count = buckets[edges[-1]]
    if count <= 0:
        return None
    rank = quantile * count
    previous_edge, previous_count = 0.0, 0.0
    for edge in edges:
        cumulative = buckets[edge]
        if cumulative >= rank:
            if math.isinf(edge):
                return previous_edge  # Prometheus returns the last finite bound
            width = edge - previous_edge
            inside = cumulative - previous_count
            fraction = (rank - previous_count) / inside if inside > 0 else 1.0
            return previous_edge + width * fraction
        previous_edge, previous_count = edge, cumulative
    return edges[-1]
