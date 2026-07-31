"""Build segment-as-node turn connectivity (the line / dual graph)."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def build_turn_edges(
    seg_records: Iterable[dict],
    exclude_uturn: bool = True,
) -> list[tuple[int, int]]:
    """Connect segment e=(u,v) -> e'=(v,w): e feeds into any segment leaving v.

    Parameters
    ----------
    seg_records
        Iterable of dicts each with keys 'segment_id', 'u', 'v'.
    exclude_uturn
        Drop the immediate reversal e=(u,v) -> e'=(v,u).

    Returns a list of (src_segment_id, dst_segment_id) directed turn edges.
    """
    out_by_tail: dict[int, list[tuple[int, int]]] = defaultdict(list)  # u -> [(segment_id, v)]
    recs = list(seg_records)
    for r in recs:
        out_by_tail[r["u"]].append((r["segment_id"], r["v"]))

    turns: list[tuple[int, int]] = []
    for r in recs:
        s, u, v = r["segment_id"], r["u"], r["v"]
        for s2, w in out_by_tail.get(v, ()):
            if s2 == s:
                continue  # no self-loop
            if exclude_uturn and w == u:
                continue  # immediate U-turn back down the same road
            turns.append((s, s2))
    return turns
