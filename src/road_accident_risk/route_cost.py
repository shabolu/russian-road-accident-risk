"""Apply alternative attribute profiles to a fixed directed routing graph."""

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd

# Pilot speed caps (km/h), fixed before evaluating crashes.
SURFACE_CAPS = {
    'paving_stones': 40, 'sett': 30, 'cobblestone': 25, 'cobblestone:flattened': 30,
    'concrete:plates': 50, 'compacted': 45, 'fine_gravel': 40, 'gravel': 35,
    'unpaved': 30, 'ground': 25, 'dirt': 25, 'earth': 25, 'sand': 15,
    'mud': 10, 'grass': 15, 'rocky': 15, 'ice': 15, 'snow': 20,
    'переходная': 40, 'низшая': 25,
}
SMOOTHNESS_CAPS = {
    'good': 80, 'intermediate': 60, 'bad': 40, 'very_bad': 25,
    'horrible': 15, 'very_horrible': 8,
}
SOURCE_PRIORITY = {'osm': 0, 'skdf': 1, 'continuity': 2, 'model': 3, 'fallback': 4}
EPS = 1e-10


@dataclass(frozen=True)
class Observation:
    start: float
    end: float
    direction: str
    field: str
    value: str
    priority: int
    scope: str


def index_observations(table: pd.DataFrame) -> dict[int, list[Observation]]:
    groups = defaultdict(list)
    for row in table.itertuples(index=False):
        start, end = float(row.start_frac), float(row.end_frac)
        if not (isfinite(start) and isfinite(end) and -EPS <= start < end <= 1 + EPS):
            raise ValueError(f'Invalid interval on OSM way {row.way_id}: {start}, {end}')
        if row.direction not in {'both', 'forward', 'backward'}:
            raise ValueError(f'Invalid direction: {row.direction}')
        if row.source not in SOURCE_PRIORITY:
            raise ValueError(f'Unknown observation source: {row.source}')
        groups[int(row.way_id)].append(Observation(
            max(0.0, start), min(1.0, end), row.direction, row.field, str(row.value),
            SOURCE_PRIORITY[row.source], getattr(row, 'value_scope', ''),
        ))
    return dict(groups)


def _number(value: str | None) -> float:
    try:
        value = float(value)
        return value if isfinite(value) and value > 0 else np.inf
    except (ValueError, TypeError):
        return np.inf


def _speed_cap(active: list[Observation]) -> float:
    chosen = {}
    for obs in active:
        chosen.setdefault(obs.field, obs)
    width_obs = chosen.get('width')
    lane_obs = next((obs for obs in active if obs.field == 'lanes'
                     and width_obs is not None and obs.scope == width_obs.scope), None)
    width = _number(width_obs.value if width_obs else None)
    lanes = _number(lane_obs.value if lane_obs else None)
    narrow_cap = np.inf
    if isfinite(width) and isfinite(lanes):
        # One cap for the joint width / lanes estimate.
        if width / lanes < 2.5:
            narrow_cap = 25
        elif width / lanes < 2.8:
            narrow_cap = 40

    values = {field: obs.value for field, obs in chosen.items()}
    return min(_number(values.get('maxspeed')), SURFACE_CAPS.get(values.get('surface'), np.inf),
               SMOOTHNESS_CAPS.get(values.get('smoothness'), np.inf), narrow_cap)


def edge_costs(edges: pd.DataFrame, observations: pd.DataFrame) -> np.ndarray:
    """Return conditional seconds indexed by edge ID, keeping graph policy fixed.

    Required edge columns include `base_speed_kmh` and `base_cost_s`. The
    excess of `base_cost_s` over travel time is kept as a multiplicative
    preference. Turn and access rules are handled by the graph.
    """
    indexed = index_observations(observations)
    by_id = edges.sort_values('edge_id')
    if not np.array_equal(by_id.edge_id.to_numpy(), np.arange(len(by_id))):
        raise ValueError('Expected dense edge IDs starting at zero')
    costs = by_id.base_cost_s.to_numpy(dtype=float, copy=True)
    cols = ['edge_id', 'way_id', 'start_frac', 'end_frac', 'base_speed_kmh', 'base_cost_s']
    for edge_id, way_id, start, end, speed, base_cost in by_id[cols].itertuples(index=False, name=None):
        if way_id not in indexed:
            continue
        direction = 'forward' if end > start else 'backward'
        low, high = sorted((start, end))
        if high - low <= EPS:
            continue
        selected = [o for o in indexed[way_id]
                    if o.end > low + EPS and o.start < high - EPS and o.direction in ('both', direction)]
        if not selected:
            continue
        selected.sort(key=lambda o: (o.priority, o.direction == 'both', o.start, o.end, o.value))
        boundaries = {low, high}
        for obs in selected:
            boundaries.update((max(low, obs.start), min(high, obs.end)))
        ordered = sorted(boundaries)
        profile_cost = 0.0
        for left, right in zip(ordered, ordered[1:]):
            midpoint = (left + right) / 2
            active = [obs for obs in selected if obs.start <= midpoint < obs.end]
            cap = _speed_cap(active)
            profile_cost += (right - left) * base_cost * (speed / min(speed, cap))
        costs[edge_id] = profile_cost / (high - low)
    return costs
