"""Recover bracketed route attributes across short gaps on one carriageway."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import groupby
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/route_proxy_experiments'
PILOTS = ('saratov', 'samara', 'ulyanovsk')
FIELDS = ('lanes', 'surface', 'width')
MAX_GAP_M = 60
MASK_M = (25, 50)
SEED = 20261124
EPS = 1e-9
GEOD = Geod(ellps='WGS84')
OUTPUT_COLS = ['way_id', 'start_frac', 'end_frac', 'direction', 'field', 'value',
               'raw_value', 'source', 'method', 'quality', 'value_scope', 'source_ref']
MATCH_TAGS = ['name', 'highway', 'service', 'bridge', 'tunnel', 'local_access', 'local_service']


@dataclass(frozen=True)
class Span:
    lo: float
    hi: float
    value: str
    scope: str
    ref: str

    @property
    def label(self):
        return self.value, self.scope


def way_geometry(graph_dir):
    edges = pd.read_parquet(graph_dir / 'edges.parquet', columns=[
        'way_id', 'u', 'v', 'start_frac', 'end_frac', 'length_m', 'direction', *MATCH_TAGS,
    ])
    total = edges.length_m / (edges.end_frac - edges.start_frac).abs()
    lengths = total.groupby(edges.way_id).first().to_dict()
    attrs = edges.groupby('way_id')[MATCH_TAGS].first()
    degrees = edges.groupby('way_id').direction.nunique()
    # Check degree using physical links, counting both halves of a two-way edge once.
    physical = edges.loc[edges.direction.eq(1) | edges.way_id.map(degrees).eq(1)]
    counts = pd.concat([physical.u, physical.v], ignore_index=True).value_counts()
    controls = json.loads((graph_dir / 'controls.json').read_text())
    forbidden = {row['node_id'] for row in controls['nodes']}
    forbidden.update(row['via_node'] for row in controls['turns'])
    allowed = set(counts[counts.eq(2)].index) - forbidden
    ends = defaultdict(list)
    for edge in physical.itertuples(index=False):
        for node, other, frac, other_frac in [(edge.u, edge.v, edge.start_frac, edge.end_frac),
                                               (edge.v, edge.u, edge.end_frac, edge.start_frac)]:
            if node in allowed and (frac <= EPS or frac >= 1 - EPS):
                leaving = (other_frac - frac) * edge.direction > 0
                ends[node].append((edge.way_id, other, int(frac >= .5), leaving))
    coords = pd.read_parquet(graph_dir / 'nodes.parquet').set_index('node_id')[['lon', 'lat']]
    joins = []
    for node, links in ends.items():
        if len(links) != 2:
            continue
        (a, other_a, side_a, leave_a), (b, other_b, side_b, leave_b) = links
        if a == b or degrees[a] != degrees[b] or degrees[a] == 1 and leave_a == leave_b:
            continue
        if not attrs.at[a, 'name'] or not attrs.loc[a].equals(attrs.loc[b]):
            continue
        lon, lat = coords.loc[node]
        other = coords.loc[[other_a, other_b]].to_numpy()
        bearings = GEOD.inv([lon, lon], [lat, lat], other[:, 0], other[:, 1])[0]
        angle = abs((bearings[0] - bearings[1] + 180) % 360 - 180)
        if angle >= 155:
            joins.append((node, a, side_a, b, side_b))
    return lengths, attrs.highway.to_dict(), joins


def observations(path):
    original = pd.read_parquet(path)
    graph_obs = original[original.field.isin(FIELDS)]
    aggs = {}
    directional_lanes = set(graph_obs.loc[graph_obs.field.eq('lanes') &
                                          graph_obs.direction.ne('both'), 'way_id'])
    graph_obs = graph_obs[graph_obs.direction.eq('both')].sort_values(
        ['way_id', 'field', 'start_frac', 'end_frac'])
    grouped = groupby(graph_obs.itertuples(index=False), key=lambda row: (row.way_id, row.field))
    for (way, field), group in grouped:
        if field == 'lanes' and way in directional_lanes:
            continue
        rows = list(group)
        direct = [row for row in rows if row.source == 'osm']
        if direct and all(row.start_frac == 0 and row.end_frac == 1 for row in direct):
            rows = direct
        spans = []
        for row in rows:
            item = Span(row.start_frac, row.end_frac, row.value, row.value_scope,
                        f'{row.source}:{row.source_ref}')
            if spans and item.lo < spans[-1].hi - EPS:
                raise ValueError(f'Overlapping source spans on OSM way {way} ({field})')
            if spans and item.lo <= spans[-1].hi + EPS and item.label == spans[-1].label:
                # Preserve independent measurement support even when adjacent values agree.
                prior = spans[-1]
                refs = '|'.join(sorted(set(prior.ref.split('|')) | {item.ref}))
                spans[-1] = Span(prior.lo, item.hi, item.value, item.scope, refs)
            else:
                spans.append(item)
        aggs[(way, field)] = spans
    return aggs


def bracketed(left, right, gap_m):
    # Agreement must use the same definition of width, lanes or surface.
    return left.label == right.label and EPS < gap_m <= MAX_GAP_M


def observation_row(way, lo, hi, field, left, right, method):
    return (way, lo, hi, 'both', field, left.value, '', 'continuity', method,
            'inferred_from_agreeing_anchors', left.scope,
            '|'.join(sorted(set(left.ref.split('|')) | set(right.ref.split('|')))))


def fill(spans_by_way, lengths, joins):
    for (way, field), spans in spans_by_way.items():
        if way not in lengths:
            continue
        for left, right in zip(spans, spans[1:]):
            if bracketed(left, right, (right.lo - left.hi) * lengths[way]):
                yield observation_row(way, left.hi, right.lo, field, left, right, 'within_way')
    for _, a, side_a, b, side_b in joins:
        for field in FIELDS:
            spans_a = spans_by_way.get((a, field))
            spans_b = spans_by_way.get((b, field))
            if not spans_a or not spans_b:
                continue
            anchor_a = spans_a[-1 if side_a else 0]
            anchor_b = spans_b[-1 if side_b else 0]
            lo_a, hi_a = (anchor_a.hi, 1.) if side_a else (0., anchor_a.lo)
            lo_b, hi_b = (anchor_b.hi, 1.) if side_b else (0., anchor_b.lo)
            gap_m = (hi_a - lo_a) * lengths[a] + (hi_b - lo_b) * lengths[b]
            if bracketed(anchor_a, anchor_b, gap_m):
                for way, lo, hi in [(a, lo_a, hi_a), (b, lo_b, hi_b)]:
                    if hi - lo > EPS:
                        yield observation_row(way, lo, hi, field, anchor_a, anchor_b, 'straight_join')


def sample_class_reference(target, all_data):
    votes = defaultdict(Counter)
    for city in PILOTS:
        if city == target:
            continue
        lengths, classes, _, spans_by_way = all_data[city]
        for (way, field), spans in spans_by_way.items():
            if way not in lengths:
                continue
            for span in spans:
                votes[(classes[way], field, span.scope)][span.value] += (
                    span.hi - span.lo) * lengths[way]
    out = {}
    for key, counts in votes.items():
        if key[1] == 'width':
            halfway = sum(counts.values()) / 2
            cumulative = 0.
            for v, weight in sorted(counts.items(), key=lambda pair: float(pair[0])):
                cumulative += weight
                if cumulative >= halfway:
                    out[key] = v
                    break
        else:
            out[key] = sorted(counts, key=lambda v: (-counts[v], v))[0]
    return out


def measured(spans, lo, hi):
    parts = [(span, min(span.hi, hi) - max(span.lo, lo)) for span in spans
             if span.hi > lo + EPS and span.lo < hi - EPS]
    if abs(sum(size for _, size in parts) - (hi - lo)) > EPS:
        return []
    return parts


def anchor_at(spans, position):
    return next((span for span in spans if span.lo <= position < span.hi), None)


def mask_examples(spans_by_way, lengths, classes, joins, class_ref):
    rng = np.random.default_rng(SEED)
    for (way, field), spans in spans_by_way.items():
        if way not in lengths or lengths[way] <= MIN_MASK_M:
            continue
        distance = float(rng.choice(MASK_M))
        gap = distance / lengths[way]
        if gap >= 1:
            continue
        # Sample continuous blocks from actual labelled support. Include windows around
        # observed transitions to avoid evaluating only artificially constant OSM ways.
        transitions = [left.hi for left, right in zip(spans, spans[1:])
                       if abs(left.hi - right.lo) <= EPS and left.label != right.label]
        bounds = [(span.lo, span.hi) for span in spans if span.hi - span.lo > gap + 2 * EPS]
        centers = []
        if bounds:
            proportions = np.array([hi - lo - gap for lo, hi in bounds])
            lo, hi = bounds[rng.choice(len(bounds), p=proportions / proportions.sum())]
            centers.append(('known_interior', rng.uniform(lo + gap / 2, hi - gap / 2)))
        if transitions:
            centers.append(('observed_transition', float(rng.choice(transitions))))
        for stratum, center in centers:
            lo, hi = center - gap / 2, center + gap / 2
            actual = measured(spans, lo, hi)
            if not actual or lo <= EPS or hi >= 1 - EPS:
                continue
            left, right = anchor_at(spans, lo - EPS), anchor_at(spans, hi + EPS)
            guess = left if left and right and bracketed(left, right, distance) else None
            yield from score(field, 'within_way', stratum, classes[way], actual,
                             guess, lengths[way], class_ref)
    for _, a, side_a, b, side_b in joins:
        for field in FIELDS:
            sides = []
            gap_m = float(rng.choice(MASK_M))
            for way, side in [(a, side_a), (b, side_b)]:
                gap = gap_m / (2 * lengths[way])
                if gap >= 1 or (way, field) not in spans_by_way:
                    break
                spans = spans_by_way[way, field]
                lo, hi = (1 - gap, 1.) if side else (0., gap)
                actual = measured(spans, lo, hi)
                if not actual:
                    break
                anchor = anchor_at(spans, lo - EPS if side else hi + EPS)
                sides.append((way, actual, anchor))
            if len(sides) != 2:
                continue
            left, right = [anchor for _, _, anchor in sides]
            guess = left if left and right and bracketed(left, right, gap_m) else None
            for way, actual, _ in sides:
                yield from score(field, 'straight_join', 'known_joint', classes[way], actual,
                                 guess, lengths[way], class_ref)


MIN_MASK_M = min(MASK_M)


def score(field, method, stratum, highway, actual, guess, full_length, class_ref):
    for span, fraction in actual:
        prediction = guess.value if guess and guess.scope == span.scope else None
        baseline = class_ref.get((highway, field, span.scope))
        yield (field, method, stratum, full_length * fraction, span.value, prediction, baseline)


def mask_report(examples):
    cols = ['field', 'method', 'stratum', 'masked_km', 'continuity_km', 'lookup_km',
            'continuity_exact_pct', 'lookup_exact_pct', 'lookup_exact_pct_on_common',
            'common_km', 'continuity_exact_pct_on_common', 'continuity_width_mae_m',
            'lookup_width_mae_m', 'continuity_width_mae_m_on_common', 'lookup_width_mae_m_on_common']
    rows = []
    for (field, method, stratum), samples in groupby(sorted(examples, key=lambda x: x[:3]),
                                                     key=lambda x: x[:3]):
        data = pd.DataFrame(samples, columns=['field', 'method', 'stratum', 'length_m',
                                               'actual', 'continuity', 'lookup'])
        k, r = data.continuity.notna(), data.lookup.notna()
        weights = data.length_m
        width = field == 'width'
        def accuracy(prediction, mask, use_mae=False):
            x = data.loc[mask]
            if x.empty:
                return np.nan
            error = ((x.actual.astype(float) - x[prediction].astype(float)).abs() if use_mae
                     else 100 * x.actual.eq(x[prediction]))
            return float(np.average(error, weights=x.length_m))
        rows.append((field, method, stratum, weights.sum() / 1000, weights[k].sum() / 1000,
                     weights[r].sum() / 1000, accuracy('continuity', k), accuracy('lookup', r),
                     accuracy('lookup', k & r), weights[k & r].sum() / 1000,
                     accuracy('continuity', k & r),
                     accuracy('continuity', k, True) if width else np.nan,
                     accuracy('lookup', r, True) if width else np.nan,
                     accuracy('continuity', k & r, True) if width else np.nan,
                     accuracy('lookup', k & r, True) if width else np.nan))
    return pd.DataFrame(rows, columns=cols)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cities', nargs='+', default=list(PILOTS))
    args = parser.parse_args()
    cities = sorted(p.name for p in (REPORT / 'observations').iterdir()
                    if (p / 'observations.parquet').exists()) if args.cities == ['all'] else args.cities
    all_data = {}
    for city in dict.fromkeys((*PILOTS, *cities)):
        lengths, classes, joins = way_geometry(REPORT / 'graph' / city)
        obs = observations(REPORT / 'observations' / city / 'observations.parquet')
        all_data[city] = lengths, classes, joins, obs
    for city in cities:
        lengths, classes, joins, obs = all_data[city]
        directory = REPORT / 'gaps' / city
        directory.mkdir(parents=True, exist_ok=True)
        fills = pd.DataFrame(fill(obs, lengths, joins), columns=OUTPUT_COLS).astype({
            'way_id': 'int64', 'start_frac': 'float64', 'end_frac': 'float64',
            **{col: 'string' for col in OUTPUT_COLS[3:]},
        })
        fills.to_parquet(directory / 'observations.parquet', index=False)
        km = ((fills.end_frac - fills.start_frac) * fills.way_id.map(lengths)) / 1000
        totals = fills.assign(km=km).groupby(['field', 'method']).agg(intervals=('km', 'size'),
                                                                      km=('km', 'sum'))
        totals.to_csv(directory / 'coverage.csv')
        reference = sample_class_reference(city, all_data)
        tests = mask_examples(obs, lengths, classes, joins, reference)
        mask_report(list(tests)).to_csv(directory / 'masked_validation.csv', index=False)
        (directory / 'metadata.json').write_text(json.dumps({
            'city': city, 'methods': ['within_way', 'straight_join'],
            'max_gap_m': MAX_GAP_M, 'mask_m': MASK_M, 'seed': SEED,
            'comparison_donors': [c for c in PILOTS if c != city],
            'straight_join': 'same name/class/service/bridge/tunnel/access; degree two; deflection <=25 degrees',
            'count_km': 'sum over fields (overlap across fields counted separately)',
            'rows': len(fills), 'km_by_field_sum': float(km.sum()), 'graph_joints': len(joins),
        }, indent=2) + '\n')
        print(f'{city}: {len(fills)} intervals, {km.sum():.3f} km summed over fields', flush=True)


if __name__ == '__main__':
    main()
