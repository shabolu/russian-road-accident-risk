"""Compare class defaults and out-of-city estimates for missing road attributes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'reports/route_proxy_experiments'
CACHE = ROOT / 'reports/multicity_350k/cache'
FIELDS = ('surface', 'lanes', 'width')
PRIORITY = {'osm': 0, 'skdf': 1, 'continuity': 2}
CATS = ['highway', 'service', 'oneway']
FEATURES = [*CATS, 'log_length', 'has_name', 'has_ref', 'lit', 'bridge', 'tunnel']
COLS = ['way_id', 'start_frac', 'end_frac', 'direction', 'field', 'value', 'raw_value',
        'source', 'method', 'quality', 'value_scope', 'source_ref']
KEYS = [CATS, CATS[::2], CATS[:1]]
PAVED = {'asphalt', 'concrete', 'concrete:plates', 'concrete:lanes', 'paved', 'cement',
         'paving_stones', 'sett', 'bricks', 'brick', 'cobblestone', 'cobblestone:flattened',
         'chipseal', 'капитальная', 'облегчённая'}
UNPAVED = {'unpaved', 'gravel', 'fine_gravel', 'compacted', 'ground', 'dirt', 'earth',
           'rocky', 'sand', 'mud', 'grass', 'stone', 'rock', 'pebblestone', 'clay',
           'unsealed', 'laterite', 'низшая'}
SURFACE = {**dict.fromkeys(PAVED, 'paved'), **dict.fromkeys(UNPAVED, 'unpaved')}


def load_city(city):
    roads = pd.read_parquet(CACHE / f'{city}_roads.parquet', columns=[
        'osm_id', 'highway', 'length_m', 'oneway', 'name', 'ref', 'lit', 'bridge', 'tunnel', 'tags_json',
    ]).rename(columns={'osm_id': 'way_id'})
    graph = pd.read_parquet(BASE / 'graph' / city / 'edges.parquet',
                            columns=['way_id', 'start_frac', 'end_frac', 'length_m'])
    # A reverse arc is the same physical road length.
    graph['low'] = graph[['start_frac', 'end_frac']].min(axis=1)
    graph['high'] = graph[['start_frac', 'end_frac']].max(axis=1)
    spans = graph.drop_duplicates(['way_id', 'low', 'high'])
    region = spans.groupby('way_id').agg(graph_m=('length_m', 'sum'),
                                        fraction=('high', 'sum'), first=('low', 'min'), last=('high', 'max'))
    region['fraction'] -= spans.groupby('way_id').low.sum()
    roads = roads.join(region, on='way_id', how='inner')
    roads['service'] = ''
    service = roads.highway.eq('service')
    roads.loc[service, 'service'] = roads.loc[service, 'tags_json'].map(lambda s: json.loads(s).get('service', ''))
    roads['oneway'] = roads.oneway.map({True: 'yes', False: 'no'}).fillna('unknown')
    roads['has_name'] = roads.name.notna() & roads.name.ne('')
    roads['has_ref'] = roads.ref.notna() & roads.ref.ne('')
    roads['log_length'] = np.log1p(roads.length_m)
    roads['city'] = city
    observations = pd.read_parquet(BASE / 'observations' / city / 'observations.parquet')
    label_rows = observations[observations.field.isin(FIELDS) & observations.direction.eq('both')].copy()
    material = label_rows.field.eq('surface')
    label_rows.loc[material, 'value'] = label_rows.loc[material, 'value'].map(SURFACE)
    label_rows = label_rows[material | label_rows.value_scope.eq('whole_carriageway')].dropna(subset=['value'])
    label_rows['priority'] = label_rows.source.map(PRIORITY)
    label_rows['fraction'] = label_rows.end_frac - label_rows.start_frac
    # Compare to unambiguous whole-way labels: at least 80% known with one value
    # from its highest-priority source. Partly known ways remain useful for inference.
    label_rows = label_rows[label_rows.priority.eq(label_rows.groupby(['way_id', 'field']).priority.transform('min'))]
    votes = label_rows.groupby(['way_id', 'field', 'value'], as_index=False).fraction.sum()
    winners = votes[votes.fraction.ge(.8)].sort_values('fraction').drop_duplicates(['way_id', 'field'], keep='last')
    labels = winners.pivot(index='way_id', columns='field', values='value').reindex(columns=FIELDS)
    # Labels quantify whole-way coverage; clipped ways are excluded from evaluation.
    labels = labels.reindex(roads.loc[roads.fraction.ge(1 - 1e-6), 'way_id'])
    for field in ('lanes', 'width'):
        labels[field] = pd.to_numeric(labels[field], errors='coerce')
    roads = roads.join(labels, on='way_id')
    ratio = roads.width / roads.lanes
    roads['width_per_lane'] = ratio.where(ratio.between(2, 6))
    return roads, observations, spans


def stable_mode(values):
    return values.dropna().mode().iat[0]


def fit_defaults(data):
    tables = {}
    for field in ('surface', 'lanes', 'width_per_lane'):
        labeled = data[data[field].notna()]
        tables[field] = []
        for keys in KEYS:
            grouped = labeled.groupby(keys, observed=True)
            agg = grouped.agg(n=('way_id', 'size'), donors=('city', 'nunique'),
                              value=(field, 'median' if field == 'width_per_lane' else stable_mode))
            tables[field].append(agg.loc[(agg.n >= 30) & (agg.donors >= 2), 'value'])
    return tables


def predict_defaults(data, tables):
    predictions = pd.DataFrame(index=data.index)
    for field, layers in tables.items():
        prediction = pd.Series(np.nan, index=data.index, dtype=object if field == 'surface' else float)
        for keys, lookup in zip(KEYS, layers):
            index = pd.Index(data[keys[0]]) if len(keys) == 1 else pd.MultiIndex.from_frame(data[keys])
            values = pd.Series(lookup.reindex(index).to_numpy(), index=data.index)
            prediction = prediction.where(prediction.notna(), values)
        predictions[field] = prediction
    return predictions


def fit_predict_models(train, target, defaults, seed, iterations):
    out = defaults.copy()
    for field in ('surface', 'lanes', 'width_per_lane'):
        train_rows = train[train[field].notna()]
        y = train_rows[field]
        if y.nunique() <= 1:
            continue
        klass = CatBoostRegressor if field == 'width_per_lane' else CatBoostClassifier
        model = klass(iterations=iterations, depth=6, learning_rate=.1, random_seed=seed,
                      cat_features=CATS, verbose=False, allow_writing_files=False, thread_count=2)
        if field == 'lanes':
            y = y.astype(int)
        weights = np.sqrt(train_rows.graph_m.to_numpy().clip(1, 1500))
        totals = pd.Series(weights, index=train_rows.index).groupby(train_rows.city).sum()
        weights /= train_rows.city.map(totals).to_numpy()
        weights /= weights.mean()
        model.fit(train_rows[FEATURES], y, sample_weight=weights)
        # Same domain and coverage as the lookup; no extrapolation to unsupported classes.
        eligible = defaults[field].notna()
        if not eligible.any():
            continue
        values = model.predict(target.loc[eligible, FEATURES]).reshape(-1)
        if field == 'width_per_lane':
            values = values.clip(2, 6)
        out.loc[eligible, field] = values
    return out


def score(target, prediction, method, split):
    scores = []
    for name, group in [('all', target), *target.groupby('highway', sort=True)]:
        p = prediction.loc[group.index]
        for field in FIELDS:
            if field == 'width':
                guess = p.lanes * p.width_per_lane
                # Width targets require compatible observed total width and lanes.
                known = group.width_per_lane.notna()
            else:
                guess = p[field]
                known = group[field].notna()
            compared = known & guess.notna()
            length = group.graph_m
            truth = group.loc[compared, field]
            estimate = guess[compared]
            weights = length[compared].to_numpy()
            error = ((estimate != truth).astype(float) if field == 'surface' else np.abs(estimate - truth))
            metric = 'misclass_rate' if field == 'surface' else 'mae_lanes' if field == 'lanes' else 'mae_m'
            scores.append(dict(city=group.city.iat[0], split=split, highway=name, method=method, field=field,
                               ways=len(group), network_km=length.sum() / 1000,
                               labeled_ways=int(known.sum()), labeled_km=length[known].sum() / 1000,
                               tested_km=weights.sum() / 1000, metric=metric,
                               error=float(np.average(error.to_numpy(dtype=float), weights=weights)) if len(weights) else np.nan))
    return scores


def export_rows(target, predictions, observations, spans, city, method, output):
    rows = []
    known = defaultdict(list)
    for row in observations[observations.field.isin(FIELDS)].itertuples(index=False):
        known[row.way_id].append(row)
    windows = {}
    clipped = target.loc[target.fraction.lt(1 - 1e-6), 'way_id']
    for way_id, group in spans[spans.way_id.isin(clipped)].groupby('way_id'):
        parts = []
        for lo, hi in group[['low', 'high']].sort_values('low').itertuples(index=False, name=None):
            if parts and lo <= parts[-1][1] + 1e-9:
                parts[-1] = parts[-1][0], max(hi, parts[-1][1])
            else:
                parts.append((lo, hi))
        windows[way_id] = parts
    lengths = defaultdict(float)
    keep = ['way_id', 'graph_m', 'fraction', 'highway']
    estimated = target[keep].join(predictions[['surface', 'lanes', 'width_per_lane']])
    for way_id, rm, frac, highway, surface, lanes, per_lane in estimated.itertuples(index=False, name=None):
        obs = known.get(way_id, [])
        for lo, hi in windows.get(way_id, [(0., 1.)]):
            cuts = {lo, hi}
            for row in obs:
                if row.start_frac < hi and row.end_frac > lo:
                    cuts.update((max(lo, row.start_frac), min(hi, row.end_frac)))
            bounds = sorted(cuts)
            for left, right in zip(bounds, bounds[1:]):
                if right - left < 1e-9:
                    continue
                midpoint = (left + right) / 2
                active = [row for row in obs if row.start_frac <= midpoint < row.end_frac]
                present = {row.field for row in active}
                compatible = {}
                for row in sorted(active, key=lambda r: PRIORITY[r.source]):
                    if row.field in ('lanes', 'width') and row.value_scope == 'whole_carriageway':
                        compatible.setdefault(row.field, float(row.value))
                # If we have an observed compatible width, preserve plausible width per lane.
                selected_lanes = compatible.get('lanes', lanes)
                if 'lanes' not in present and np.isfinite(selected_lanes) and 'width' in compatible:
                    w = compatible['width']
                    if not 2 <= w / selected_lanes <= 6:
                        selected_lanes = np.nan
                width = per_lane * selected_lanes if np.isfinite(selected_lanes) and np.isfinite(per_lane) else np.nan
                if width > 100 or ('lanes' in present and 'lanes' not in compatible):
                    width = np.nan
                for field, candidate in zip(FIELDS, (surface, selected_lanes, width)):
                    if field in present or pd.isna(candidate):
                        continue
                    value = candidate if field == 'surface' else f'{candidate:.0f}' if field == 'lanes' else f'{candidate:.2f}'
                    lengths[(highway, field)] += rm / frac * (right - left) / 1000
                    rows.append((way_id, left, right, 'both', field, value, '', method,
                                 'class_lookup_loco' if method == 'fallback' else 'catboost_loco',
                                 'estimated_low_confidence', 'coarse' if field == 'surface' else 'whole_carriageway', 'other_cities'))
    out = output / city / method
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLS).to_parquet(out / 'observations.parquet', index=False)
    return [dict(city=city, highway=highway, field=field, method=method, added_km=km)
            for (highway, field), km in sorted(lengths.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cities', nargs='+', default=['saratov', 'samara', 'ulyanovsk'])
    parser.add_argument('--output', type=Path, default=BASE / 'imputation')
    parser.add_argument('--iterations', type=int, default=120)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--methods', nargs='+', choices=('fallback', 'model'), default=['fallback', 'model'])
    parser.add_argument('--metrics-only', action='store_true')
    args = parser.parse_args()
    cities = sorted(path.parent.name for path in (BASE / 'observations').glob('*/observations.parquet')
                    if (BASE / 'graph' / path.parent.name / 'edges.parquet').is_file())
    targets = cities if args.cities == ['all'] else args.cities
    if not set(targets).issubset(cities) or len(cities) < 3:
        parser.error('Missing complete city inputs or fewer than three cities')
    args.output.mkdir(parents=True, exist_ok=True)
    frames = [load_city(city)[0] for city in cities]
    data = pd.concat(frames, ignore_index=True)
    all_scores, all_coverage = [], []
    for city in targets:
        target = data[data.city.eq(city)]
        # Shared ways on municipal buffer boundaries cannot enter a donor fold.
        train = data[~data.city.eq(city) & ~data.way_id.isin(target.way_id)]
        donor = train.drop_duplicates('way_id').copy()
        defaults = predict_defaults(target, fit_defaults(donor))
        variants = {'fallback': defaults}
        if 'model' in args.methods:
            variants['model'] = fit_predict_models(donor, target, defaults, args.seed, args.iterations)
        for method in dict.fromkeys(args.methods):
            predictions = variants[method]
            all_scores.extend(score(target, predictions, method, 'leave_city_out'))
        if not args.metrics_only:
            _, observations, spans = load_city(city)
            gap_path = BASE / 'gaps' / city / 'observations.parquet'
            if gap_path.is_file():
                observations = pd.concat([observations, pd.read_parquet(gap_path)], ignore_index=True)
            for method in dict.fromkeys(args.methods):
                all_coverage.extend(export_rows(target, variants[method], observations, spans, city, method, args.output))
        print(f'{city}: {len(donor):,} eligible donor ways from {donor.city.nunique()} cities, {len(target):,} test ways', flush=True)
        pd.DataFrame(all_scores).to_csv(args.output / 'city_scores.csv', index=False)
        pd.DataFrame(all_coverage).to_csv(args.output / 'city_estimated_coverage.csv', index=False)
    protocol = dict(cities=cities, targets=targets, methods=args.methods, seed=args.seed, iterations=args.iterations,
                    evaluation='Entire city and overlapping way IDs excluded. Length-weighted error on known labels.',
                    labels='Recognized paved/unpaved or numeric coverage on at least 80% of a whole OSM way.',
                    fallback='Class, service subtype and direction with simpler class fallback if needed. Minimum 30 labeled ways and 2 cities.',
                    learned='CatBoost on OSM covariates. Predict lane count plus plausible width per lane; estimate is advisory.',
                    metrics_only=args.metrics_only,
                    limitations='Known roads are selectively labeled. Scope is full buffered graph. SKDF transition surface excluded from labels.')
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')


if __name__ == '__main__':
    main()
