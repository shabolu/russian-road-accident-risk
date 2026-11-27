"""City-held-out test of a single geometry-derived route-passage feature."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd

from multicity_models import FEATURES, OUT as BASE, cb, evaluate, load, vy
from multicity_equal_city_weights import OUT as CONTROL, SEEDS, YEARS

EXPERIMENTS = BASE.parent / 'route_proxy_experiments'
OUT = EXPERIMENTS / 'evaluation_no_service'
METRIC = 'coverage_top_10pct'
PAIRED_METRICS = (METRIC, 'coverage_top_1pct', 'coverage_top_5pct', 'poisson_deviance')
PRIMARY_PROFILES = ['baseline', 'strict']
RECIPE = dict(iterations=800, learning_rate=.05, depth=6)
LOSS = 'Tweedie:variance_power=1.3'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_write(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        writer(tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path, data):
    atomic_write(path, lambda tmp: tmp.write_text(json.dumps(data, indent=2) + '\n'))


def write_csv(path, data):
    atomic_write(path, lambda tmp: data.to_csv(tmp, index=False))


def check_cached_control():
    expected = json.loads((CONTROL / 'input_fingerprints.json').read_text())
    for name, fingerprint in expected.items():
        if digest(BASE.parent.parent / name) != fingerprint:
            raise ValueError(f'Cached control input changed: {name}')
    control = json.loads((CONTROL / 'protocol.json').read_text())
    if (control['features'] != FEATURES or control['train_years'] != YEARS
            or control['test_years'] != [2023, 2024, 2025]):
        raise ValueError('Cached control protocol does not match experiment')
    return expected


def load_profiles(cities, frames, root, profiles):
    fingerprints = {}
    for city in cities:
        path = root / city / 'features.parquet'
        frame = pd.read_parquet(path, columns=['city', 'h3', 'profile', 'route_proxy'])
        if not frame.city.eq(city).all():
            raise ValueError(f'Mixed city names in {path}')
        frame = frame.loc[frame.profile.isin(profiles)]
        selected = frame.pivot(index='h3', columns='profile', values='route_proxy')
        if set(selected.columns) != set(profiles) or set(selected.index) != set(frames[city].h3):
            raise ValueError(f'Incomplete experiment profile or H3 coverage for {city}')
        values = selected.to_numpy()
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError(f'Invalid proxy values in {path}')
        enriched = frames[city].merge(selected.add_prefix('route_').reset_index(), on='h3',
                                      how='left', validate='one_to_one', sort=False)
        if not enriched.h3.equals(frames[city].h3):
            raise ValueError(f'Reordered original H3 cohort in {city}')
        frames[city] = enriched
        # Fingerprint each feature independently so extra profiles can be compared later.
        ordered = selected.sort_index()
        fingerprints[city] = {p: hashlib.sha256(pd.util.hash_pandas_object(ordered[p], index=True)
                                                .to_numpy().tobytes()).hexdigest() for p in profiles}
    return fingerprints


def fold_data(frames, target, feature_names):
    donors = {city: f for city, f in frames.items() if city != target}
    held_out = frames[target]
    held_out_cells = set(held_out.h3)
    if any(held_out_cells.intersection(frame.h3) for frame in donors.values()):
        raise ValueError(f'Test H3 cells enter training for {target}')
    train = pd.concat([frame[feature_names].assign(city=city, target=frame[str(year)].to_numpy())
                       for city, frame in donors.items() for year in YEARS], ignore_index=True)
    counts = train.groupby('city').size()
    weights = train.city.map(len(train) / (len(counts) * counts)).to_numpy()
    totals = pd.Series(weights).groupby(train.city).sum()
    if not np.allclose(totals, totals.iloc[0]) or not np.isclose(weights.mean(), 1):
        raise ValueError(f'City weights do not balance in {target}')
    caps = cb.fit_p99_clip_caps(train, feature_names)
    return cb.apply_p99_clipping(train, caps), cb.apply_p99_clipping(held_out, caps), weights, caps


def fit_predict(train, held_out, weights, feature_names, seed, threads):
    model = vy.make_model(loss=LOSS, seed=seed, params={**RECIPE, 'thread_count': threads})
    model.fit(train[feature_names], train.target.astype(int), sample_weight=weights)
    predictions = model.predict(held_out[feature_names], prediction_type='Exponent', thread_count=threads)
    if not np.isfinite(predictions).all() or (predictions <= 0).any():
        raise ValueError('Tweedie mean predictions must be finite and positive')
    return predictions


def verify_parity(frames, out, threads):
    city, seed = 'saratov', 42
    train, test, weights, caps = fold_data(frames, city, FEATURES)
    expected_caps = json.loads((CONTROL / f'{city}_caps.json').read_text())
    if caps != expected_caps:
        raise ValueError('Cached control clipping caps changed')
    cached = pd.read_parquet(CONTROL / f'{city}_predictions.parquet')
    cached = cached.loc[cached.seed.eq(seed)].set_index('h3').loc[test.h3, 'predicted'].to_numpy()
    predictions = 3 * fit_predict(train, test, weights, FEATURES, seed, threads)
    np.testing.assert_allclose(predictions, cached, atol=1e-10, rtol=1e-10)
    result = dict(city=city, seed=seed, n_predictions=len(cached), exact=np.array_equal(predictions, cached),
                  max_absolute_error=float(np.max(np.abs(cached - predictions))))
    write_json(out / 'control_parity.json', result)
    print('CONTROL PARITY', json.dumps(result), flush=True)


def checkpoint_path(out, city, profile, seed):
    return out / 'folds' / city / profile / f'{seed}.json'


def run_city(frames, city, seeds, signatures, out, threads):
    for profile, signature in signatures.items():
        pending = []
        for seed in seeds:
            path = checkpoint_path(out, city, profile, seed)
            if not path.exists() or json.loads(path.read_text())['signature'] != signature:
                pending.append((seed, path))
        if not pending:
            continue
        feature_names = signature['features']
        train, test, weights, caps = fold_data(frames, city, feature_names)
        for seed, path in pending:
            started = time.monotonic()
            predictions = fit_predict(train, test, weights, feature_names, seed, threads)
            result = evaluate(test, predictions, {'profile': profile, 'target': city, 'seed': seed})
            elapsed = round(time.monotonic() - started, 2)
            write_json(path, dict(signature=signature, caps=caps, metrics=result, fit_seconds=elapsed))
            score = next(row[METRIC] for row in result
                         if row['subset'] == 'municipality' and row['period'] == 'sum')
            print(f'FIT {city} {profile} seed={seed} coverage@10={score:.6f} seconds={elapsed}', flush=True)


def update_summary(out, targets, seeds, signatures):
    reference = pd.read_csv(CONTROL / 'metrics.csv', dtype={'period': str})
    reference = reference.loc[reference.target.isin(targets) & reference.seed.isin(seeds)].copy()
    reference['profile'] = 'no_route'
    frames = [reference.drop(columns=['kind', 'source', 'window'])]
    for city in targets:
        for profile, signature in signatures.items():
            for seed in seeds:
                path = checkpoint_path(out, city, profile, seed)
                if path.is_file():
                    result = json.loads(path.read_text())
                    if result['signature'] == signature:
                        frames.append(pd.DataFrame(result['metrics']))
    metrics = pd.concat(frames, ignore_index=True)
    write_csv(out / 'metrics.csv', metrics)

    joined = metrics.pivot(index=['target', 'seed', 'subset', 'period'], columns='profile',
                           values=list(PAIRED_METRICS))
    main = joined.xs(('municipality', 'sum'), level=('subset', 'period'))
    comparisons = [(profile, 'no_route') for profile in signatures]
    comparisons += [(profile, 'baseline') for profile in signatures if profile != 'baseline']
    paired, summary = [], []
    for model, control in comparisons:
        needed = [(metric, name) for metric in PAIRED_METRICS for name in (model, control)]
        if any(column not in main for column in needed):
            continue
        records = []
        for city, group in main[needed].dropna().groupby(level='target'):
            scores = dict(target=city, profile=model, control=control, n_seeds=len(group))
            for metric in PAIRED_METRICS:
                scores[metric] = group[metric][model].mean()
                scores[f'control_{metric}'] = group[metric][control].mean()
                scores[f'delta_{metric}'] = (group[metric][model] - group[metric][control]).mean()
            records.append(scores)
        if not records:
            continue
        paired.extend(records)
        complete = pd.DataFrame([row for row in records if row['n_seeds'] == len(seeds)])
        if complete.empty:
            continue
        delta = complete[f'delta_{METRIC}']
        lower_quartile = max(1, int(np.ceil(len(complete) / 4)))
        weakest = complete.nsmallest(lower_quartile, f'control_{METRIC}')
        summary.append(dict(profile=model, control=control, completed_cities=len(complete),
                            requested_cities=len(targets), seeds=len(seeds),
                            macro_coverage_at_10=complete[METRIC].mean(),
                            control_macro_coverage_at_10=complete[f'control_{METRIC}'].mean(),
                            macro_delta_coverage_at_10=delta.mean(),
                            weak_control_cities=','.join(weakest.target),
                            weak_control_cities_delta=weakest[f'delta_{METRIC}'].mean(),
                            lowest_quartile_mean=complete[METRIC].nsmallest(lower_quartile).mean(),
                            control_lowest_quartile_mean=complete[f'control_{METRIC}'].nsmallest(lower_quartile).mean(),
                            improved_cities=int((delta > 0).sum()),
                            worsened_cities=int((delta < 0).sum())))
    if paired:
        write_csv(out / 'city_paired.csv', pd.DataFrame(paired))
    if summary:
        write_csv(out / 'summary.csv', pd.DataFrame(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', '--target', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--profiles', nargs='+', default=PRIMARY_PROFILES)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--h3-dir', type=Path, default=EXPERIMENTS / 'h3_no_service')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--verify-parity', action='store_true')
    parser.add_argument('--parity-only', action='store_true')
    args = parser.parse_args()
    if min(args.workers, args.threads) < 1 or args.workers * args.threads > 4:
        parser.error('Use one to four threads in total across all workers')
    if set(args.seeds) - set(SEEDS) or len(set(args.seeds)) != len(args.seeds):
        parser.error(f'Use distinct seeds from {SEEDS} to pair with the cached control')
    if len(set(args.profiles)) != len(args.profiles) or any(not p.isidentifier() for p in args.profiles):
        parser.error('Use distinct H3 profile identifiers')
    baseline_hashes = check_cached_control()
    _, frames = load()
    if args.verify_parity or args.parity_only:
        verify_parity(frames, args.out, args.threads)
    if args.parity_only:
        return
    targets = args.targets or list(frames)
    if set(targets) - frames.keys() or len(set(targets)) != len(targets):
        parser.error('Use distinct target city slugs from the 14-city cohort')
    route_hashes = load_profiles(frames, frames, args.h3_dir, args.profiles)
    signatures = {profile: dict(control_inputs=baseline_hashes,
                                route_inputs={c: route_hashes[c][profile] for c in frames},
                                loss=LOSS, recipe=RECIPE, train_years=YEARS,
                                features=FEATURES + [f'route_{profile}']) for profile in args.profiles}
    write_json(args.out / 'protocol.json', dict(
        cities=list(frames), requested_targets=targets, seeds=args.seeds, profiles=args.profiles,
        train_years=YEARS, test_years=[2023, 2024, 2025],
        control='Cached equal-city results with source fingerprints and optional prediction parity',
        added_feature='Share of fixed OD sample routes crossing the H3 cell, one contribution per route',
        model=dict(**RECIPE, loss=LOSS), clipping='Unweighted p99 fit on training rows only',
        weights='Equal total training weight per donor city, normalized to mean one',
        interpretation='Retrospective experiment reconstructing earlier features from 2026 OSM and SKDF',
        primary_metric='Municipal coverage@10 across 2023-2025, averaged across seeds then cities',
        lower_quartile='Mean of the lowest ceil(n/4) city scores; also report delta on the control’s weakest cities',
        route_inputs=route_hashes, control_inputs=baseline_hashes,
    ))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        active = [pool.submit(run_city, frames, target, args.seeds, signatures, args.out, args.threads)
                  for target in targets]
        update_summary(args.out, targets, args.seeds, signatures)
        for future in as_completed(active):
            future.result()
            update_summary(args.out, targets, args.seeds, signatures)
    print('EVALUATION COMPLETE', str(args.out), flush=True)


if __name__ == '__main__':
    main()
