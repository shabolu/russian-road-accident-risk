"""Compare fixed shortest path samples and aggregate route passages into H3."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import zip_longest
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import shapely
from pyproj import Geod
from shapely.geometry import Polygon
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from road_accident_risk.route_cost import edge_costs
from road_accident_risk.route_graph import Graph

INPUT = ROOT / 'reports/multicity_350k/features'
EXPERIMENTS = ROOT / 'reports/route_proxy_experiments'
GEOD = Geod(ellps='WGS84')
ORIGIN_BANDS = (('core', 0, 5), ('outer', 5, 15), ('periphery', 15, np.inf))
TRIP_BANDS = (('short', 0.5, 3), ('medium', 3, 10), ('long', 10, np.inf))
PROFILES = ('baseline', 'osm', 'strict', 'continuity', 'inferred', 'strict_relaxed_service', 'length')


def load_cells(city: str) -> pd.DataFrame:
    cells = pd.read_parquet(INPUT / f'{city}.parquet', columns=['h3', 'center_distance_km'])
    if cells.h3.duplicated().any() or not all(h3.get_resolution(cell) == 9 for cell in cells.h3):
        raise ValueError(f'Expected unique H3 resolution 9 cells for {city}')
    cells = cells.sort_values('h3').reset_index(drop=True)
    cells[['lat', 'lon']] = [h3.cell_to_latlng(cell) for cell in cells.h3]
    cells['tile'] = [h3.cell_to_parent(cell, 7) for cell in cells.h3]
    return cells


def spatial_draw(cells: pd.DataFrame, size: int, rng: np.random.Generator) -> list[int]:
    """Draw cells round robin from shuffled coarser H3 groups."""
    pools = [rng.permutation(group.index).tolist() for _, group in cells.groupby('tile')]
    tile_order = rng.permutation(len(pools))
    selected = []
    for round_rows in zip_longest(*(pools[i] for i in tile_order)):
        for idx in round_rows:
            if idx is not None:
                selected.append(idx)
                if len(selected) == size:
                    return selected
    return selected


def sample_pairs(cells: pd.DataFrame, *, origins_per_band: int, targets_per_band: int,
                 seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pairs = []
    lon, lat = cells.lon.to_numpy(), cells.lat.to_numpy()
    for origin_band, lo, hi in ORIGIN_BANDS:
        eligible = cells[(cells.center_distance_km >= lo) & (cells.center_distance_km < hi)]
        for source in spatial_draw(eligible, origins_per_band, rng):
            origin = cells.loc[source]
            _, _, meters = GEOD.inv(
                np.full(len(cells), origin.lon), np.full(len(cells), origin.lat), lon, lat,
            )
            km = meters / 1_000
            for trip_band, min_km, max_km in TRIP_BANDS:
                target_pool = cells.loc[(km >= min_km) & (km < max_km)]
                for target in spatial_draw(target_pool, targets_per_band, rng):
                    pairs.append({
                        'pair_id': len(pairs),
                        'origin_idx': source,
                        'target_idx': target,
                        'origin_h3': origin.h3,
                        'target_h3': cells.loc[target, 'h3'],
                        'origin_band': origin_band,
                        'trip_band': trip_band,
                        'direct_distance_km': km[target],
                        'weight': 1.0,
                    })
    if not pairs:
        raise ValueError('No origin-destination pairs available')
    return pd.DataFrame(pairs)


def snap_pairs(graph: Graph, cells: pd.DataFrame, pairs: pd.DataFrame,
               max_snap_m: float) -> pd.DataFrame:
    selected = np.union1d(pairs.origin_idx, pairs.target_idx)
    points = cells.loc[selected, ['lon', 'lat']].to_numpy()
    nodes = graph.snap(points, public_only=True)
    coordinates = graph.nodes.set_index('node_id').loc[nodes, ['lon', 'lat']].to_numpy()
    _, _, meters = GEOD.inv(points[:, 0], points[:, 1], coordinates[:, 0], coordinates[:, 1])
    by_cell = pd.DataFrame({'node_id': nodes, 'meters': meters}, index=selected)
    for which in ('origin', 'target'):
        match = by_cell.loc[pairs[f'{which}_idx']]
        pairs[f'{which}_node'] = match.node_id.to_numpy()
        pairs[f'{which}_snap_m'] = match.meters.to_numpy()
    pairs['snap_ok'] = pairs[['origin_snap_m', 'target_snap_m']].max(axis=1) <= max_snap_m
    pairs['distinct_nodes'] = pairs.origin_node.ne(pairs.target_node)
    return pairs.drop(columns=['origin_idx', 'target_idx'])


def profile_costs(graph: Graph, profile: str, observations: pd.DataFrame | None,
                  gap_rows: pd.DataFrame | None, estimated: pd.DataFrame | None) -> np.ndarray:
    if profile == 'baseline':
        return graph.edges.base_cost_s.to_numpy(dtype=float)
    if profile == 'length':
        return graph.edges.length_m.to_numpy(dtype=float)
    if observations is None:
        raise ValueError(f'{profile} requires observed attribute rows')
    rows = observations.loc[observations.source.eq('osm')] if profile == 'osm' else observations
    if profile == 'continuity':
        if gap_rows is None:
            raise ValueError('continuity requires --gap-rows')
        rows = pd.concat([rows, gap_rows], ignore_index=True)
    if profile == 'inferred':
        if estimated is None:
            raise ValueError('inferred requires --estimated-rows')
        additions = [estimated] if gap_rows is None else [gap_rows, estimated]
        rows = pd.concat([rows, *additions], ignore_index=True)
    return edge_costs(graph.edges, rows)


def through_service(is_service: np.ndarray) -> bool:
    """A service run is through traffic if public roads occur on both sides."""
    nonservice = np.flatnonzero(~is_service)
    return bool(len(nonservice) > 1 and is_service[nonservice[0]:nonservice[-1]].any())


def run_routes(graph: Graph, pairs: pd.DataFrame, profile: str, costs: np.ndarray) -> pd.DataFrame:
    router = graph.prepare(costs, relax_service_local=(profile == 'strict_relaxed_service'))
    eligible = pairs[pairs.snap_ok & pairs.distinct_nodes]
    routes = {}
    for source, group in eligible.groupby('origin_node', sort=False):
        destinations = group.target_node.unique()
        for destination, route in router.shortest_paths(int(source), destinations).items():
            routes[int(source), int(destination)] = route
    lengths = graph.edges.length_m.to_numpy(dtype=float)
    service = graph.edges.highway.eq('service').to_numpy()
    rows = []
    for pair in pairs.itertuples(index=False):
        route = routes.get((pair.origin_node, pair.target_node))
        row = {'pair_id': pair.pair_id, 'profile': profile, 'weight': pair.weight,
               'found': route is not None, 'status': 'found', 'edge_ids': None, 'cost': None,
               'route_length_m': None, 'service_length_m': None, 'through_service': None}
        if route is None:
            row['status'] = ('snap_limit' if not pair.snap_ok
                             else 'same_node' if not pair.distinct_nodes else 'unreachable')
        else:
            ids = np.asarray(route.edge_ids, dtype=int)
            if not len(ids):
                raise ValueError(f'Empty path for distinct nodes on pair {pair.pair_id}')
            local = service[ids]
            row.update(edge_ids=ids.tolist(), cost=route.cost_s, route_length_m=lengths[ids].sum(),
                       service_length_m=lengths[ids[local]].sum(),
                       through_service=through_service(local))
        rows.append(row)
    return pd.DataFrame(rows)


def edge_cell_intersections(edges: pd.DataFrame, cells: pd.DataFrame,
                            used_edges: set[int]) -> dict[int, list[int]]:
    """Map used directed edges to model cells by positive length intersection."""
    if not used_edges:
        return {}
    polygons = np.asarray([Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(cell)])
                           for cell in cells.h3])
    edge_ids = np.asarray(sorted(used_edges), dtype=int)
    lines = shapely.from_wkb(edges.geometry.to_numpy()[edge_ids])
    edge_pos, cell_pos = STRtree(polygons).query(lines, predicate='intersects')
    crossed = shapely.length(shapely.intersection(lines[edge_pos], polygons[cell_pos])) > 1e-12
    mapping = {}
    for edge_id, cell_idx in zip(edge_ids[edge_pos[crossed]], cell_pos[crossed]):
        mapping.setdefault(int(edge_id), []).append(int(cell_idx))
    return mapping


def aggregate(city: str, routes: pd.DataFrame, pairs: pd.DataFrame, cells: pd.DataFrame,
              edges: pd.DataFrame) -> pd.DataFrame:
    used_edges = {i for edge_sequence in routes.edge_ids.dropna() for i in edge_sequence}
    mapping = edge_cell_intersections(edges, cells, used_edges)
    denominator = pairs.weight.sum()
    frames = []
    for profile, group in routes.groupby('profile', sort=False):
        counts = np.zeros(len(cells), dtype=int)
        weights = np.zeros(len(cells), dtype=float)
        for row in group[group.found].itertuples(index=False):
            visited = list({cell_idx for i in row.edge_ids for cell_idx in mapping.get(i, ())})
            counts[visited] += 1
            weights[visited] += row.weight
        if (weights > denominator + 1e-12).any():
            raise ValueError('Cell receives more than one contribution per planned route')
        frames.append(pd.DataFrame({'city': city, 'h3': cells.h3, 'profile': profile,
                                    'route_proxy': weights / denominator, 'route_weight': weights,
                                    'route_count': counts}))
    return pd.concat(frames, ignore_index=True)


def profile_summary(routes: pd.DataFrame, features: pd.DataFrame, baseline: str) -> dict:
    result = {}
    reference = routes[routes.profile.eq(baseline)].set_index('pair_id')
    baseline_values = features[features.profile.eq(baseline)].set_index('h3').route_proxy
    size = int(np.ceil(0.1 * len(baseline_values)))
    baseline_top = set(baseline_values.sort_values(ascending=False, kind='stable').head(size).index)
    for profile, group in routes.groupby('profile', sort=False):
        group = group.set_index('pair_id')
        successful = group[group.found]
        jointly_found = group.found & reference.found
        changed = [tuple(a) != tuple(b) for a, b in zip(
            group.loc[jointly_found, 'edge_ids'], reference.loc[jointly_found, 'edge_ids'],
        )]
        values = features[features.profile.eq(profile)].set_index('h3').route_proxy
        corr = values.rank().corr(baseline_values.rank()) if values.nunique() > 1 and baseline_values.nunique() > 1 else None
        top = set(values.sort_values(ascending=False, kind='stable').head(size).index)
        result[profile] = {
            'found': int(group.found.sum()),
            'found_share_of_planned_weight': float(successful.weight.sum() / group.weight.sum()),
            'status_counts': group.status.value_counts().to_dict(),
            'both_found': int(jointly_found.sum()),
            'changed_edge_sequence': int(sum(changed)),
            'changed_share_of_both_found': float(np.mean(changed)) if changed else None,
            'service_share_of_route_km': (float(successful.service_length_m.sum() / successful.route_length_m.sum())
                                          if len(successful) else None),
            'through_service_share_of_found': float(successful.through_service.mean()) if len(successful) else None,
            'cells_with_positive_proxy': int(values.gt(0).sum()),
            'h3_spearman': corr,
            'h3_top_10pct_overlap': len(top & baseline_top) / size,
        }
    return result


def read_optional(path: Path | None) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--city', default='saratov')
    parser.add_argument('--profiles', nargs='+', default=['baseline', 'osm', 'strict'], choices=PROFILES)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--origins-per-band', default=24, type=int)
    parser.add_argument('--targets-per-band', default=4, type=int)
    parser.add_argument('--max-snap-m', default=350, type=float)
    parser.add_argument('--graph-dir', type=Path)
    parser.add_argument('--allow-service', action='store_true', help='Reproduce historical routes on an explicit graph')
    parser.add_argument('--observations', type=Path)
    parser.add_argument('--gap-rows', type=Path)
    parser.add_argument('--estimated-rows', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if min(args.origins_per_band, args.targets_per_band, args.max_snap_m) <= 0:
        parser.error('sample sizes and max snap distance must be positive')
    variants = list(dict.fromkeys(args.profiles))
    city = args.city
    if args.allow_service and (args.graph_dir is None or args.out is None):
        parser.error('Historical mode requires explicit --graph-dir and --out')
    graph_dir = args.graph_dir or EXPERIMENTS / 'graph_no_service' / city
    out = args.out or EXPERIMENTS / 'h3_no_service' / city
    graph = Graph.load(graph_dir)
    service_edges = int(graph.edges.highway.eq('service').sum())
    if service_edges and not args.allow_service:
        raise ValueError('Routing graph includes highway=service. Rebuild the graph without service.')
    cells = load_cells(city)
    pairs = sample_pairs(cells, origins_per_band=args.origins_per_band,
                         targets_per_band=args.targets_per_band, seed=args.seed)
    pairs = snap_pairs(graph, cells, pairs, args.max_snap_m)
    observations = None
    if set(variants) - {'baseline', 'length'}:
        obs_path = args.observations or EXPERIMENTS / 'observations' / city / 'observations.parquet'
        observations = pd.read_parquet(obs_path)
    gap_rows, estimated = read_optional(args.gap_rows), read_optional(args.estimated_rows)
    frames = []
    for profile in variants:
        costs = profile_costs(graph, profile, observations, gap_rows, estimated)
        frame = run_routes(graph, pairs, profile, costs)
        frames.append(frame)
        print(f'{city} {profile}: {frame.found.sum()}/{len(frame)} found', flush=True)
    routes = pd.concat(frames, ignore_index=True)
    features = aggregate(city, routes, pairs, cells, graph.edges)
    baseline = 'baseline' if 'baseline' in variants else variants[0]
    summary = {
        'city': city, 'seed': args.seed, 'profiles': variants,
        'graph_directory': str(graph_dir), 'cost_reference': baseline,
        'graph_service_edges': service_edges,
        'n_h3': len(cells), 'n_planned_pairs': len(pairs), 'planned_weight': float(pairs.weight.sum()),
        'origins_per_band': args.origins_per_band, 'targets_per_band': args.targets_per_band,
        'origin_distance_edges_km': [0, 5, 15, None],
        'trip_distance_edges_km': [0.5, 3, 10, None],
        'max_snap_m': args.max_snap_m, 'snap_node_scope': 'public routable OSM nodes',
        'geometry': 'positive-length intersections of WGS84 road lines and planar H3 boundary polygons',
        'weighting': 'equal planned OD weights; denominator includes unsnapped and unreachable pairs',
        'strata': pairs.groupby(['origin_band', 'trip_band']).size().rename('n').reset_index().to_dict('records'),
        'metrics': profile_summary(routes, features, baseline),
    }
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out / 'od_pairs.parquet', index=False)
    routes.to_parquet(out / 'routes.parquet', index=False)
    features.to_parquet(out / 'features.parquet', index=False)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
