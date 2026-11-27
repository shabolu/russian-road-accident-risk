"""Export direction-aware OSM and matched SKDF observations for routing.

Intervals reference consecutive WGS84 geodesic distances along the complete
OSM way in node order. SKDF interval borders are estimates from sampled
matching, not surveyed boundaries. OSM rows retain the full way extent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Geod, Transformer
from shapely.geometry import LineString, box, shape

# Shared parsers and accepted code lists from the existing attribute audit.
from route_attribute_coverage import (BASE, RAW, ROOT, COATING_CLASS, COATING_KNOWN,
                                      SMOOTH, SURFACES, lines, numeric, parse_oneway_v2)

OUT = ROOT / 'reports/route_proxy_experiments/observations'
GEOD = Geod(ellps='WGS84')
SAMPLE_COLUMNS = ['way_index', 'x', 'y', 'angle', 'start_frac', 'end_frac', 'length_m']
FIELDS = ['maxspeed', 'surface', 'lanes', 'width', 'smoothness']
COLS = ['way_id', 'start_frac', 'end_frac', 'direction', 'field', 'value', 'raw_value',
        'source', 'method', 'quality', 'value_scope', 'source_ref']


def as_text(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f'{value:g}'
    return str(value)


def osm_rows(way_id, tags, oneway):
    def item(field, direction, value, raw_value, scope):
        return (way_id, 0.0, 1.0, direction, field, as_text(value), as_text(raw_value),
                'osm', 'osm_tag', 'direct_tag', scope, str(way_id))

    speed = tags.get('maxspeed')
    speeds = {k: tags.get(f'maxspeed:{k}') for k in ('forward', 'backward')}
    if all(v is None for v in speeds.values()):
        speeds = {'both': speed}
    else:
        speeds = {k: speed if v is None else v for k, v in speeds.items()}
    for direction, raw in speeds.items():
        value = numeric(raw, 5, 150, speed=True)
        if np.isfinite(value):
            yield item('maxspeed', direction, value, raw, 'speed_limit')

    total = tags.get('lanes')
    value = numeric(total, 1, 20, integer=True)
    if np.isfinite(value):
        lane_rows = [item('lanes', 'both', value, total, 'whole_carriageway')]
    else:
        lane_rows = []
    known_directional = {}
    for direction in ('forward', 'backward'):
        raw = tags.get(f'lanes:{direction}')
        value = numeric(raw, 1, 20, integer=True)
        if np.isfinite(value):
            known_directional[direction] = value, raw
            lane_rows.append(item('lanes', direction, value, raw, 'travel_direction'))
    if not oneway and not np.isfinite(numeric(total, 1, 20, integer=True)) and len(known_directional) == 2:
        combined = sum(v for v, _ in known_directional.values())
        if combined <= 20:
            raw = ' + '.join(as_text(v) for _, v in known_directional.values())
            lane_rows.append((way_id, 0.0, 1.0, 'both', 'lanes', as_text(combined), raw,
                              'osm', 'directional_lane_sum', 'derived_from_tags', 'whole_carriageway',
                              str(way_id)))
    yield from lane_rows

    width = tags.get('width')
    value = numeric(width, 1, 100, unit=True)
    if np.isfinite(value):
        yield item('width', 'both', value, width, 'whole_carriageway')
    for field, allowed, scope in (('surface', SURFACES, 'material'), ('smoothness', SMOOTH, 'condition')):
        raw = tags.get(field)
        value = str(raw or '').strip().lower()
        if value in allowed:
            yield item(field, 'both', value, raw, scope)


def collect_ways(slug, transformer, area):
    data = pd.read_parquet(BASE / 'cache' / f'{slug}_roads.parquet',
                           columns=['osm_id', 'name', 'highway', 'coords_json', 'tags_json'])
    if not data.osm_id.is_unique:
        raise ValueError(f'OSM way IDs are not unique: {slug}')
    ways, buffers, rows = [], [], []
    for row in data.itertuples(index=False):
        tags = json.loads(row.tags_json)
        oneway = parse_oneway_v2(tags) or (row.highway == 'motorway' and 'oneway' not in tags)
        divided = oneway or tags.get('dual_carriageway') == 'yes'
        rows.extend(osm_rows(row.osm_id, tags, oneway))
        lat, lon = np.asarray(json.loads(row.coords_json)).T
        xy = np.column_stack(transformer.transform(lon, lat))
        whole = LineString(xy)
        if not shapely.intersects(area, whole):
            continue
        deltas = np.diff(xy, axis=0)
        metric = np.linalg.norm(deltas, axis=1)
        geodesic = np.asarray(GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])[2])
        accumulated = np.r_[0, geodesic.cumsum()]
        fractions = accumulated / accumulated[-1]
        way_index = len(ways)
        permitted = str(tags.get('oneway', '')).strip().lower()
        travel = ('backward' if permitted in {'-1', 'reverse'} else 'forward') if oneway else 'both'
        ways.append((row.osm_id, row.highway, row.name, divided, travel))
        if shapely.covers(area, whole):
            segments = ((i, 0., 1.) for i in np.flatnonzero(metric > 1e-6))
        else:
            def clipped_segments():
                for i in np.flatnonzero(metric > 1e-6):
                    for part in lines(LineString(xy[i:i+2]).intersection(area)):
                        ends = np.asarray(part.coords)[[0, -1]]
                        a, b = np.sort(np.clip((ends - xy[i]) @ deltas[i] / metric[i] ** 2, 0, 1))
                        if metric[i] * (b - a) > 1e-6:
                            yield i, a, b

            segments = clipped_segments()
        for i, a, b in segments:
            length = metric[i] * (b - a)
            count = max(1, int(np.ceil(length / 20)))
            ends = np.linspace(a, b, count + 1)
            midpoints = xy[i] + (ends[:-1] + ends[1:])[:, None] * deltas[i] / 2
            along = fractions[i] + ends * (fractions[i+1] - fractions[i])
            angle = np.degrees(np.arctan2(*deltas[i])) % 180
            buffers.append(np.column_stack((np.full(count, way_index), midpoints,
                                            np.full(count, angle), along[:-1], along[1:],
                                            np.full(count, length / count))))
    if not buffers:
        raise ValueError(f'No OSM observations inside sample area: {slug}')
    return (pd.DataFrame(ways, columns=['way_id', 'highway', 'name', 'divided', 'travel']),
            pd.DataFrame(np.concatenate(buffers), columns=SAMPLE_COLUMNS),
            pd.DataFrame(rows, columns=COLS))


def collect_passports(raw, project, clipping):
    records, segments, record_indices, angles = [], [], [], []
    for feature in raw['features']:
        if not feature.get('geometry'):
            continue
        area = project(shape(feature['geometry'])).intersection(clipping)
        if area.is_empty:
            continue
        p = feature['properties']
        code = p.get('pavement_kind_gid')
        values = {'maxspeed': numeric(p.get('speed_limit'), 5, 150),
                  'lanes': numeric(p.get('lanes'), 1, 20, integer=True),
                  'width': numeric(p.get('roadway_width'), 1, 100),
                  'surface': COATING_CLASS[code] if code in COATING_KNOWN else None}
        record_index = len(records)
        raw_fields = {'maxspeed': p.get('speed_limit'), 'lanes': p.get('lanes'),
                      'width': p.get('roadway_width'), 'surface': code}
        records.append({'passport_id': str(p.get('id')), 'road_id': str(p.get('road_id') or p.get('id')),
                        'road_name': p.get('road_part_name') or p.get('road_name'),
                        **values, **{f'raw_{key}': value for key, value in raw_fields.items()},
                        'signature': str((values['maxspeed'], values['lanes'], values['width'], code))})
        for part in lines(area):
            xy = np.asarray(part.coords)
            for a, b in zip(xy[:-1], xy[1:]):
                delta = b - a
                if np.linalg.norm(delta) > 1e-6:
                    segments.append(LineString([a, b]))
                    record_indices.append(record_index)
                    angles.append(np.degrees(np.arctan2(*delta)) % 180)
    return (pd.DataFrame(records), np.asarray(segments, dtype=object),
            np.asarray(record_indices, dtype=int), np.asarray(angles))


def match_samples(samples, passports, segments, record_indices, angles):
    """Apply the audit's thresholds and preserve the nearest conflicting alternative."""
    n = len(samples)
    matches = pd.DataFrame({'best_record': np.full(n, -1, dtype=int), 'distance_m': np.full(n, np.nan),
                            'angle_delta': np.full(n, np.nan), 'alt_record': np.full(n, -1, dtype=int),
                            'alt_distance_m': np.full(n, np.nan)})
    if not len(segments):
        return matches
    tree = shapely.STRtree(segments)
    points = shapely.points(samples.x.to_numpy(), samples.y.to_numpy())
    sample_angles = samples.angle.to_numpy()
    roads = passports.road_id.to_numpy()
    signatures = passports.signature.to_numpy()
    for start in range(0, n, 20000):
        q, s = tree.query(points[start:start + 20000], predicate='dwithin', distance=20)
        q += start
        delta = abs(angles[s] - sample_angles[q])
        delta = np.minimum(delta, 180 - delta)
        eligible = delta <= 25
        q, s, delta = q[eligible], s[eligible], delta[eligible]
        if not len(q):
            continue
        distance = shapely.distance(points[q], segments[s])
        sort = np.lexsort((s, distance, q))
        q, s, distance, delta = q[sort], s[sort], distance[sort], delta[sort]
        record = record_indices[s]
        lookup, first, inv = np.unique(q, return_index=True, return_inverse=True)
        nearest = record[first]
        matches.loc[lookup, ['best_record', 'distance_m', 'angle_delta']] = np.column_stack(
            (nearest, distance[first], delta[first]))
        best_for_candidate = nearest[inv]
        conflict = ((roads[record] != roads[best_for_candidate]) |
                    (signatures[record] != signatures[best_for_candidate]))
        near_conflict = conflict & (distance <= distance[first][inv] + 5)
        other_q = q[near_conflict]
        alt, first_alt = np.unique(other_q, return_index=True)
        matches.loc[alt, ['alt_record', 'alt_distance_m']] = np.column_stack(
            (record[near_conflict][first_alt], distance[near_conflict][first_alt]))
    matches[['best_record', 'alt_record']] = matches[['best_record', 'alt_record']].astype(int)
    return matches


def strict_observations(samples, passports):
    rows = []
    strict = samples[(samples.best_record >= 0) & (samples.alt_record < 0)]
    scopes = {'maxspeed': 'speed_limit', 'lanes': 'whole_carriageway',
              'width': 'whole_carriageway', 'surface': 'coarse'}
    for field, scope in scopes.items():
        selected = strict.loc[~strict.divided] if field in {'lanes', 'width'} else strict
        if selected.empty:
            continue
        passport = passports.loc[selected.best_record, [field, f'raw_{field}', 'passport_id']]
        passport.index = selected.index
        present = selected[['way_id', 'start_frac', 'end_frac']].join(passport).dropna(subset=[field])
        # Intervals may meet across OSM nodes. Never merge across an unknown sample or gap.
        previous = present.shift()
        new_run = (present.way_id.ne(previous.way_id) | present.passport_id.ne(previous.passport_id) |
                   present.start_frac.sub(previous.end_frac).abs().gt(1e-12))
        merged = present.groupby(new_run.cumsum(), sort=False, as_index=False).agg(
            {'way_id': 'first', 'start_frac': 'first', 'end_frac': 'last', field: 'first',
             f'raw_{field}': 'first', 'passport_id': 'first'})
        rows.append(merged.assign(direction='both', field=field, value=merged[field].map(as_text),
                                  raw_value=merged[f'raw_{field}'].map(as_text), source='skdf',
                                  method='distance_bearing_strict', quality='geometry_match',
                                  value_scope=scope, source_ref=merged.passport_id)[COLS])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=COLS)


def audit(samples, passports, osm, inverse):
    def lookup(which, column):
        if column not in passports:
            return pd.Series(index=samples.index, dtype=object)
        return samples[which].map(passports[column])

    result = samples[['way_id', 'highway', 'name', 'start_frac', 'end_frac',
                      'length_m', 'distance_m', 'angle_delta', 'alt_distance_m']].copy()
    result['lon'], result['lat'] = inverse.transform(samples.x.to_numpy(), samples.y.to_numpy())
    for column in ['passport_id', 'road_id', 'road_name']:
        result[column] = lookup('best_record', column)
        result[f'alt_{column}'] = lookup('alt_record', column)
    result['reason'] = np.select([samples.best_record.lt(0), samples.alt_record.ge(0)],
                                 ['no_candidate', 'ambiguous_match'], default='accepted_strict')
    strict = result.reason.eq('accepted_strict')
    withheld = []
    conflicts = []
    coverage = []
    for field in FIELDS:
        direct = osm.loc[osm.field.eq(field)]
        general = direct.loc[direct.direction.eq('both')].set_index('way_id').value
        forward = direct.loc[direct.direction.eq('forward')].set_index('way_id').value
        backward = direct.loc[direct.direction.eq('backward')].set_index('way_id').value
        both_val = samples.way_id.map(general)
        forward_val = samples.way_id.map(forward)
        backward_val = samples.way_id.map(backward)
        supplied = (both_val.notna() |
                    (forward_val.notna() & (samples.travel.eq('forward') | backward_val.notna())) |
                    (backward_val.notna() & samples.travel.eq('backward')))
        value = lookup('best_record', field)
        passport_present = strict & value.notna()
        usable = passport_present & ~samples.divided if field in {'lanes', 'width'} else passport_present
        coverage.append((field, supplied, usable))
        if field in {'lanes', 'width'}:
            withheld.append(passport_present & samples.divided)
        if field in {'maxspeed', 'lanes', 'width'}:
            # Directional speed limits may disagree with an undirected passport value.
            refs = [both_val] if field in {'lanes', 'width'} else [both_val, forward_val, backward_val]
            different = pd.Series(False, index=samples.index)
            for reference in refs:
                observed = pd.to_numeric(reference, errors='coerce')
                different |= observed.notna() & (observed.sub(value).abs() > 1e-6)
            specific = result.loc[usable & different].copy()
            specific['reason'] = 'open_conflict'
            specific['field'] = field
            specific['skdf_value'] = value.loc[specific.index].map(as_text)
            specific['osm_value'] = pd.concat(refs, axis=1).loc[specific.index].apply(
                lambda r: ' / '.join(r.dropna().astype(str)), axis=1)
            conflicts.append(specific)
    masks = pd.concat(withheld, axis=1).any(axis=1)
    carriageway = result.loc[masks].assign(reason='oneway_carriageway_unknown')
    result['refinement'] = None
    ambiguous = result.reason.eq('ambiguous_match')
    names = [s.fillna('').str.lower().str.replace('ё', 'е').str.replace(r'[^\w]+', ' ', regex=True).str.strip()
             for s in [result.name, result.road_name, result.alt_road_name]]
    current, best, alt = names
    result.loc[ambiguous & current.ne('') & current.eq(best) & current.ne(alt), 'refinement'] = (
        'name_agrees_with_nearest_unapproved')
    result.loc[ambiguous & current.ne('') & current.ne(best) & current.eq(alt), 'refinement'] = (
        'name_agrees_with_alternative_unapproved')
    sampled = pd.concat([result, carriageway, *conflicts], ignore_index=True)
    return sampled, coverage


def length_report(samples, availability):
    parts = []
    combined = []
    for field, osm, skdf in availability:
        combined.append((field, osm | skdf))
        report = pd.DataFrame({'highway': samples.highway, 'total_km': samples.length_m / 1000,
                               'osm_km': samples.length_m.where(osm, 0) / 1000,
                               'skdf_km': samples.length_m.where(skdf, 0) / 1000,
                               'combined_km': samples.length_m.where(osm | skdf, 0) / 1000})
        summary = report.groupby('highway', as_index=False).sum()
        parts.append(summary.assign(field=field))
    report = pd.concat(parts, ignore_index=True)
    report = pd.concat([report, report.groupby('field', as_index=False)[
        ['total_km', 'osm_km', 'skdf_km', 'combined_km']].sum().assign(highway='all')], ignore_index=True)
    all_four = pd.concat([flag for field, flag in combined if field != 'smoothness'], axis=1).all(axis=1)
    summary = {
        'sampled_total_km': samples.length_m.sum() / 1000,
        'complete_four_km': samples.length_m[all_four].sum() / 1000,
        'complete_four_nonservice_km': samples.length_m[all_four & samples.highway.ne('service')].sum() / 1000,
    }
    return report, summary


def build_city(city, extent, out):
    slug = city['slug']
    zone = int((city['lon'] + 180) // 6) + 1
    tr = Transformer.from_crs(4326, 32600 + zone, always_xy=True)
    inverse = Transformer.from_crs(32600 + zone, 4326, always_xy=True)

    def project(geometry):
        return shapely.transform(geometry, tr.transform, interleaved=False)

    if extent == 'bbox':
        south, west, north, east = city['extract_bbox']
        geometry = box(west, south, east, north)
    else:
        geometry = shape(city['geometry'])
    area = project(geometry)
    ways, samples, osm = collect_ways(slug, tr, area)
    samples = samples.join(ways, on=samples.way_index.astype(int))
    raw = json.loads((RAW / f'{slug}.geojson').read_text())
    passports, segments, record_indices, angles = collect_passports(raw, project, area.buffer(50))
    samples = samples.join(match_samples(samples, passports, segments, record_indices, angles))
    skdf = strict_observations(samples, passports)
    review, availability = audit(samples, passports, osm, inverse)
    report, counts = length_report(samples, availability)
    observations = pd.concat([osm, skdf], ignore_index=True)
    if not (observations.start_frac.ge(0) & observations.end_frac.le(1) &
            observations.end_frac.gt(observations.start_frac)).all():
        raise ValueError(f'Invalid interval in {slug}')
    if observations.duplicated(['way_id', 'start_frac', 'end_frac', 'direction', 'field', 'source']).any():
        raise ValueError(f'Duplicate interval in {slug}')
    folder = out / slug
    folder.mkdir(parents=True, exist_ok=True)
    observations.to_parquet(folder / 'observations.parquet', index=False)
    review.to_parquet(folder / 'review_candidates.parquet', index=False)
    report.to_csv(folder / 'coverage.csv', index=False)
    meta = {'city': slug, 'extent': extent, 'samples': len(samples), 'sampled_ways': samples.way_id.nunique(),
            'rows': observations.source.value_counts().to_dict(), 'review_rows': review.reason.value_counts().to_dict(),
            **counts, 'osm_extent': 'full ways from existing city bbox cache',
            'osm_snapshot': '2026-06-03', 'skdf_provenance': raw['provenance'],
            'coordinate_system': f'EPSG:{32600 + zone}',
            'referencing': 'consecutive OSM nodes WGS84 geodesic fractions on the complete original way in node order',
            'matching': {'sample_max_length_m': 20, 'max_distance_m': 20, 'max_angle_error_deg': 25,
                         'alternative_conflict_margin_m': 5, 'passport_geometry_buffer_m': 50},
            'notes': ['SKDF interval boundaries are sampled approximations.',
                      'Coarse SKDF road construction classes are distinct from OSM surface materials.',
                      'OSM fields take precedence over overlapping SKDF values. Conditional tags are not inferred.',
                      'SKDF lanes and width withheld from one-way or explicitly divided ways.',
                      'Name refinements are hypotheses and never enter observations.']}
    (folder / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    print(f'{slug}: {len(observations):,} observations, {len(samples):,} intervals at {extent} extent', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cities', nargs='+', default=['saratov'], help='City slugs or all')
    parser.add_argument('--extent', choices=['municipality', 'bbox'], default='municipality')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    cities = {row['slug']: row for row in json.loads((BASE / 'cities.json').read_text())}
    targets = list(cities) if args.cities == ['all'] else list(dict.fromkeys(args.cities))
    unknown = set(targets) - cities.keys()
    if unknown:
        parser.error(f'Unknown cities: {", ".join(sorted(unknown))}')
    for slug in targets:
        build_city(cities[slug], args.extent, args.output)


if __name__ == '__main__':
    main()
