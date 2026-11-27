"""Draw counts of simulated routes through the Saratov routing graph."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely.geometry import shape

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/route_proxy_experiments'
GRAPH = REPORT / 'graph_no_service' / 'saratov'
ROUTES = REPORT / 'h3_no_service' / 'saratov'
OUTPUT = Path.home() / 'Downloads/Саратов_кратчайшие_маршруты_по_рёбрам.png'
ROUTE_CMAP = LinearSegmentedColormap.from_list('route_count', [
    (0.00, '#e0e2e5'), (0.07, '#8cb9db'), (0.28, '#426daf'),
    (0.52, '#643a91'), (0.76, '#9c205f'), (1.00, '#a20929'),
])


def load_counts():
    edges = pd.read_parquet(GRAPH / 'edges.parquet', columns=[
        'edge_id', 'way_id', 'start_frac', 'end_frac', 'geometry', 'highway',
    ]).sort_values('edge_id', ignore_index=True)
    if edges.highway.eq('service').any():
        raise ValueError('The map requires a routing graph without highway=service')
    if not np.array_equal(edges.edge_id.to_numpy(), np.arange(len(edges))):
        raise ValueError('Route edge IDs must index the graph edge table')
    edges['lo'] = np.minimum(edges.start_frac, edges.end_frac)
    edges['hi'] = np.maximum(edges.start_frac, edges.end_frac)
    keys = ['way_id', 'lo', 'hi']
    # Reciprocal directions share a line. Give it one vote per route in either direction.
    line_ids = edges.groupby(keys, sort=False).ngroup().to_numpy(dtype=int)
    geometry = shapely.from_wkb(edges.drop_duplicates(keys).geometry.to_numpy())
    if not np.all(shapely.get_num_coordinates(geometry) == 2):
        raise ValueError('Expected routing edges between consecutive nodes')
    routes = pd.read_parquet(ROUTES / 'routes.parquet', filters=[('profile', '==', 'baseline')])
    if not routes.pair_id.is_unique:
        raise ValueError('Expected one baseline row per pair of points')
    counts = np.zeros(len(geometry), dtype=int)
    for ids in routes.loc[routes.found, 'edge_ids']:
        counts[np.unique(line_ids[ids])] += 1
    return shapely.get_coordinates(geometry).reshape(-1, 2, 2), counts, int(routes.found.sum()), len(routes)


def draw_panel(ax, segments, counts, norm, boundary, bounds, scale_km, detail=False):
    xmin, ymin, xmax, ymax = bounds
    low, high = segments.min(axis=1), segments.max(axis=1)
    visible = ((high[:, 0] >= xmin) & (low[:, 0] <= xmax)
               & (high[:, 1] >= ymin) & (low[:, 1] <= ymax))
    ax.add_collection(LineCollection(segments[visible & (counts == 0)],
                                     colors=[ROUTE_CMAP(0.0)], linewidths=0.22))
    used = np.flatnonzero(visible & (counts > 0))
    used = used[np.argsort(counts[used], kind='stable')]
    paths = LineCollection(segments[used], cmap=ROUTE_CMAP, norm=norm,
                           linewidths=1.05, capstyle='round', zorder=3)
    paths.set_array(counts[used])
    ax.add_collection(paths)
    decorate_map(ax, boundary, bounds, scale_km, detail)


def decorate_map(ax, boundary, bounds, scale_km, detail=False):
    xmin, ymin, xmax, ymax = bounds
    for part in shapely.get_parts(boundary.boundary):
        xy = shapely.get_coordinates(part)
        ax.plot(*xy.T, color='#686e75', linewidth=0.6, linestyle=(0, (4, 3)))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box', anchor='N')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#767a7e')
        spine.set_linewidth(0.55)
    x = xmin + 0.055 * (xmax - xmin)
    y = ymin + 0.05 * (ymax - ymin)
    ax.plot([x, x + scale_km * 1000], [y, y], color='#27384b', linewidth=2, solid_capstyle='butt')
    ax.annotate(f'{scale_km} км', (x, y), xytext=(0, 4), textcoords='offset points',
                fontsize=9, color='#27384b', bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': 1})
    if not detail:
        ax.annotate('С', (0.95, 0.95), xytext=(0.95, 0.88), xycoords='axes fraction',
                    textcoords='axes fraction', ha='center', color='#27384b', fontsize=10,
                    arrowprops={'arrowstyle': '-|>', 'color': '#27384b', 'lw': 1.1})


def map_context(segments):
    projection = Transformer.from_crs('EPSG:4326', 'EPSG:32638', always_xy=True)
    x, y = projection.transform(segments[..., 0].ravel(), segments[..., 1].ravel())
    segments = np.column_stack((x, y)).reshape(-1, 2, 2)
    boundaries = json.loads((ROOT / 'reports/multicity_350k/boundaries.geojson').read_text())
    boundary = next(shape(f['geometry']) for f in boundaries['features'] if f['properties']['slug'] == 'saratov')
    boundary = shapely.transform(boundary, lambda coords: np.column_stack(projection.transform(*coords.T)))
    full = (*tuple(segments.min(axis=(0, 1)) - 1600), *tuple(segments.max(axis=(0, 1)) + 1600))
    cx, cy = projection.transform(46.003, 51.53)
    center = (cx - 7800, cy - 5850, cx + 7800, cy + 5850)
    return segments, boundary, full, center, projection


def main():
    segments, counts, found, planned = load_counts()
    segments, boundary, full, center, _ = map_context(segments)
    max_count = int(counts.max())
    cap = int(np.ceil(np.quantile(counts[counts > 0], 0.99)))
    norm = Normalize(vmin=0, vmax=cap, clip=True)
    ratios = [(xmax - xmin) / (ymax - ymin) for xmin, ymin, xmax, ymax in [full, center]]

    plt.rcParams['font.family'] = 'DejaVu Serif'
    fig, axes = plt.subplots(1, 2, figsize=(13, 8.4), gridspec_kw={'width_ratios': ratios})
    fig.subplots_adjust(left=0.04, right=0.96, top=0.842, bottom=0.202, wspace=0.06)
    fig.text(0.5, 0.956, 'Число кратчайших маршрутов на рёбрах дорожного графа Саратова',
             fontsize=17, ha='center', va='top')
    fig.text(0.5, 0.910, f'Стоимость по классу, без service. Найдено {found} маршрутов для {planned} пар.',
             fontsize=10.5, ha='center')
    for ax, bounds, detail, title, scale in zip(
            axes, (full, center), (False, True), ('(а) Город и окрестности', '(б) Городское ядро'), (10, 2)):
        draw_panel(ax, segments, counts, norm, boundary, bounds, scale, detail)
        ax.set_title(title, loc='left', pad=7, fontsize=11)
    axes[0].add_patch(Rectangle(center[:2], center[2] - center[0], center[3] - center[1],
                               fill=False, linewidth=1.0, edgecolor='#344e68'))
    cax = fig.add_axes((0.25, 0.132, 0.50, 0.020))
    ticks = [0, *(t for t in range(10, cap, 10) if cap - t >= 5), cap]
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=ROUTE_CMAP), cax=cax,
                           orientation='horizontal', ticks=ticks, extend='max')
    colorbar.outline.set_linewidth(0.55)
    colorbar.ax.tick_params(labelsize=9, width=0.55, length=3)
    colorbar.set_label('Число маршрутов через участок', fontsize=11)
    fig.text(0.04, 0.067, f'Верх шкалы — 99-й процентиль по участкам с проходами ({cap}). Фактический максимум — {max_count}.',
             fontsize=8.5)
    fig.text(0.04, 0.048, 'Каждый маршрут учитывается на участке один раз. Суммированы проходы в обоих направлениях.',
             fontsize=8.5)
    fig.text(0.04, 0.028, 'Пунктиром показана муниципальная граница. Дорожная сеть © OpenStreetMap contributors, срез 03.06.2026.',
             fontsize=8.5, color='#44484d')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=260)
    plt.close(fig)
    print(f'{OUTPUT}\n{len(counts):,} segments, {(counts > 0).sum():,} used, p99 cap={cap}, '
          f'{(counts > cap).sum():,} above cap, max={max_count}')


if __name__ == '__main__':
    main()
