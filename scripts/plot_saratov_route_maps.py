"""Export four Saratov maps of baseline route counts on edges and H3 cells."""

import argparse
from pathlib import Path

import h3
import numpy as np
import pandas as pd

from plot_saratov_route_edge_counts import (
    ROOT, ROUTES, ROUTE_CMAP, decorate_map, draw_panel, load_counts, map_context,
)

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

OUTPUT = Path.home() / 'Downloads'
CELL_ALPHA = 0.60
cell_rgb = ROUTE_CMAP(np.linspace(0, 1, 256))[:, :3]
CELL_BAR_CMAP = LinearSegmentedColormap.from_list('route_count_over_white', 1 - CELL_ALPHA * (1 - cell_rgb))


def load_model_cells(projection):
    cells = pd.read_parquet(ROUTES / 'features.parquet',
                            columns=['h3', 'route_count'], filters=[('profile', '==', 'baseline')])
    cohort = pd.read_parquet(ROOT / 'reports/multicity_350k/features/saratov.parquet', columns=['h3'])
    if not cells.h3.is_unique or not cohort.h3.is_unique or set(cells.h3) != set(cohort.h3):
        raise ValueError('Route counts must match the Saratov model cells exactly')
    polygons = []
    for cell in cells.h3:
        lat, lon = np.asarray(h3.cell_to_boundary(cell)).T
        polygons.append(np.column_stack(projection.transform(lon, lat)))
    return polygons, cells.route_count.to_numpy()


def export_map(path, title, unit, subtitle, counts, bounds, draw, method_note, bar_cmap=ROUTE_CMAP):
    xmin, ymin, xmax, ymax = bounds
    aspect = (xmax - xmin) / (ymax - ymin)
    fig = plt.figure(figsize=(1.2 + 6.6 * aspect, 8.4))
    ax = fig.add_axes((0.045, 0.178, 0.91, 0.71))
    cap = int(np.ceil(np.quantile(counts[counts > 0], 0.99)))
    norm = Normalize(vmin=0, vmax=cap, clip=True)
    draw(ax, norm)
    fig.text(0.5, 0.964, title, fontsize=16, ha='center', va='top')
    fig.text(0.5, 0.915, subtitle, fontsize=10, ha='center')
    cax = fig.add_axes((0.16, 0.125, 0.68, 0.019))
    ticks = [0, *(t for t in range(10, cap, 10) if cap - t >= 5), cap]
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=bar_cmap), cax=cax,
                       orientation='horizontal', ticks=ticks, extend='max')
    bar.outline.set_linewidth(0.55)
    bar.ax.tick_params(labelsize=9, length=3, width=0.55)
    bar.set_label(f'Число маршрутов через {unit}', fontsize=10)
    fig.text(0.05, 0.060, f'Верх шкалы — 99-й процентиль ненулевых значений ({cap}). '
             f'Максимум — {int(counts.max())}.', fontsize=8.1)
    fig.text(0.05, 0.041, method_note, fontsize=8.1)
    fig.text(0.05, 0.022, 'Пунктир — муниципальная граница. Дорожная сеть © OpenStreetMap contributors, 03.06.2026.',
             fontsize=7.6)
    fig.savefig(path, dpi=280, facecolor='white')
    plt.close(fig)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells-only', action='store_true', help='Update the two H3 maps')
    args = parser.parse_args()
    geometry, edge_counts, found, planned = load_counts()
    segments, boundary, full, core, projection = map_context(geometry)
    polygons, cell_counts = load_model_cells(projection)
    subtitle = f'Стоимость по классу, без service. Найдено {found} маршрутов из {planned} пар.'
    plt.rcParams['font.family'] = 'DejaVu Serif'
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for scope, bounds, scale_km, detail in [
        ('город_и_окрестности', full, 10, False), ('городское_ядро', core, 2, True),
    ]:
        location = 'Саратов, городское ядро' if detail else 'Саратов и окрестности'

        def plot_edges(ax, norm):
            draw_panel(ax, segments, edge_counts, norm, boundary, bounds, scale_km, detail)

        def plot_cells(ax, norm):
            roads = LineCollection(segments, colors='#757c84', linewidths=0.24, alpha=0.72, zorder=1)
            ax.add_collection(roads, autolim=False)
            cells = PolyCollection(polygons, cmap=ROUTE_CMAP, norm=norm, edgecolors='none',
                                   alpha=CELL_ALPHA, zorder=2)
            cells.set_array(cell_counts)
            ax.add_collection(cells, autolim=False)
            decorate_map(ax, boundary, bounds, scale_km, detail)

        if not args.cells_only:
            export_map(OUTPUT / f'Саратов_{scope}_граф.png', f'{location}. Маршруты по рёбрам',
                       'участок', subtitle, edge_counts, bounds, plot_edges,
                       'Каждый маршрут учитывается на участке один раз. Суммированы оба направления.')
        export_map(OUTPUT / f'Саратов_{scope}_ячейки_H3.png', f'{location}. Маршруты по ячейкам H3',
                   'ячейку', subtitle, cell_counts, bounds, plot_cells,
                   'Ячейки модели H3, разрешение 9. Каждый маршрут учитывается в ячейке один раз.',
                   bar_cmap=CELL_BAR_CMAP)


if __name__ == '__main__':
    main()
