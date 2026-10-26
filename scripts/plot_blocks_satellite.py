"""Межперекрёсточные отрезки улиц (этап 4) поверх спутникового снимка.

Отрезок = кусок уличной сети между двумя соседними перекрёстками
(после склейки дублей, без service). Цвет — длина отрезка.
Подложка — Esri World Imagery (тайлы кэшируются в data/raw/tiles/), только
для визуальной проверки; снимки не используются как данные модели.

    PYTHONPATH=src venv/bin/python scripts/plot_blocks_satellite.py
"""
from __future__ import annotations

import io
import json
import math
import time
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TILES = ROOT / "data/raw/tiles/esri_world_imagery"
OUT = ROOT / "docs/analysis/features/stage4_geometry/figures"
URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
Z = 17

WINDOWS = [
    ("а) центр: регулярная сетка кварталов", (51.5420, 46.0050, 51.5505, 46.0185)),
    ("б) частный сектор: густая сетка коротких отрезков", (51.5095, 45.9505, 51.5180, 45.9640)),
]

BINS = [0, 50, 100, 150, 250, 400, 1e9]
LABELS = ["< 50 м", "50–100", "100–150", "150–250", "250–400", "≥ 400 м"]
COLORS = ["#d7191c", "#fd8d3c", "#fecc5c", "#a1dab4", "#41b6c4", "#2c7fb8"]


def lonlat_to_px(lon, lat, z=Z):
    n = 2 ** z * 256
    x = (np.asarray(lon) + 180.0) / 360.0 * n
    lat_r = np.radians(np.asarray(lat))
    y = (1.0 - np.log(np.tan(lat_r) + 1.0 / np.cos(lat_r)) / math.pi) / 2.0 * n
    return x, y


def tile(z, x, y):
    p = TILES / str(z) / str(x) / f"{y}.jpg"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(URL.format(z=z, x=x, y=y),
                                     headers={"User-Agent": "road-accident-risk/research"})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    p.write_bytes(r.read())
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(1.5 * (attempt + 1))
        time.sleep(0.05)
    return Image.open(p).convert("RGB")


def mosaic(lat1, lon1, lat2, lon2):
    x0, y1 = lonlat_to_px(lon1, lat1)
    x1, y0 = lonlat_to_px(lon2, lat2)
    tx0, tx1 = int(x0 // 256), int(x1 // 256)
    ty0, ty1 = int(y0 // 256), int(y1 // 256)
    img = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            img.paste(tile(Z, tx, ty), ((tx - tx0) * 256, (ty - ty0) * 256))
    ox, oy = tx0 * 256, ty0 * 256
    crop = (int(x0 - ox), int(y0 - oy), int(x1 - ox), int(y1 - oy))
    return np.asarray(img.crop(crop)), (x0, y0)


def main() -> None:
    blocks = pd.read_parquet(ROOT / "parquet_output/osm_street_blocks.parquet")
    nodes = pd.read_parquet(ROOT / "parquet_output/osm_street_graph_consolidated_nodes.parquet")
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(BINS, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(18, 10.5))
    rows = []
    for ax, (title, (lat1, lon1, lat2, lon2)) in zip(axes, WINDOWS):
        img, (ox, oy) = mosaic(lat1, lon1, lat2, lon2)
        ax.imshow(img, extent=(0, img.shape[1], img.shape[0], 0))
        segs, lens = [], []
        for r in blocks.itertuples():
            c = np.array(json.loads(r.coords_json))
            if not ((c[:, 0] >= lat1 - 0.002) & (c[:, 0] <= lat2 + 0.002)
                    & (c[:, 1] >= lon1 - 0.003) & (c[:, 1] <= lon2 + 0.003)).any():
                continue
            px, py = lonlat_to_px(c[:, 1], c[:, 0])
            segs.append(np.column_stack([px - ox, py - oy]))
            lens.append(r.length_m)
            rows.append({"window": title, "block_id": r.block_id, "highway": r.highway,
                         "length_m": round(r.length_m, 1), "at_edge": r.at_edge})
        lc = LineCollection(segs, colors=cmap(norm(np.array(lens))), linewidths=3.2, alpha=0.95)
        ax.add_collection(lc)
        nd = nodes[(nodes.lat.between(lat1, lat2)) & (nodes.lng.between(lon1, lon2))
                   & (nodes.degree >= 3)]
        nx_, ny_ = lonlat_to_px(nd.lng.values, nd.lat.values)
        ax.scatter(nx_ - ox, ny_ - oy, s=34, facecolor="white", edgecolor="black",
                   linewidth=0.9, zorder=5)
        ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)
        ax.set_xticks([]); ax.set_yticks([])
        in_win = [l for s, l in zip(segs, lens)]
        ax.set_title(f"{title}\nотрезков: {len(in_win)}, медиана {np.median(in_win):.0f} м", fontsize=13, loc="left")
        # масштабная линейка 200 м
        m_per_px = 156543.03392 * math.cos(math.radians((lat1 + lat2) / 2)) / 2 ** Z
        L = 200 / m_per_px
        x0b, y0b = img.shape[1] * 0.05, img.shape[0] * 0.95
        ax.plot([x0b, x0b + L], [y0b, y0b], color="white", lw=4)
        ax.text(x0b + L / 2, y0b - 12, "200 м", color="white", ha="center", fontsize=11, weight="bold")

    handles = [plt.Line2D([0], [0], color=c, lw=4) for c in COLORS]
    handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                              markeredgecolor="black", markersize=8, lw=0))
    fig.legend(handles, LABELS + ["перекрёсток (после склейки дублей)"], loc="lower center",
               ncol=7, fontsize=11, frameon=False, title="длина отрезка улицы между соседними перекрёстками")
    fig.suptitle("Рис. 9. Межперекрёсточные отрезки уличной сети (без service) поверх спутникового снимка. "
                 "Подложка: Esri World Imagery", fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig9_blocks_satellite.png", dpi=300)
    pd.DataFrame(rows).to_csv(OUT / "fig9_blocks_satellite.csv", index=False)
    print("saved", OUT / "fig9_blocks_satellite.png")


if __name__ == "__main__":
    main()
