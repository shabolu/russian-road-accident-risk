"""Plot fully missing optional route fields by Saratov non-service road class."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/route_proxy_coverage/ways/saratov.parquet"
DOWNLOADS = Path.home() / "Downloads"
ABSOLUTE = DOWNLOADS / "Саратов_без_service_нет_маршрутных_полей_по_классам.png"
SHARES = DOWNLOADS / "Саратов_без_service_доля_без_маршрутных_полей_по_классам.png"
FIELDS = [
    "combined_speed_length_share",
    "combined_surface_coarse_length_share",
    "combined_lanes_length_share",
    "combined_width_length_share",
    "osm_smoothness_length_share",
]
CLASS_NAMES = {
    "residential": "residential · жилые",
    "unclassified": "unclassified · местные",
    "tertiary": "tertiary · районные",
    "living_street": "living_street · жилые зоны",
    "secondary": "secondary · второстепенные",
    "tertiary_link": "tertiary_link · съезды",
    "secondary_link": "secondary_link · съезды",
    "primary_link": "primary_link · съезды",
    "trunk": "trunk · скоростные",
    "primary": "primary · основные",
    "trunk_link": "trunk_link · съезды",
}


def make_axes(title, subtitle):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.6), sharey=True)
    fig.patch.set_facecolor("#f6f8fb")
    fig.suptitle(title, x=0.04, y=0.985, ha="left", fontsize=20,
                 fontweight="bold", color="#172b46")
    fig.text(0.04, 0.935, subtitle, fontsize=11, color="#52657c")
    for ax in axes:
        ax.set_facecolor("white")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#dce3eb", alpha=0.65)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.21, right=0.96, top=0.87, bottom=0.17, wspace=0.16)
    return fig, axes


def plot_bars(ax, data, color, title, formatter, limit):
    y = np.arange(len(data))
    bars = ax.barh(y, data, color=color, height=0.72)
    ax.set_yticks(y)
    ax.set_xlim(0, limit)
    ax.set_title(title, loc="left", fontsize=14, color="#20364f", pad=14)
    for bar, value in zip(bars, data):
        ax.text(value + limit * 0.014, bar.get_y() + bar.get_height() / 2,
                formatter(value), va="center", fontsize=10.5, color="#263d55")


def main():
    ways = pd.read_parquet(SOURCE, columns=["osm_id", "highway", "length_m", *FIELDS])
    assert ways.osm_id.is_unique
    ways = ways.loc[ways.highway.ne("service")].copy()
    ways["no_extra_fields"] = ~ways[FIELDS].ge(1 - 1e-6).any(axis=1)
    ways["missing_length_m"] = ways.length_m.where(ways.no_extra_fields, 0)
    table = ways.groupby("highway").agg(
        ways=("osm_id", "size"),
        total_km=("length_m", lambda s: s.sum() / 1000),
        missing_ways=("no_extra_fields", "sum"),
        missing_km=("missing_length_m", lambda s: s.sum() / 1000),
    ).sort_values(["missing_km", "missing_ways"], ascending=False)
    assert int(table.missing_ways.sum()) == 3639
    labels = [CLASS_NAMES.get(name, name) for name in table.index]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

    fig, axes = make_axes(
        "Дороги Саратова без дополнительных маршрутных полей",
        "Без service · классы OSM · 3 639 объектов, 1 098 км · порядок классов по длине без полей",
    )
    plot_bars(axes[0], table.missing_ways, "#3a70ad", "Число объектов OSM",
              lambda v: f"{int(v):,}".replace(",", " "), table.missing_ways.max() * 1.19)
    plot_bars(axes[1], table.missing_km, "#148979", "Суммарная длина, км",
              lambda v: f"{v:,.1f}".replace(",", " "), table.missing_km.max() * 1.19)
    axes[0].invert_yaxis()
    axes[0].set_yticklabels(labels)
    fig.text(0.04, 0.075,
             "Поле считается известным только при наличии на всей длине OSM-way. "
             "Класс highway не входит в пять дополнительных полей.",
             fontsize=10, color="#52657c")
    fig.text(0.04, 0.045,
             "Источник: локальный OSM-срез 03.06.2026 и СКДФ; геометрия обрезана муниципальной границей.",
             fontsize=9, color="#74849a")
    fig.savefig(ABSOLUTE, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    fig, axes = make_axes(
        "Где внутри класса не хватает маршрутных полей",
        "Без service · доля объектов с нулём из пяти дополнительных полей",
    )
    count_share = table.missing_ways / table.ways * 100
    length_share = table.missing_km / table.total_km * 100
    plot_bars(axes[0], count_share, "#3a70ad", "Доля OSM-way класса", lambda v: f"{v:.1f}%", 110)
    plot_bars(axes[1], length_share, "#148979", "Доля длины класса", lambda v: f"{v:.1f}%", 110)
    axes[0].invert_yaxis()
    axes[0].set_yticklabels(labels)
    fig.text(0.04, 0.075,
             "Например, у residential без полей 2 125 из 4 456 объектов и 573 из 1 145 км. "
             "Для класса дороги базовую стоимость можно задать и без этих полей.",
             fontsize=10, color="#52657c")
    fig.text(0.04, 0.045,
             "Источник: локальный OSM-срез 03.06.2026 и СКДФ; поле считается пропуском, если известно лишь на части way.",
             fontsize=9, color="#74849a")
    fig.savefig(SHARES, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(table.round(3).to_string())
    print(ABSOLUTE)
    print(SHARES)


if __name__ == "__main__":
    main()
