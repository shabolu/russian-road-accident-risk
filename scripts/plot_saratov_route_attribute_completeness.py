"""Compare route-attribute completeness for Saratov with and without service."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/route_proxy_coverage/ways/saratov.parquet"
AUDIT = ROOT / "docs/analysis/route_proxy/coverage/completeness_histogram_by_city.csv"
OUTPUT = Path.home() / "Downloads/Заполненность_маршрутных_признаков_Саратов_с_и_без_service.png"

FIELDS = [
    "combined_speed_length_share",
    "combined_surface_coarse_length_share",
    "combined_lanes_length_share",
    "combined_width_length_share",
    "osm_smoothness_length_share",
]


def main():
    ways = pd.read_parquet(SOURCE, columns=["osm_id", "highway", *FIELDS])
    assert ways.osm_id.is_unique
    known4 = sum(ways[col].ge(1 - 1e-6).astype(int) for col in FIELDS[:4])
    audit = pd.read_csv(AUDIT)
    expected = (audit[audit["Город"] == "Саратов"]
                .set_index("Признаков известно на всей длине")["Число объектов OSM"]
                .reindex(range(5), fill_value=0))
    assert known4.value_counts().reindex(range(5), fill_value=0).equals(expected)

    ways["known5"] = known4 + ways[FIELDS[4]].ge(1 - 1e-6).astype(int)
    all_counts = ways.known5.value_counts().reindex(range(6), fill_value=0)
    nonservice = ways[ways.highway != "service"]
    nonservice_counts = nonservice.known5.value_counts().reindex(range(6), fill_value=0)
    service_zero = int(((ways.highway == "service") & (ways.known5 == 0)).sum())
    assert int(all_counts.sum()) == len(ways)
    assert int(nonservice_counts.sum()) == len(nonservice)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(2, 1, figsize=(13, 9.3), sharex=True,
                             gridspec_kw={"hspace": 0.33})
    fig.patch.set_facecolor("#f6f8fb")
    for ax, counts, title, color, total in [
        (axes[0], all_counts, "Все автомобильные классы", "#4965a8", len(ways)),
        (axes[1], nonservice_counts, "Без класса service", "#148979", len(nonservice)),
    ]:
        ax.set_facecolor("#ffffff")
        bars = ax.bar(range(6), counts.to_numpy(), width=0.66, color=color,
                      edgecolor="white", linewidth=1.5)
        ax.set_ylim(0, counts.max() * 1.18)
        ax.set_yticks([])
        ax.set_title(f"{title} · {total:,} объектов OSM".replace(",", " "),
                     loc="left", fontsize=16, fontweight="bold", color="#20364f", pad=12)
        ax.grid(axis="y", alpha=0.15)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#cbd5e1")
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + counts.max() * 0.023,
                    f"{count:,}\n({count / total:.1%})".replace(",", " "),
                    ha="center", va="bottom", fontsize=10.5,
                    color="#20364f", fontweight="bold")
        ax.text(0.98, 0.84, "Число OSM-ways; шкала своя для каждой панели",
                transform=ax.transAxes, ha="right", color="#65758a", fontsize=9)

    axes[1].set_xticks(range(6), [str(x) for x in range(6)])
    axes[1].set_xlabel("Сколько из пяти дополнительных полей известно на всей длине объекта",
                       fontsize=12, color="#20364f", labelpad=9)
    fig.suptitle("Заполненность данных для маршрутной стоимости: Саратов",
                 x=0.08, ha="left", fontsize=21, fontweight="bold", color="#172b46", y=0.985)
    fig.text(0.08, 0.925,
             "Поля: ограничение скорости · покрытие/дорожная одежда · полосы · ширина · smoothness",
             fontsize=11, color="#52657c")
    zero_share = service_zero / int(all_counts.loc[0])
    footer = (
        f"Среди {int(all_counts.loc[0]):,} объектов с 0 полей {service_zero:,} ({zero_share:.1%}) — service. "
        "Базовую стоимость по классу можно задать и при 0 полей."
    ).replace(",", " ")
    fig.text(0.08, 0.045, footer, fontsize=10, color="#3d526a")
    fig.text(0.08, 0.02,
             "Источник: локальный OSM-срез 03.06.2026 и СКДФ; объект = OSM-way внутри муниципальной границы. "
             "Частично известное поле здесь считается пропуском.",
             fontsize=9, color="#74849a")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.87, bottom=0.12)
    fig.savefig(OUTPUT, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {OUTPUT}")
    print(f"all: {all_counts.to_dict()}")
    print(f"without service: {nonservice_counts.to_dict()}")
    print(f"service with zero: {service_zero}/{int(all_counts.loc[0])}")


if __name__ == "__main__":
    main()
