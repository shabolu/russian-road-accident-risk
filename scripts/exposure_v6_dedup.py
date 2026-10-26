"""Этап 6a: Spearman новых признаков (кандидатов) внутри блоков и с набором A = road_geo_v2_skdf (32).

По ячейкам панели (road_length_total_m > 0, одно значение на ячейку). Выход:
    reports/road_geo_v2_exposure_a_spearman_pairs.csv  — все пары |ρ| ≥ 0.5 (новый × новый, новый × A)
    docs/analysis/features/stage6_exposure/blocks_a/tables/spearman_new_vs_a.csv — матрица новых × (новые + A)

    PYTHONPATH=src venv/bin/python scripts/exposure_v6_dedup.py
"""
from __future__ import annotations

import pandas as pd

from road_accident_risk import exposure_v6 as ev
from road_accident_risk.catboost_baseline import attach_exposure_v6_features
from road_accident_risk.paths import PROJECT_ROOT
from road_accident_risk.v2_eval import load_panel, variant_features

TAB = PROJECT_ROOT / "docs/analysis/features/stage6_exposure/blocks_a/tables"
BLOCK = {**{f: "E1" for f in [*ev.E1_CENTER_CANDIDATES, *ev.E1_OTHER]},
         **{f: "E2" for m in ev.E2_MODES for f in ev.e2_names(m)},
         **{f: "E3" for f in ev.E3_FEATURES}, **{f: "E4" for f in ev.E4_FEATURES},
         **{f: "diag" for f in ev.DIAGNOSTIC_FEATURES}}


def main() -> None:
    panel = load_panel(attach=[attach_exposure_v6_features])
    a = variant_features("road_geo_v2_skdf", list(panel.columns))
    cells = panel.drop_duplicates("h3").set_index("h3")
    new = ev.EXPOSURE_V6_FEATURES
    rho = cells[[*new, *a]].corr(method="spearman")
    rho.loc[new].to_csv(TAB / "spearman_new_vs_a.csv")
    rows = []
    for i, f in enumerate(new):
        for g in [*new[i + 1:], *a]:
            r = rho.loc[f, g]
            if abs(r) >= 0.5:
                rows.append({"feature": f, "block": BLOCK[f], "other": g, "other_block": BLOCK.get(g, "A"), "rho": r})
    pairs = pd.DataFrame(rows).sort_values("rho", key=abs, ascending=False)
    pairs.to_csv(PROJECT_ROOT / "reports/road_geo_v2_exposure_a_spearman_pairs.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(pairs[pairs.rho.abs() >= 0.7].round(3).to_string(index=False))
        print("\nмакс |ρ| с A по каждому новому:")
        m = rho.loc[new, a].abs()
        print(pd.DataFrame({"max_abs_rho_A": m.max(axis=1), "argmax": m.idxmax(axis=1),
                            "nan_share": cells[new].isna().mean()}).round(3).to_string())


if __name__ == "__main__":
    main()
