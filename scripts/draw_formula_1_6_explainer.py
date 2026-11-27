"""Draw a Russian-language explainer for coursework formula 1.6."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path.home() / "Downloads"
STEM = "Формула_1_6_как_бустинг_считает_ДТП"

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(16, 9), facecolor="#f6f8fc")
ax.set(xlim=(0, 16), ylim=(0, 9))
ax.axis("off")


def box(x, y, w, h, face, edge="#d8e2ef", radius=0.18):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.03,rounding_size={radius}",
        linewidth=1.3, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)


def label(x, y, s, size=16, color="#17324d", weight="normal", ha="center"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            va="center", ha=ha, linespacing=1.25)


def arrow(x1, y1, x2, y2, color="#52708f", width=1.8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=17, linewidth=width, color=color))


label(0.7, 8.45, "Как формула 1.6 получает прогноз ДТП", size=28, weight="bold", ha="left")
label(0.7, 7.85, "Пример для одной дорожной ячейки за год", size=16, color="#536b82", ha="left")

box(0.55, 5.25, 2.9, 1.65, "#e8f2ff")
label(2, 6.39, "Признаки\nячейки x", size=15, weight="bold")
label(2, 5.57, "дороги · перекрёстки · уклон", size=10)
arrow(3.5, 6.08, 4.0, 6.08)

box(4.05, 5.25, 4.35, 1.65, "#eaf7f2", edge="#b5ddcb")
label(6.225, 6.40, "Старт + поправки\nдеревьев", size=15, weight="bold")
label(6.225, 5.63, "b₀ + η·f₁(x) + η·f₂(x) + η·f₃(x)", size=14)
arrow(8.46, 6.08, 8.98, 6.08)

box(9.05, 5.25, 2.25, 1.65, "#fff3dc", edge="#ead6ac")
label(10.175, 6.39, "Логарифм\nпрогноза", size=13, weight="bold")
label(10.175, 5.62, "a(x) = 0,70", size=16)
arrow(11.36, 6.08, 11.9, 6.08)

box(12.0, 5.25, 3.35, 1.65, "#e9e6ff", edge="#cec5ef")
label(13.675, 6.39, "Обратное\nпреобразование", size=13, weight="bold")
label(13.675, 5.62, "μ = exp(0,70) ≈ 2,01", size=15)

label(0.7, 4.63, "Числовой пример", size=20, weight="bold", ha="left")
box(0.55, 2.02, 14.8, 2.28, "#ffffff")

items = [
    (1.85, "Начало", "b₀ = 0", "#e8f2ff"),
    (5.05, "Дерево 1", "+0,40", "#eaf7f2"),
    (8.25, "Дерево 2", "−0,10", "#fdebec"),
    (11.45, "Дерево 3", "+0,40", "#eaf7f2"),
]
for idx, (cx, title, value, color) in enumerate(items):
    box(cx - 1.16, 2.55, 2.32, 1.23, color)
    label(cx, 3.39, title, size=15, weight="bold")
    label(cx, 2.93, value, size=19)
    if idx < len(items) - 1:
        arrow(cx + 1.2, 3.16, cx + 1.92, 3.16)

label(7.98, 2.31, "0 + 0,40 − 0,10 + 0,40 = 0,70", size=16, weight="bold")
label(0.7, 1.31, "η = 0,05 в нашей модели; на схеме показаны поправки деревьев уже с учётом шага.",
      size=16, ha="left")
label(0.7, 0.77, "Итог 2,01 — среднее ожидаемое число ДТП, а не обещание ровно двух происшествий.",
      size=16, color="#536b82", ha="left")

fig.savefig(OUT / f"{STEM}.png", dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.18)
fig.savefig(OUT / f"{STEM}.pdf", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.18)
plt.close(fig)
print(OUT / f"{STEM}.png")
print(OUT / f"{STEM}.pdf")
