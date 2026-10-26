#!/bin/bash
# Этап 7, шаг 2A: прогоны годовых моделей (7.2, 7.3) — 5 сидов, folds + test, тег stage7A_models.
# Шаги можно запускать по отдельности: scripts/stage7A_run_models.sh 3 4 (по умолчанию — все 1..6).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
R="venv/bin/python scripts/run_v2_eval_year.py --split both --tag stage7A_models"
V="--variants road_geo_v2_skdf"
STEPS="${@:-1 2 3 4 5 6}"
for s in $STEPS; do
  echo "=== шаг $s: $(date +%H:%M:%S)"
  case $s in
    1) $R $V --mode binary --class-weights Balanced --monthly mean,max ;;
    2) $R $V --mode binary --class-weights None ;;
    3) $R $V --mode count --loss Poisson ;;
    4) $R $V --mode count --loss 'Tweedie:variance_power=1.3' ;;
    5) $R $V --mode count --loss 'Tweedie:variance_power=1.6' ;;
    6) $R --baselines accidents_prev_year,accidents_prev_3y,major_road_length_m ;;
  esac
  echo "=== шаг $s готов: $(date +%H:%M:%S)"
done
