"""Finish dependent study stages once initial model run completes; no background sleeps."""
# This wrapper is intentionally sequential and checkpointed at each stage.
import sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for script in ['multicity_recent_sensitivity.py','multicity_eda.py','report_multicity.py']:
    result=subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT)
    if result.returncode:raise SystemExit(result.returncode)
