"""Replot EDA from cached tables without refitting models."""
from pathlib import Path
p=Path(__file__).with_name('eda_city_transfer.py');code=p.read_text();exec(code.split('counts={}; overview=')[0])
for c,f in frames.items():frames[c]=f.merge(preds[preds.city==c][['h3','pred','observed']],on='h3')
annual=pd.read_csv(OUT/'annual.csv');calibration=pd.read_csv(OUT/'calibration.csv');desc=pd.read_csv(OUT/'feature_descriptive.csv');shaprows=pd.read_csv(OUT/'shap.csv');overview=pd.read_csv(OUT/'overview.csv');mat=pd.read_csv(OUT/'transfer_matrix_diagnostic.csv')
exec(code[code.index('RU=json.loads'):])
