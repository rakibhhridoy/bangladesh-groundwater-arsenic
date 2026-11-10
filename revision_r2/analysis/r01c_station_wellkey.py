"""Well key used by r02-r07: one well per BWDB station identifier.
Five identifiers are truncated ('...-'); those are separated by latitude."""
import os, numpy as np, pandas as pd
from pathlib import Path
R=os.environ.get("PAPER1_ROOT", str(Path(__file__).resolve().parents[2]))
A=f"{R}/revision_r2/output"; os.makedirs(A, exist_ok=True)
d=pd.read_csv(f"{R}/data/integrated/bangladesh_groundwater_arsenic_1807samples_CLEAN.csv")
sid=d['Sample ID'].astype(str).str.strip()
key=np.where(sid.str.endswith('-'), sid+'@'+d.Latitude.round(4).astype(str), sid)
pd.DataFrame({'row':np.arange(len(d)),'Sample ID':d['Sample ID'],'Season':d.Season,'well':key}).to_csv(f"{A}/r01c_station_wellkey.csv",index=False)
print('wells',len(set(key)))
