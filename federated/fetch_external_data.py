#!/usr/bin/env python3
"""Download the three public credit datasets used in external_datasets.py into ../data/external/.
Sources: UCI Machine Learning Repository via the `ucimlrepo` package (Taiwan credit card default, id 350;
Taiwanese bankruptcy, id 572; Statlog German credit, id 144 for verification) and the labelled copy of the German
credit data shipped with the `scorecardpy` package (the same 1,000 records with readable category labels).
Needs: pip install ucimlrepo scorecardpy   (scorecardpy is only read for its data file)
"""
import os, pathlib, numpy as np, pandas as pd
from ucimlrepo import fetch_ucirepo
out = pathlib.Path(__file__).resolve().parent.parent / "data" / "external"; out.mkdir(parents=True, exist_ok=True)
for uid, name in [(350, "taiwan_default"), (572, "taiwan_bankruptcy")]:
    d = fetch_ucirepo(id=uid); pd.concat([d.data.features, d.data.targets], axis=1).to_csv(out / f"{name}.csv", index=False)
import importlib.util
pkg = os.path.dirname(importlib.util.find_spec("scorecardpy").origin)   # located, not imported
lab = pd.read_csv(os.path.join(pkg, "data", "germancredit.csv"))
raw = fetch_ucirepo(id=144); r = pd.concat([raw.data.features, raw.data.targets], axis=1)
# verification: the labelled copy must match the UCI original on the numeric columns and the class counts
for a, b in [("duration_in_month", "Attribute2"), ("credit_amount", "Attribute5"), ("age_in_years", "Attribute13")]:
    assert (lab[a].values == r[b].values).all(), a
assert (lab["creditability"].eq("bad").values == (r["class"].values == 2)).all()
lab.to_csv(out / "german_credit.csv", index=False)
print("written:", sorted(p.name for p in out.iterdir()))
