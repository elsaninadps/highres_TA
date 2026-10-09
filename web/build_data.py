"""Export real numbers from the 1993-2022 quantile ensemble run to web/data.json.

Run from the repository root:  uv run python web/build_data.py
Only reads existing outputs (no training). Re-run after retraining to refresh the page.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import highres_ta as ta  # noqa: E402
import train_quantile_ensemble as tq  # noqa: E402

RUN = ROOT / "models" / "quantile_ensemble_split_hpo_1993-2022"
QS = [0.1, 0.5, 0.9]
N_SAMPLE = 700
rng = np.random.default_rng(0)

config = yaml.safe_load((ROOT / "scripts" / "quantile_ensemble_config.yaml").read_text())
dc = config["data"]
# the finished run required ssh_adt (hence 1993+); reproduce that
non_null = dc["non_null_columns"] + ["ssh_adt"]

# ---- preprocessing funnel -------------------------------------------------
raw = ta.load_data(str(ROOT / dc["parquet_glob"]))
raw = raw[raw.year.between(1993, 2022)] if "year" in raw else raw
raw_feat = [n for n in dc["feature_names"] if n not in dc["engineered_feature_names"]]
cols = list(dict.fromkeys(dc["coordinate_columns"] + raw_feat + ["talk"] + dc["quality_columns"]))
df = raw[cols].set_index(dc["coordinate_columns"], drop=False)
funnel = [("Collocated bottle samples (1993-2021)", len(df), df.expocode.nunique())]
df = ta.drop_extreme_salinities(df, min=dc["minimum_salinity"], max=dc["maximum_salinity"])
funnel.append(("Salinity within 20-40", len(df), df.expocode.nunique()))
df = ta.add_talk_adjustment(df, fname=str(ROOT / dc["talk_adjustment_csv"]))
df = ta.drop_bad_quality_talk(df)
funnel.append(("Good TA flag and |adjustment| <= 6 umol/kg", len(df), df.expocode.nunique()))
df = df.dropna(subset=non_null)
funnel.append(("Complete TA, S, T, nitrate and SSH", len(df), df.expocode.nunique()))
df = df.drop_duplicates(subset=dc["coordinate_columns"], keep="first")
funnel.append(("Duplicate coordinates removed", len(df), df.expocode.nunique()))
meta = df.reset_index(drop=True)[["expocode", "time", "lat", "lon", "depth", "talk"]]
print("final rows", len(df), "(expected 27210)")

# ---- blocks / splits ----------------------------------------------------------
splits_df = pd.read_parquet(RUN / "splits" / "split_1.parquet")
assert len(splits_df) == len(meta), (len(splits_df), len(meta))
data = df.select_dtypes(include=[np.number])  # placeholder to mirror prepare_data order
blocks_info = []
test_ids = set(splits_df.query("split=='test'").row_id)
block_of = {}
cv = config["cross_validation"]
# recover block membership: block 0 = test; block k = validation set of split k
for k in range(1, cv["n_parts"]):
    s = pd.read_parquet(RUN / "splits" / f"split_{k}.parquet")
    for r in s.query("split=='validation'").row_id:
        block_of[r] = k
for r in test_ids:
    block_of[r] = 0
meta["block"] = meta.index.map(block_of)
assert meta.block.notna().all()
for b in range(cv["n_parts"]):
    m = meta[meta.block == b]
    blocks_info.append({"block": b, "observations": int(len(m)), "cruises": int(m.expocode.nunique())})

# ---- members ------------------------------------------------------------------
metrics = json.loads((RUN / "ensemble_test_metrics.json").read_text())
names = metrics["members"]
members = []
preds = []
for name in names:
    j = json.loads((RUN / "models" / f"quantile_{name}.json").read_text())
    sp, rep = int(name.split("_")[1]), int(name.split("_")[3])
    t = j["scores"]["test"]["all"]
    v = j["scores"]["validation"]["all"]
    members.append({
        "name": name, "split": sp, "replicate": rep, "seed": 100 * sp + rep,
        "params": {k: round(float(x), 4) for k, x in j["best_params"].items()},
        "iterations": j["refit_iterations"], "inner_cv_loss": round(j["best_inner_cv_loss"], 3),
        "n_train": j["observation_counts"]["train"], "n_validation": j["observation_counts"]["validation"],
        "cruises_train": j["cruise_counts"]["train"],
        "test": {"rmse": round(t["median_rmse"], 2), "mae": round(t["median_mae"], 2),
                 "crps": round(t["quantile_crps"], 3), "coverage80": round(t["interval_80pct_coverage"], 3)},
        "validation": {"rmse": round(v["median_rmse"], 2)},
    })
    p = pd.read_parquet(RUN / "predictions" / f"{name}_test.parquet").sort_values("row_id")
    preds.append(p[[f"q_{q}" for q in QS]].to_numpy())
P = np.stack(preds)  # (members, obs, quantiles)
test = pd.read_parquet(RUN / "predictions" / "ensemble_test.parquet").sort_values("row_id")
rows = test.row_id.to_numpy()
y = test.observed.to_numpy()
ens = P.mean(0)  # mean aggregation like the saved ensemble
assert np.allclose(ens, test[[f"q_{q}" for q in QS]].to_numpy(), atol=1e-3)
sp_idx = np.array([m["split"] for m in members])

# ---- uncertainty diagnostics on the full test set -------------------------------------
med = P[:, :, 1]
ens_med = med.mean(0)
err = y - ens_med
member_std = med.std(0)                                   # spread of member medians
iq = (ens[:, 2] - ens[:, 0]) / 2.563                      # scaled Q90-Q10
total = np.sqrt(member_std**2 + iq**2)
def binned(u, nb=8):
    edges = np.quantile(u, np.linspace(0, 1, nb + 1)); out = []
    for i in range(nb):
        m = (u >= edges[i]) & (u <= edges[i + 1]) if i == nb - 1 else (u >= edges[i]) & (u < edges[i + 1])
        out.append({"u": round(float(np.sqrt(np.mean(u[m] ** 2))), 2), "rmse": round(float(np.sqrt(np.mean(err[m] ** 2))), 2), "n": int(m.sum())})
    return out
spread = {"member_std": binned(member_std), "scaled_iqr": binned(iq), "combined": binned(total),
          "corr_abs_err": {"member_std": round(float(np.corrcoef(member_std, np.abs(err))[0, 1]), 3),
                           "scaled_iqr": round(float(np.corrcoef(iq, np.abs(err))[0, 1]), 3)}}

# error / coverage as ensemble grows (random subsets, 30 draws)
grow = []
for n in [1, 2, 3, 5, 7, 10, 14, 21, 28, 42]:
    r, c, w = [], [], []
    for _ in range(30):
        sel = rng.choice(len(members), n, replace=False)
        e = P[sel].mean(0)
        r.append(np.sqrt(np.mean((y - e[:, 1]) ** 2)))
        c.append(np.mean((y >= e[:, 0]) & (y <= e[:, 2])))
        w.append(np.mean(e[:, 2] - e[:, 0]))
    grow.append({"n": n, "rmse": round(float(np.mean(r)), 2), "coverage80": round(float(np.mean(c)), 3), "width80": round(float(np.mean(w)), 2)})

calib = {"quantile": QS, "empirical": [round(float(np.mean(y <= ens[:, i])), 3) for i in range(3)]}

# ---- sample of test observations with all member predictions --------------------------
m_test = meta.iloc[rows].reset_index(drop=True)
pick = np.sort(rng.choice(len(rows), N_SAMPLE, replace=False))
sample = []
for i in pick:
    sample.append({
        "id": int(rows[i]), "cruise": m_test.expocode[i],
        "date": str(m_test.time[i])[:10], "lat": round(float(m_test.lat[i]), 2), "lon": round(float(m_test.lon[i]), 2),
        "depth": round(float(m_test.depth[i]), 0), "observed": round(float(y[i]), 1),
        "pred": np.round(P[:, i, :], 1).tolist(),  # [member][quantile]
    })

out = {
    "run": {"name": RUN.name, "n_members": len(members), "n_splits": int(sp_idx.max()), "n_replicates": 7,
            "n_total": int(len(meta)), "n_cruises": int(meta.expocode.nunique()), "test_observations": int(len(rows)),
            "years": [1993, 2021], "sample_size": N_SAMPLE},
    "funnel": [{"step": s, "rows": int(n), "cruises": int(c)} for s, n, c in funnel],
    "blocks": blocks_info,
    "members": members,
    "ensemble_scores": metrics["scores"],
    "variance_means": metrics["variance_means"],
    "spread_vs_error": spread, "growth": grow, "calibration": calib,
    "test_sample": sample, "member_order": names,
}
(Path(__file__).parent / "data.json").write_text(json.dumps(out, separators=(",", ":")))
print("wrote data.json", round((Path(__file__).parent / "data.json").stat().st_size / 1e6, 2), "MB")
print(json.dumps(out["funnel"], indent=1)); print(blocks_info); print(spread["corr_abs_err"]); print(grow)
