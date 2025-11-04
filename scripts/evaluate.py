# -*- coding: utf-8 -*-
"""
汇总 ./save/MUG_RSVP 下的 *_metrics.json，输出 summary.csv / summary.json
"""
import os
import json
import glob
import numpy as np
import csv

def collect_metrics(results_dir):
    rows = []
    for path in glob.glob(os.path.join(results_dir, "*_metrics.json")):
        name = os.path.basename(path).replace("_metrics.json","")
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        rows.append({"tag": name, "acc": m["acc"], "ba": m["ba"]})
    return rows

def aggregate(rows):
    accs = [r["acc"] for r in rows]
    bas  = [r["ba"]  for r in rows]
    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "ba_mean": float(np.mean(bas)),  "ba_std":  float(np.std(bas))}

def main(results_dir="./save/MUG_RSVP"):
    rows = collect_metrics(results_dir)
    agg  = aggregate(rows)
    out_csv = os.path.join(results_dir, "summary.csv")
    with open(out_csv, "w", newline='', encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["tag","acc","ba"])
        for r in rows: w.writerow([r["tag"], r["acc"], r["ba"]])
        w.writerow([]); w.writerow(["acc_mean", agg["acc_mean"]])
        w.writerow(["acc_std",  agg["acc_std"]])
        w.writerow(["ba_mean",  agg["ba_mean"]])
        w.writerow(["ba_std",   agg["ba_std"]])
    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "aggregate": agg}, f, ensure_ascii=False, indent=2)
    print("saved:", out_csv)

if __name__ == "__main__":
    main()
