# -*- coding: utf-8 -*-
"""
将 *_scores.mat 中的概率与标签导出为 CSV，便于在 Excel / Matlab 再分析。
"""
import os
import glob
import numpy as np
import scipy.io as sio
import csv

def export_mat_to_csv(results_dir="./save/MUG_RSVP"):
    mats = glob.glob(os.path.join(results_dir, "*_scores.mat"))
    for m in mats:
        name = os.path.basename(m).replace("_scores.mat","")
        D = sio.loadmat(m)
        out_csv = os.path.join(results_dir, f"{name}_scores.csv")
        with open(out_csv, "w", newline='', encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["y_true_tri","y_true_bin","p0","p1","p2","pe0","pe1","pe2","py0","py1","py2","pb0","pb1"])
            for i in range(D["probs"].shape[0]):
                row = [int(D["y"][i]), int(D["y_bin"][i])]
                row += list(np.asarray(D["probs"][i]).ravel())
                row += list(np.asarray(D["probs_eeg"][i]).ravel())
                row += list(np.asarray(D["probs_eye"][i]).ravel())
                row += list(np.asarray(D["probs_bin"][i]).ravel())
                w.writerow(row)
        print("saved:", out_csv)

if __name__ == "__main__":
    export_mat_to_csv()
