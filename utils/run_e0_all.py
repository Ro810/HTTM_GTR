"""E0: chay detect_cycles tren phan TRAIN cua nhieu dataset, so voi --cycle tac gia dat tay.
Luat dong bang (K=2, r_min=0.2, tol=0.05, delta=0.02) - khong chinh theo tung dataset.
Chay:  python -I utils/run_e0_all.py
"""
import csv, gzip, importlib.util, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("dc", os.path.join(HERE, "detect_cycles.py"))
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)
D = os.path.join(HERE, "..", "dataset")


def load_ett(name):
    rows = list(csv.reader(open(os.path.join(D, name + ".csv"), encoding="utf-8")))
    return np.array([[float(v) for v in r[1:]] for r in rows[1:]])


def load_electricity():
    with gzip.open(os.path.join(D, "electricity.txt.gz"), "rt") as f:
        return np.loadtxt(f, delimiter=",")


# (ten, loader, so dong train theo data_loader.py, L tac gia dat tay trong scripts/GTR)
CASES = [
    ("ETTh1", lambda: load_ett("ETTh1"), 12 * 30 * 24, 24),
    ("ETTh2", lambda: load_ett("ETTh2"), 12 * 30 * 24, 24),
    ("ETTm1", lambda: load_ett("ETTm1"), 12 * 30 * 24 * 4, 96),
    ("ETTm2", lambda: load_ett("ETTm2"), 12 * 30 * 24 * 4, 96),
    ("Electricity", load_electricity, None, 168),   # None = 70% (Dataset_Custom)
]

results, curves = [], {}
for name, loader, n_train, L_author in CASES:
    data = loader()
    n_train = n_train or int(len(data) * 0.7)
    train = data[:n_train]
    res, r, R = dc.detect(train)
    cyc = res["cycles"]
    ok = L_author in cyc
    base = cyc[0] if cyc else None
    verdict = "KHOP DUNG" if cyc == [L_author] else ("CO MAT (kem ung vien khac)" if ok else "KHONG KHOP")
    print(f"{name:12s} data={data.shape} train={n_train:6d} | DACD={cyc} | tac gia L={L_author} | "
          f"{verdict} | r={[round(a, 3) for a in res['acf_at_cycles']]}")
    # top peak manh nhat (de thay co cu ky nao khac khong)
    pk = dc.find_peaks(r, k_max=min(len(r) // 3, 800))
    top = sorted(pk, key=lambda k: -r[k])[:6]
    print(f"{'':12s} top dinh ACF (k, r): {[(k, round(float(r[k]), 3)) for k in top]}")
    if base:
        ex = {m * base: round(dc.harmonic_excess(r, base, m), 3) for m in (2, 3, 4, 7)}
        print(f"{'':12s} do noi cua boi so cua {base}: {ex}")
    results.append({"dataset": name, "n_train": n_train, "author_L": L_author, "match": ok, **res})
    curves[name] = (r, cyc, L_author)

json.dump(results, open(os.path.join(HERE, "..", "cycles_all.json"), "w"), indent=1)

fig, axes = plt.subplots(len(curves), 1, figsize=(9, 2.1 * len(curves)))
for ax, (name, (r, cyc, La)) in zip(axes, curves.items()):
    kmax = max(400, 3 * La) if La >= 96 else 400
    kmax = min(kmax, len(r) - 1)
    ax.plot(np.arange(kmax), r[:kmax], lw=1, color="#1F3A5F")
    ax.axvline(La, color="#059669", ls=":", lw=1.4, label=f"tác giả đặt L={La}")
    for c in cyc:
        ax.axvline(c, color="#DC2626", ls="--", lw=1)
    ax.set_title(f"{name}: DACD chọn {cyc}", fontsize=9, loc="left")
    ax.set_ylabel("r", fontsize=8); ax.legend(fontsize=7, loc="upper right")
axes[-1].set_xlabel("độ trễ (bước)")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "..", "..", "e0_acf_all.png"), dpi=150)
print("saved")
