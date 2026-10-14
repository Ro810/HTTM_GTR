import importlib.util, csv, numpy as np, sys, json
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
spec = importlib.util.spec_from_file_location("dc", "D:/HTTM/GTR-main/utils/detect_cycles.py"); dc = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc)
rows = list(csv.reader(open("D:/HTTM/GTR-main/dataset/ETTh1.csv", encoding="utf-8")))
data = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
train = data[:8640]
res, r, R = dc.detect(train)
print("ETTh1 train (8640 dong), delta=0.02:", res["cycles"], [round(a,3) for a in res["acf_at_cycles"]])
# kiem tra lai tren du lieu mo phong cua demo (cung seed) -> phai ra [24,168]
rng = np.random.default_rng(2026); T=8640; t=np.arange(T)
wk=((t//24)%7>=5).astype(float); xs=np.sin(2*np.pi*t/24)*(1-0.5*wk)-1.2*wk+rng.normal(0,0.6,T)
rs, _, _ = dc.detect(xs[:,None]); print("Mo phong (co tuan that):", rs["cycles"], [round(a,3) for a in rs["acf_at_cycles"]])
# do nhay theo delta
print("\ndo nhay theo delta (ETTh1 train):")
for d in [0.0, 0.01, 0.02, 0.05]:
    print(f"  delta={d:.2f} -> {dc.detect(train, delta=d)[0]['cycles']}")
# on dinh khi chi dung 1 phan train
print("\non dinh theo do dai train (delta=0.02):")
for n in [2160, 4320, 6480, 8640]:
    print(f"  {n} dong -> {dc.detect(train[:n])[0]['cycles']}")
# luu ket qua tai lap
json.dump({"dataset":"ETTh1","n_train":8640, **res}, open("D:/HTTM/GTR-main/cycles_ETTh1.json","w"), indent=1)
# hinh
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ks = np.arange(0, 400)
for n in range(R.shape[0]): ax[0].plot(ks, R[n][:400], lw=0.6, alpha=0.55, color="#6B7280")
ax[0].plot(ks, r[:400], lw=1.6, color="#1F3A5F", label="trung bình 7 kênh")
for c in res["cycles"]:
    ax[0].axvline(c, color="#DC2626", ls="--", lw=1); ax[0].annotate(f"L={c}  r={r[c]:.2f}", (c, r[c]), xytext=(8, 6), textcoords="offset points", color="#DC2626", fontsize=9)
ax[0].axvline(168, color="#059669", ls=":", lw=1); ax[0].text(172, -0.15, "168 (tuần)", color="#059669", fontsize=8)
ax[0].set_title("ETTh1 thật: ACF trên 8.640 dòng train", fontsize=10); ax[0].set_xlabel("độ trễ (giờ)"); ax[0].set_ylabel("tương quan r"); ax[0].legend(fontsize=8, loc="upper right")
m = np.arange(2, 15); ex = [dc.harmonic_excess(r, 24, int(i)) for i in m]
ax[1].bar(m*24, ex, width=14, color=["#059669" if e >= 0.02 else "#9CA3AF" for e in ex])
ax[1].axhline(0.02, color="#DC2626", ls="--", lw=1); ax[1].text(250, 0.012, "ngưỡng δ = 0,02", color="#DC2626", fontsize=8)
ax[1].set_title("Bội số của 24 có nổi lên khỏi đường suy giảm không?", fontsize=10); ax[1].set_xlabel("độ trễ k = m×24"); ax[1].set_ylabel("độ nổi (excess)")
ax[1].set_xticks(m*24); ax[1].tick_params(axis="x", labelsize=7, rotation=60)
plt.tight_layout(); plt.savefig("D:/HTTM/etth1_acf_real.png", dpi=170)
