"""DACD khoi 1: tu dong phat hien chu ky bang ACF, chi dung tap TRAIN (numpy only).

Chay thu:  python -I utils/detect_cycles.py dataset/ETTh1.csv 8640
"""
import sys
import json
import numpy as np


def acf_fft(z):
    """ACF chuan hoa r(0)=1 cua chuoi 1 chieu, tinh bang FFT (dem 0 de tranh vong)."""
    T = len(z)
    f = np.fft.rfft(z, 2 * T)
    r = np.fft.irfft(f * np.conj(f))[:T]
    return r / r[0]


def channel_acf(train):
    """train: (T, N). z-score + khu xu huong tuyen tinh tung kenh, tra ve ACF tung kenh (N, T)."""
    T, N = train.shape
    t = np.arange(T)
    out = np.empty((N, T))
    for n in range(N):
        if train[:, n].std() < 1e-8:      # kenh hang so tren train: khong co ACF, bo qua
            out[n] = np.nan
            continue
        z = (train[:, n] - train[:, n].mean()) / train[:, n].std()
        z = z - np.polyval(np.polyfit(t, z, 1), t)
        out[n] = acf_fft(z)
    return out


def find_peaks(r, k_min=4, k_max=None, r_min=0.2, win=6):
    """Dinh cuc dai cua ACF trong [k_min, k_max], r >= r_min, cuc dai trong +-win buoc."""
    k_max = k_max or len(r) // 3
    ks = [k for k in range(k_min, min(k_max, len(r) - 1))
          if r[k] > r[k - 1] and r[k] >= r[k + 1] and r[k] >= r_min
          and r[k] == r[max(0, k - win):k + win + 1].max()]
    return ks


def harmonic_excess(r, base, m):
    """Do noi cua dinh tai m*base so voi 2 boi so ke ben. <=0: chi la hoa am cua chu ky co so."""
    k = base * m
    if k + base >= len(r):
        return -1.0
    return float(r[k] - max(r[k - base], r[k + base]))


def refine_comb(r, base, M=5, tol=0.07):
    """v4: tinh chinh chu ky co so = k trong +-tol*base cuc dai trung binh ACF tai k, 2k, ..., Mk.
    Dinh don le co the lech vai buoc do nhieu (Weather: 141 thay vi 144); lech se cong don qua cac boi so."""
    cand = range(max(2, int(base * (1 - tol))), int(base * (1 + tol)) + 1)
    comb = {k: np.mean([r[m * k] for m in range(1, M + 1) if m * k < len(r)]) for k in cand}
    return max(comb, key=comb.get)


def select_cycles(r, K=2, r_min=0.2, tol=0.05, k_max=None, delta=0.02, nested=False, refine=False):
    """Chu ky co so = dinh nho nhat du manh. Them dinh khac (cao nhat truoc) neu:
    - khong nam trong +-tol cua ung vien da chon, va
    - neu la boi so nguyen cua chu ky co so thi phai NOI LEN >= delta so voi cac boi so ke ben
      (tranh chon 48, 72... chi vi ACF suy giam cham).
    nested=True (v2): chu ky phu BAT BUOC la boi so nguyen cua chu ky co so (ngay -> tuan -> nam),
      loai cac dinh kieu 336 = 3,5 ngay tren ETTm2.
    refine=True (v4): tinh chinh chu ky co so bang refine_comb truoc khi chon chu ky phu."""
    peaks = find_peaks(r, r_min=r_min, k_max=k_max)
    if not peaks:
        return []
    base = refine_comb(r, min(peaks)) if refine else min(peaks)
    chosen = [base]
    for k in sorted(peaks, key=lambda k: -r[k]):
        if len(chosen) >= K:
            break
        if not all(abs(k - c) / c > tol for c in chosen):
            continue
        m = round(k / base)
        if nested and (m < 2 or abs(k - m * base) > tol * k):
            continue
        if m >= 2 and abs(k - m * base) <= tol * k and harmonic_excess(r, base, m) < delta:
            continue
        chosen.append(k)
    return sorted(chosen)


def detect(train, K=2, r_min=0.2, tol=0.05, delta=0.02, nested=False, refine=False):
    """train (T, N) -> dict ket qua (de luu JSON, tai lap)."""
    R = channel_acf(train)
    r = np.nanmean(R, axis=0)
    cyc = select_cycles(r, K=K, r_min=r_min, tol=tol, delta=delta, nested=nested, refine=refine)
    return {"cycles": cyc, "acf_at_cycles": [float(r[c]) for c in cyc],
            "params": {"K": K, "r_min": r_min, "tol": tol, "delta": delta, "nested": nested, "refine": refine, "k_max": len(r) // 3}}, r, R


if __name__ == "__main__":
    import csv
    path, n_train = sys.argv[1], int(sys.argv[2])
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    data = np.array([[float(v) for v in row[1:]] for row in rows[1:]])
    delta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02   # tuy chon: thu nguong khac
    res, r, R = detect(data[:n_train], delta=delta)
    print(json.dumps(res, indent=1))
